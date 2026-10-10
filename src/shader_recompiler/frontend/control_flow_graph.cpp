// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/ir/condition.h"

namespace Shader::Gcn {

struct Compare {
    bool operator()(const Block& lhs, u32 rhs) const noexcept {
        return lhs.begin < rhs;
    }

    bool operator()(u32 lhs, const Block& rhs) const noexcept {
        return lhs < rhs.begin || (lhs == rhs.begin && rhs.dispatch_order != 0);
    }

    bool operator()(const Block& lhs, const Block& rhs) const noexcept {
        return lhs < rhs;
    }
};

static IR::Condition MakeCondition(const GcnInst& inst) {
    if (inst.IsCmpx()) {
        return IR::Condition::Execnz;
    }

    switch (inst.opcode) {
    case Opcode::S_CBRANCH_SCC0:
        return IR::Condition::Scc0;
    case Opcode::S_CBRANCH_SCC1:
        return IR::Condition::Scc1;
    case Opcode::S_CBRANCH_VCCZ:
        return IR::Condition::Vccz;
    case Opcode::S_CBRANCH_VCCNZ:
        return IR::Condition::Vccnz;
    case Opcode::S_CBRANCH_EXECZ:
        return IR::Condition::Execz;
    case Opcode::S_CBRANCH_EXECNZ:
        return IR::Condition::Execnz;
    default:
        return IR::Condition::True;
    }
}

static bool IgnoresExecMask(const GcnInst& inst) {
    // EXEC mask does not affect scalar instructions or branches.
    switch (inst.category) {
    case InstCategory::ScalarALU:
    case InstCategory::ScalarMemory:
    case InstCategory::FlowControl:
        return true;
    default:
        break;
    }
    // Read/Write Lane instructions are not affected either.
    switch (inst.opcode) {
    case Opcode::V_READLANE_B32:
    case Opcode::V_WRITELANE_B32:
    case Opcode::V_READFIRSTLANE_B32:
        return true;
    default:
        break;
    }
    return false;
}

static std::optional<u32> ResolveSetPcTarget(std::span<const GcnInst> list, u32 setpc_index,
                                             std::span<const u32> pc_map) {
    if (setpc_index < 3) {
        return std::nullopt;
    }

    const auto& getpc = list[setpc_index - 3];
    const auto& arith = list[setpc_index - 2];
    const auto& setpc = list[setpc_index];

    if (getpc.opcode != Opcode::S_GETPC_B64 ||
        !(arith.opcode == Opcode::S_ADD_U32 || arith.opcode == Opcode::S_SUB_U32) ||
        setpc.opcode != Opcode::S_SETPC_B64)
        return std::nullopt;

    if (getpc.dst[0].code != setpc.src[0].code || arith.dst[0].code != setpc.src[0].code)
        return std::nullopt;

    if (arith.src_count < 2 || arith.src[1].field != OperandField::LiteralConst)
        return std::nullopt;

    const u32 imm = arith.src[1].code;

    const s32 signed_offset =
        (arith.opcode == Opcode::S_ADD_U32) ? static_cast<s32>(imm) : -static_cast<s32>(imm);

    const u32 base_pc = pc_map[setpc_index - 3] + getpc.length;

    const u32 result_pc = static_cast<u32>(static_cast<s32>(base_pc) + signed_offset);
    LOG_DEBUG(Render_Recompiler, "SetPC target: {} + {} = {}", base_pc, signed_offset, result_pc);
    return result_pc & ~0x3u;
}

static constexpr size_t LabelReserveSize = 32;

static std::optional<u32> Immediate(const InstOperand& operand) {
    switch (operand.field) {
    case OperandField::LiteralConst:
        return operand.code;
    case OperandField::ConstZero:
        return 0;
    case OperandField::SignedConstIntPos:
        return operand.code - 128;
    case OperandField::SignedConstIntNeg:
        return 192 - operand.code;
    default:
        return std::nullopt;
    }
}

static bool WritesScalar(const GcnInst& inst, u32 reg) {
    for (u32 i = 0; i < inst.dst_count; ++i) {
        const auto& dst = inst.dst[i];
        if (dst.field == OperandField::VectorGPR) {
            continue;
        }
        const u32 count =
            inst.category == InstCategory::ScalarMemory
                ? inst.control.smrd.count
                : (dst.type == ScalarType::Uint64 || dst.type == ScalarType::Sint64 ? 2 : 1);
        if (reg >= dst.code && reg - dst.code < count) {
            return true;
        }
    }
    return false;
}

static std::optional<u32> FindScalarWrite(std::span<const GcnInst> list, u32 before, u32 reg) {
    while (before > 0) {
        const auto& inst = list[--before];
        if (inst.IsTerminateInstruction()) {
            return std::nullopt;
        }
        if (WritesScalar(inst, reg)) {
            return before;
        }
    }
    return std::nullopt;
}

std::optional<CFG::JumpTable> CFG::ResolveJumpTable(u32 setpc_index) const {
    if (setpc_index < 3) {
        return std::nullopt;
    }
    const auto& getpc = inst_list[setpc_index - 3];
    const auto& add = inst_list[setpc_index - 2];
    const auto& carry = inst_list[setpc_index - 1];
    const u32 pc_reg = inst_list[setpc_index].src[0].code;
    if (getpc.opcode != Opcode::S_GETPC_B64 || getpc.dst[0].code != pc_reg ||
        add.opcode != Opcode::S_ADD_U32 || add.dst[0].code != pc_reg ||
        carry.opcode != Opcode::S_ADDC_U32 || carry.dst[0].code != pc_reg + 1) {
        return std::nullopt;
    }
    const auto other_source = [](const GcnInst& inst, u32 reg) -> const InstOperand* {
        if (inst.src[0].field == OperandField::ScalarGPR && inst.src[0].code == reg) {
            return &inst.src[1];
        }
        if (inst.src[1].field == OperandField::ScalarGPR && inst.src[1].code == reg) {
            return &inst.src[0];
        }
        return nullptr;
    };
    const auto* offset = other_source(add, pc_reg);
    const auto* offset_hi = other_source(carry, pc_reg + 1);
    if (!offset || !offset_hi || offset->field != OperandField::ScalarGPR ||
        offset_hi->field != OperandField::ScalarGPR || offset_hi->code != offset->code + 1 ||
        (offset->code <= pc_reg + 1 && offset_hi->code >= pc_reg)) {
        return std::nullopt;
    }
    const auto load_index = FindScalarWrite(inst_list, setpc_index - 3, offset->code);
    if (!load_index || *load_index < 3) {
        return std::nullopt;
    }
    const auto& load = inst_list[*load_index];
    if (load.opcode != Opcode::S_LOAD_DWORDX2 || load.dst[0].code != offset->code ||
        load.control.smrd.imm || load.src[1].field != OperandField::ScalarGPR) {
        return std::nullopt;
    }
    const u32 index_reg = load.control.smrd.offset;
    for (u32 i = *load_index; i < setpc_index; ++i) {
        if (WritesScalar(inst_list[i], index_reg) ||
            (i > *load_index && i < setpc_index - 3 &&
             (WritesScalar(inst_list[i], offset->code) ||
              WritesScalar(inst_list[i], offset_hi->code)))) {
            return std::nullopt;
        }
    }
    const u32 base_reg = load.src[0].code * 2;
    const auto& table_pc = inst_list[*load_index - 3];
    const auto& table_add = inst_list[*load_index - 2];
    const auto& table_carry = inst_list[*load_index - 1];
    if (table_pc.opcode != Opcode::S_GETPC_B64 || table_pc.dst[0].code != base_reg ||
        table_add.opcode != Opcode::S_ADD_U32 || table_add.dst[0].code != base_reg ||
        table_carry.opcode != Opcode::S_ADDC_U32 || table_carry.dst[0].code != base_reg + 1) {
        return std::nullopt;
    }
    const auto* table_offset = other_source(table_add, base_reg);
    const auto* table_high = other_source(table_carry, base_reg + 1);
    if (!table_offset || !table_high || !Immediate(*table_offset) || Immediate(*table_high) != 0) {
        return std::nullopt;
    }
    const auto shift_index = FindScalarWrite(inst_list, *load_index, index_reg);
    if (!shift_index) {
        return std::nullopt;
    }
    const auto& shift = inst_list[*shift_index];
    if (shift.opcode != Opcode::S_LSHL_B32 || Immediate(shift.src[1]) != 3) {
        return std::nullopt;
    }
    const auto clamp_index = FindScalarWrite(inst_list, *shift_index, shift.src[0].code);
    if (!clamp_index) {
        return std::nullopt;
    }
    const auto& clamp = inst_list[*clamp_index];
    if (clamp.opcode != Opcode::S_MIN_U32) {
        return std::nullopt;
    }
    auto bound = Immediate(clamp.src[0]);
    if (!bound) {
        bound = Immediate(clamp.src[1]);
    }
    if (!bound || *bound >= 256) {
        return std::nullopt;
    }
    const u32 table_address =
        index_to_pc[*load_index - 3] + table_pc.length + *Immediate(*table_offset);
    const u32 branch_base = index_to_pc[setpc_index - 3] + getpc.length;
    JumpTable result{index_reg, {}, index_to_pc[*load_index + 1], {}};
    for (u32 i = 0; i <= *bound; ++i) {
        u64 relative;
        std::memcpy(&relative, reinterpret_cast<const void*>(pgm_base + table_address + i * 8),
                    sizeof(relative));
        const u64 target = (branch_base + relative) & ~u64{3};
        if (target >= index_to_pc.back() ||
            !std::ranges::binary_search(index_to_pc, static_cast<u32>(target))) {
            return std::nullopt;
        }
        result.targets.push_back(static_cast<u32>(target));
        result.offsets.push_back(relative);
    }
    LOG_INFO(Render_Recompiler, "Resolved jump table at {:#x}: {} entries", table_address,
             result.targets.size());
    return result;
}

CFG::CFG(Common::ObjectPool<Block>& block_pool_, std::span<const GcnInst> inst_list_,
         VAddr pgm_base_)
    : block_pool{block_pool_}, inst_list{inst_list_}, pgm_base{pgm_base_} {
    index_to_pc.resize(inst_list.size() + 1);
    for (u32 i = 0; i < inst_list.size(); ++i) {
        index_to_pc[i + 1] = index_to_pc[i] + inst_list[i].length;
    }
    labels.reserve(LabelReserveSize);
    EmitLabels();
    EmitBlocks();
    LinkBlocks();
    SplitDivergenceScopes();
    RemoveUnreachableBlocks();
}

void CFG::EmitLabels() {
    // Always set a label at entry point.
    u32 pc = 0;
    AddLabel(pc);

    // Iterate instruction list and add labels to branch targets.
    for (u32 i = 0; i < inst_list.size(); i++) {
        index_to_pc[i] = pc;
        const GcnInst inst = inst_list[i];
        if (inst.IsUnconditionalBranch()) {
            u32 target = inst.BranchTarget(pc);
            if (inst.opcode == Opcode::S_SETPC_B64) {
                if (auto t = ResolveSetPcTarget(inst_list, i, index_to_pc)) {
                    target = *t;
                } else if (auto table = ResolveJumpTable(i)) {
                    for (const u32 address : table->targets) {
                        AddLabel(address);
                    }
                    jump_tables.emplace(i, std::move(*table));
                    AddLabel(pc + inst.length);
                    pc += inst.length;
                    continue;
                } else {
                    ASSERT_MSG(
                        false,
                        "S_SETPC_B64 without a resolvable offset at PC {:#x} (Index {}): Involved "
                        "instructions not recognized or invalid pattern",
                        pc, i);
                }
            }
            AddLabel(target);
            // Emit this label so that the block ends with the branching instruction
            AddLabel(pc + inst.length);
        } else if (inst.IsConditionalBranch()) {
            const u32 true_label = inst.BranchTarget(pc);
            const u32 false_label = pc + inst.length;
            if (true_label != false_label) {
                AddLabel(true_label);
                AddLabel(false_label);
            }
        } else if (inst.opcode == Opcode::S_ENDPGM) {
            const u32 next_label = pc + inst.length;
            AddLabel(next_label);
        }

        pc += inst.length;
    }
    index_to_pc[inst_list.size()] = pc;

    // Sort labels to make sure block insertion is correct.
    std::ranges::sort(labels);
}

void CFG::SplitDivergenceScopes() {
    const auto is_open_scope = [](const GcnInst& inst) {
        // An open scope instruction is an instruction that modifies EXEC
        // but also saves the previous value to restore later. This indicates
        // we are entering a scope.
        return inst.opcode == Opcode::S_AND_SAVEEXEC_B64 ||
               // While this instruction does not save EXEC it is often used paired
               // with SAVEEXEC to mask the threads that didn't pass the condition
               // of initial branch.
               (inst.opcode == Opcode::S_ANDN2_B64 && inst.dst[0].field == OperandField::ExecLo) ||
               inst.IsCmpx();
    };
    const auto is_close_scope = [](const GcnInst& inst) {
        // Closing an EXEC scope can be either a branch instruction
        // (typical case when S_AND_SAVEEXEC_B64 is right before a branch)
        // or by a move instruction to EXEC that restores the backup.
        return (inst.opcode == Opcode::S_MOV_B64 && inst.dst[0].field == OperandField::ExecLo) ||
               // Sometimes compiler might insert instructions between the SAVEEXEC and the branch.
               // Those instructions need to be wrapped in the condition as well so allow branch
               // as end scope instruction.
               inst.opcode == Opcode::S_CBRANCH_EXECZ || inst.opcode == Opcode::S_ENDPGM ||
               (inst.opcode == Opcode::S_ANDN2_B64 && inst.dst[0].field == OperandField::ExecLo);
    };
    const auto is_split_scope = [](const GcnInst& inst) {
        // These move a value between a lane of a VGPR and an SGPR whatever the exec mask: a read
        // gives the SGPR to every lane of the wave, a write sets its lane even when that lane's
        // exec bit is clear. Inside a divergence block only the lanes with their exec bit set would
        // run them, so they end the block, run in every lane, and the scope goes on after them.
        return inst.opcode == Opcode::V_READFIRSTLANE_B32 ||
               inst.opcode == Opcode::V_READLANE_B32 || inst.opcode == Opcode::V_WRITELANE_B32;
    };

    // An epilogue block that starts inside a scope a split ended; its first instructions are still
    // under the scope's exec mask.
    const Block* reopen_block = nullptr;
    for (auto blk = blocks.begin(); blk != blocks.end(); blk++) {
        if (blk->indirect_condition) {
            continue;
        }
        auto next_blk = std::next(blk);
        s32 curr_begin = -1;
        if (&*blk == reopen_block) {
            curr_begin = static_cast<s32>(blk->begin_index) - 1;
            reopen_block = nullptr;
        }
        for (size_t index = blk->begin_index; index <= blk->end_index; index++) {
            const auto& inst = inst_list[index];
            const bool is_close = is_close_scope(inst);
            const bool is_split = !is_close && curr_begin != -1 && is_split_scope(inst);
            // A close or split instruction is not part of the scope it ends.
            const bool ends_scope = is_close || is_split;
            if ((ends_scope || index == blk->end_index) && curr_begin != -1) {
                // If there are no instructions inside scope don't do anything.
                if (index - curr_begin == 1 && ends_scope) {
                    curr_begin = (is_split || is_open_scope(inst)) ? static_cast<s32>(index) : -1;
                    continue;
                }
                // If all instructions in the scope ignore exec masking, we shouldn't insert a
                // scope.
                const auto start = inst_list.begin() + curr_begin + 1;
                if (!std::ranges::all_of(start, inst_list.begin() + index + !ends_scope,
                                         IgnoresExecMask)) {
                    // Determine the first instruction affected by the exec mask.
                    do {
                        ++curr_begin;
                    } while (IgnoresExecMask(inst_list[curr_begin]));

                    // Determine the last instruction affected by the exec mask.
                    s32 curr_end = index;
                    while (IgnoresExecMask(inst_list[curr_end])) {
                        --curr_end;
                    }

                    // Create a new block for the divergence scope.
                    Block* block = block_pool.Create();
                    block->begin = index_to_pc[curr_begin];
                    block->end = index_to_pc[curr_end];
                    block->begin_index = curr_begin;
                    block->end_index = curr_end;
                    block->end_inst = inst_list[curr_end];
                    block->num_predecessors = 1;
                    blocks.insert_before(next_blk, *block);

                    // If we are inside the parent block, make an epilogue block and jump to it.
                    if (curr_end != blk->end_index) {
                        Block* epi_block = block_pool.Create();
                        epi_block->begin = index_to_pc[curr_end + 1];
                        epi_block->end = blk->end;
                        epi_block->begin_index = curr_end + 1;
                        epi_block->end_index = blk->end_index;
                        epi_block->end_inst = blk->end_inst;
                        epi_block->cond = blk->cond;
                        epi_block->end_class = blk->end_class;
                        epi_block->branch_true = blk->branch_true;
                        epi_block->branch_false = blk->branch_false;
                        epi_block->num_predecessors = 2;
                        blocks.insert_before(next_blk, *epi_block);

                        // Have divergence block always jump to epilogue block.
                        block->cond = IR::Condition::True;
                        block->branch_true = epi_block;
                        block->branch_false = nullptr;

                        // If the parent block fails to enter divergence block make it jump to
                        // epilogue too
                        blk->branch_false = epi_block;

                        // After a split the rest of the scope is in the epilogue.
                        if (is_split) {
                            reopen_block = epi_block;
                        }
                    } else {
                        // No epilogue block is needed since the divergence block
                        // also ends the parent block. Inherit the end condition.
                        auto& parent_blk = *blk;
                        ASSERT(blk->cond == IR::Condition::True && blk->branch_true);
                        block->cond = IR::Condition::True;
                        block->branch_true = blk->branch_true;
                        block->branch_false = nullptr;

                        // If the parent block didn't enter the divergence scope
                        // have it jump directly to the next one
                        blk->branch_false = blk->branch_true;
                        blk->branch_true->num_predecessors++;
                    }

                    // Shrink parent block to end right before curr_begin
                    // and make it jump to divergence block
                    --curr_begin;
                    blk->end = index_to_pc[curr_begin];
                    blk->end_index = curr_begin;
                    blk->end_inst = inst_list[curr_begin];
                    blk->cond = IR::Condition::Execnz;
                    blk->end_class = EndClass::Branch;
                    blk->branch_true = block;
                }
                // Reset scope begin, or after a split go on with the scope.
                curr_begin = is_split ? static_cast<s32>(index) : -1;
            }
            // Mark a potential start of an exec scope.
            if (is_open_scope(inst)) {
                curr_begin = index;
            }
        }
    }
}

void CFG::EmitBlocks() {
    for (auto it = labels.cbegin(); it != labels.cend(); ++it) {
        const Label start = *it;
        const auto next_it = std::next(it);
        const bool is_last = (next_it == labels.cend());
        if (is_last) {
            // Last label is special.
            return;
        }
        // The end label is the start instruction of next block.
        // The end instruction of this block is the previous one.
        const Label end = *next_it;
        const size_t end_index = GetIndex(end) - 1;
        const auto& end_inst = inst_list[end_index];

        // Insert block between the labels using the last instruction
        // as an indicator for branching type.
        Block* block = block_pool.Create();
        block->begin = start;
        block->end = end;
        block->begin_index = GetIndex(start);
        block->end_index = end_index;
        block->end_inst = end_inst;
        block->cond = MakeCondition(end_inst);
        blocks.insert(*block);
    }
}

void CFG::LinkBlocks() {
    const auto get_block = [this](u32 address) {
        auto it = blocks.find(address, Compare{});
        ASSERT_MSG(it != blocks.cend() && it->begin == address);
        return &*it;
    };

    std::vector<Block*> indirect_blocks;
    for (auto it = blocks.begin(); it != blocks.end(); it++) {
        auto& block = *it;
        const auto end_inst{block.end_inst};
        if (jump_tables.contains(block.end_index)) {
            indirect_blocks.push_back(&block);
            block.end_class = EndClass::Branch;
            continue;
        }

        // If the block doesn't end with a branch we simply
        // need to link with the next block.
        if (!end_inst.IsTerminateInstruction()) {
            auto* next_block = get_block(block.end);
            block.branch_true = next_block;
            block.end_class = EndClass::Branch;
            next_block->num_predecessors++;
            continue;
        }

        // Find the branch targets from the instruction and link the blocks.
        // Note: Block end address is one instruction after end_inst.
        const u32 branch_pc = block.end - end_inst.length;
        u32 target_pc = 0;
        if (end_inst.opcode == Opcode::S_SETPC_B64) {
            auto tgt = ResolveSetPcTarget(inst_list, block.end_index, index_to_pc);
            ASSERT_MSG(tgt,
                       "S_SETPC_B64 without a resolvable offset at PC {:#x} (Index {}): Involved "
                       "instructions not recognized or invalid pattern",
                       branch_pc, block.end_index);
            target_pc = *tgt;
        } else {
            target_pc = end_inst.BranchTarget(branch_pc);
        }

        if (end_inst.IsUnconditionalBranch()) {
            auto* target_block = get_block(target_pc);
            block.branch_true = target_block;
            block.end_class = EndClass::Branch;
            target_block->num_predecessors++;
        } else if (end_inst.IsConditionalBranch()) {
            auto* end_block = get_block(block.end);
            block.end_class = EndClass::Branch;
            if (target_pc != block.end) {
                auto* target_block = get_block(target_pc);
                block.branch_true = target_block;
                block.branch_false = end_block;
                target_block->num_predecessors++;
            } else {
                block.branch_true = end_block;
                block.cond = IR::Condition::True;
            }
            end_block->num_predecessors++;
        } else if (end_inst.opcode == Opcode::S_ENDPGM) {
            block.end_class = EndClass::Exit;
        } else {
            UNREACHABLE();
        }
    }

    for (auto* source : indirect_blocks) {
        const auto& table = jump_tables.at(source->end_index);
        Block** next = &source->branch_true;
        for (u32 i = 0; i < table.targets.size(); ++i) {
            auto* block = block_pool.Create();
            block->begin = source->end - 1;
            block->end = source->end;
            block->dispatch_order = i + 1;
            block->end_class = EndClass::Branch;
            block->cond =
                i + 1 == table.targets.size() ? IR::Condition::True : IR::Condition::Indirect;
            block->indirect_condition = Block::IndirectCondition{table.index_reg, i * 8};
            block->branch_true = get_block(table.targets[i]);
            block->branch_true->num_predecessors++;
            block->num_predecessors = 1;
            blocks.insert(*block);
            *next = block;
            next = &block->branch_false;
        }
    }
}

void CFG::RemoveUnreachableBlocks() {
    for (auto it = std::next(blocks.begin()); it != blocks.end();) {
        if (it->num_predecessors == 0) {
            LOG_WARNING(Render_Recompiler, "Removing unreachable block begin={:#x}", it->begin);
            it = blocks.erase(it);
        } else {
            it++;
        }
    }
}

std::string CFG::Dot() const {
    int node_uid{0};

    const auto name_of = [](const Block& block) { return fmt::format("\"{:#x}\"", block.begin); };

    std::string dot{"digraph shader {\n"};
    dot += fmt::format("\tsubgraph cluster_{} {{\n", 0);
    dot += fmt::format("\t\tnode [style=filled];\n");
    for (const Block& block : blocks) {
        const std::string name{name_of(block)};
        const auto add_branch = [&](Block* branch, bool add_label) {
            dot += fmt::format("\t\t{}->{}", name, name_of(*branch));
            if (add_label && block.cond != IR::Condition::True &&
                block.cond != IR::Condition::False) {
                dot += fmt::format(" [label=\"{}\"]", block.cond);
            }
            dot += '\n';
        };
        dot += fmt::format("\t\t{};\n", name);
        switch (block.end_class) {
        case EndClass::Branch:
            if (block.cond != IR::Condition::False) {
                add_branch(block.branch_true, true);
            }
            if (block.cond != IR::Condition::True) {
                add_branch(block.branch_false, false);
            }
            break;
        case EndClass::Exit:
            dot += fmt::format("\t\t{}->N{};\n", name, node_uid);
            dot +=
                fmt::format("\t\tN{} [label=\"Exit\"][shape=square][style=stripped];\n", node_uid);
            ++node_uid;
            break;
        }
    }
    dot += "\t\tlabel = \"main\";\n\t}\n";
    if (blocks.empty()) {
        dot += "Start;\n";
    } else {
        dot += fmt::format("\tStart -> {};\n", name_of(*blocks.begin()));
    }
    dot += fmt::format("\tStart [shape=diamond];\n");
    dot += "}\n";
    return dot;
}

} // namespace Shader::Gcn
