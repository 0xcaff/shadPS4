// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/runtime_info.h"
#include "translator.hpp"

#include <algorithm>
#include <iostream>

#include "common/io_file.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "video_core/amdgpu/pixel_format.h"

using namespace Shader;

namespace Shader::Optimization {
void ResourceTrackingPassStub(IR::Program& program, const Profile& profile);
void OrderedCountPassStub(IR::Program& program);
} // namespace Shader::Optimization

std::vector<u32> TranslateToSpirv(u64 raw_gcn_inst) {
    return TranslateToSpirv(std::span<const u64>{&raw_gcn_inst, 1});
}

std::vector<u32> TranslateToSpirv(std::span<const u64> raw_gcn_insts) {
    return TranslateToSpirv(raw_gcn_insts, ComputeTestConfig{});
}

std::vector<u32> TranslateToSpirv(std::span<const u64> raw_gcn_insts,
                                  const ComputeTestConfig& config) {
    std::array<u32, 2> store{
        0xe0700000,
        0x80000000 // buffer_store_dword v0, v0, s[0:3], 0
    };
    if (config.store_per_invocation) {
        store = {
            0xe0701000,
            0x800000ff // buffer_store_dword v0, v255, s[0:3], 0 offen
        };
    }
    Gcn::GcnCodeSlice second(store.data(), store.data() + store.size());

    Gcn::GcnDecodeContext decoder;
    std::vector<Gcn::GcnInst> instructions;
    instructions.reserve(raw_gcn_insts.size());
    for (const u64 raw_gcn_inst : raw_gcn_insts) {
        std::array<u32, 2> provided_inst{static_cast<u32>(raw_gcn_inst & 0xFFFFFFFFU),
                                         static_cast<u32>(raw_gcn_inst >> 32)};
        Gcn::GcnCodeSlice slice(provided_inst.data(), provided_inst.data() + provided_inst.size());
        instructions.push_back(decoder.decodeInstruction(slice));
    }
    Gcn::GcnInst store_inst = decoder.decodeInstruction(second);

    Shader::Info info{};
    info.hw_stage = HwStage::Compute;
    info.sw_stage = SwStage::Compute;
    info.flattened_ud_buf.resize(4);
    AmdGpu::Buffer buf = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &buf, sizeof(buf));
    info.uses_ordered_count = std::ranges::any_of(instructions, [](const Gcn::GcnInst& inst) {
        return inst.opcode == Gcn::Opcode::DS_ORDERED_COUNT;
    });

    IR::Program program{info};
    Pools pools{};

    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(block);

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(HwStage::Compute, SwStage::Compute);
    runtime_info.props.num_user_data = 4;
    runtime_info.hw.cs.workgroup_size = config.workgroup_size;
    runtime_info.hw.cs.tgid_enable = config.tgid_enable;
    runtime_info.hw.cs.tg_size_enable = config.tg_size_enable;
    runtime_info.hw.cs.ordered_append = config.ordered_append;

    Gcn::Translator translator(program.info, runtime_info, profile);
    translator.EmitPrologue(block);

    for (int i = 0; i < 4; ++i) {
        // copy user data from SGPR to VGPR as (most?) instructions cannot access
        // two SGPRs
        Shader::Gcn::GcnInst mov{};
        mov.src[0].field = Shader::Gcn::OperandField::ScalarGPR;
        mov.src[0].code = i;
        mov.dst[0].field = Shader::Gcn::OperandField::VectorGPR;
        mov.dst[0].code = i;
        translator.S_MOV(mov);
    }
    if (config.local_index_in_v4) {
        IR::IREmitter ir{*block};
        ir.SetVectorReg(IR::VectorReg::V4,
                        ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex));
    }
    for (const Gcn::GcnInst& inst : instructions) {
        translator.TranslateInstruction(inst);
    }
    if (config.store_per_invocation) {
        IR::IREmitter ir{*block};
        const auto [size_x, size_y, size_z] = config.workgroup_size;
        const IR::U32 group_base = ir.IMul(ir.GetAttributeU32(IR::Attribute::WorkgroupIndex),
                                           ir.Imm32(size_x * size_y * size_z));
        const IR::U32 invocation =
            ir.IAdd(group_base, ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex));
        ir.SetVectorReg(IR::VectorReg::V255, ir.ShiftLeftLogical(invocation, ir.Imm32(2u)));
    }
    translator.TranslateInstruction(store_inst);

    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::ResourceTrackingPassStub(program, profile);
    Shader::Optimization::ConstantPropagationPass(program.blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    Shader::Optimization::OrderedCountPassStub(program);

    Backend::Bindings bindings{};

    const auto spirv = Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);

    return spirv;
}

std::vector<u32> TranslateFragmentFrontFaceToSpirv(bool front_face_all_bits) {
    Shader::Info info{};
    info.hw_stage = HwStage::Fragment;
    info.sw_stage = SwStage::Fragment;

    IR::Program program{info};
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(block);

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(HwStage::Fragment, SwStage::Fragment);
    runtime_info.hw.fs.en_flags.front_face_ena = 1;
    runtime_info.hw.fs.addr_flags.front_face_ena = 1;
    runtime_info.hw.fs.front_face_all_bits = front_face_all_bits;
    runtime_info.hw.fs.color_buffers[0].num_format = AmdGpu::NumberFormat::Float;

    Gcn::Translator translator(program.info, runtime_info, profile);
    translator.EmitPrologue(block);

    IR::IREmitter ir{*block};
    const IR::U32 front_face = ir.GetVectorReg<IR::U32>(IR::VectorReg::V0);
    ir.SetAttribute(IR::Attribute::RenderTarget0, ir.BitCast<IR::F32>(front_face));
    ir.Epilogue();

    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::CollectShaderInfoPass(program, profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

std::vector<u32> TranslateFragmentPullModelToSpirv(bool use_amd_barycentrics) {
    Shader::Info info{};
    info.hw_stage = HwStage::Fragment;
    info.sw_stage = SwStage::Fragment;

    IR::Program program{info};
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(block);

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.supports_amd_shader_explicit_vertex_parameter = use_amd_barycentrics;
    profile.supports_fragment_shader_barycentric = !use_amd_barycentrics;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(HwStage::Fragment, SwStage::Fragment);
    runtime_info.hw.fs.addr_flags.persp_pull_model_ena = 1;
    runtime_info.hw.fs.color_buffers[0].num_format = AmdGpu::NumberFormat::Float;

    IR::IREmitter ir{*block};
    ir.Prologue();
    IR::F32 sum = ir.Imm32(0.0f);
    for (u32 comp = 0; comp < 3; ++comp) {
        sum = ir.FPAdd(sum, ir.GetAttribute(IR::Attribute::BaryCoordPullModel, comp));
    }
    ir.SetAttribute(IR::Attribute::RenderTarget0, sum);
    ir.Epilogue();

    Optimization::CollectShaderInfoPass(program, profile);
    Backend::Bindings bindings{};
    return Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
}

FragmentInterpMovInfo TranslateFragmentInterpMovSelector(u32 src_select, bool flat_shade,
                                                         bool offset5) {
    Shader::Info info{};
    info.hw_stage = HwStage::Fragment;
    info.sw_stage = SwStage::Fragment;

    IR::Program program{info};
    Pools pools{};
    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    Profile profile{};
    profile.supports_fragment_shader_barycentric = true;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(HwStage::Fragment, SwStage::Fragment);
    runtime_info.hw.fs.inputs[0].is_flat = flat_shade;
    runtime_info.hw.fs.inputs[0].is_default = offset5;

    Gcn::Translator translator(program.info, runtime_info, profile);
    translator.EmitPrologue(block);

    Gcn::GcnInst inst{};
    inst.src[0].code = src_select;
    inst.dst[0].field = Gcn::OperandField::VectorGPR;
    inst.dst[0].code = 0;
    inst.control.vintrp.attr = 0;
    inst.control.vintrp.chan = 0;
    translator.V_INTERP_MOV_F32(inst);

    FragmentInterpMovInfo interp_info{};
    for (const IR::Inst& ir_inst : block->Instructions()) {
        if (ir_inst.GetOpcode() == IR::Opcode::GetAttribute &&
            ir_inst.Arg(0).Attribute() == IR::Attribute::Param0) {
            interp_info.attribute_indices.push_back(ir_inst.Arg(2).U32());
        } else if (ir_inst.GetOpcode() == IR::Opcode::FPSub32) {
            ++interp_info.fsub_count;
        }
    }
    return interp_info;
}
