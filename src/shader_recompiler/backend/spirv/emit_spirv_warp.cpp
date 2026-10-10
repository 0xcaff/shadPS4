// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/microinstruction.h"

namespace Shader::Backend::SPIRV {

Id SubgroupScope(EmitContext& ctx) {
    return ctx.ConstU32(static_cast<u32>(spv::Scope::Subgroup));
}

Id EmitWarpId(EmitContext& ctx) {
    UNREACHABLE();
}

Id EmitLaneId(EmitContext& ctx) {
    return ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id);
}

Id EmitQuadBroadcast(EmitContext& ctx, Id value, Id index) {
    return ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, index);
}

Id EmitReadFirstLane(EmitContext& ctx, IR::Inst* inst, Id value, Id exec) {
    const Id scope{SubgroupScope(ctx)};
    if (const IR::Value exec_value{inst->Arg(1)}; exec_value.IsImmediate() && exec_value.U1()) {
        return ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], scope, value);
    }
    // The first lane with its exec bit set. Outside a divergence block every lane runs the
    // instruction, whatever its exec bit. With no bit set the PS4 reads lane 0: this reads the
    // first active lane, which is lane 0 when every lane runs, as an inactive lane has no value.
    const Id ballot{ctx.OpGroupNonUniformBallot(ctx.U32[4], scope, exec)};
    const Id any_set{
        ctx.OpINotEqual(ctx.U1[1],
                        ctx.OpBitwiseOr(ctx.U32[1], ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U),
                                        ctx.OpCompositeExtract(ctx.U32[1], ballot, 1U)),
                        ctx.u32_zero_value)};
    const Id active{ctx.OpGroupNonUniformBallot(ctx.U32[4], scope, ctx.true_value)};
    const Id exec_lane{ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], scope, ballot)};
    const Id active_lane{ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], scope, active)};
    const Id lane{ctx.OpSelect(ctx.U32[1], any_set, exec_lane, active_lane)};
    return ctx.OpGroupNonUniformBroadcast(ctx.U32[1], scope, value, lane);
}

Id EmitShuffle(EmitContext& ctx, Id value, Id index) {
    return ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, index);
}

Id EmitShuffleXor(EmitContext& ctx, Id value, Id mask) {
    return ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, mask);
}

Id EmitReadLane(EmitContext& ctx, Id value, Id lane) {
    return ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, lane);
}

Id EmitWriteLane(EmitContext& ctx, Id value, Id write_value, u32 lane) {
    // The selected lane takes the new value, every other lane keeps its own.
    const Id lane_id{ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id)};
    const Id is_lane{ctx.OpIEqual(ctx.U1[1], lane_id, ctx.ConstU32(lane))};
    return ctx.OpSelect(ctx.U32[1], is_lane, write_value, value);
}

Id EmitBallot(EmitContext& ctx, Id bit) {
    const Id ballot{ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), bit)};
    return ctx.OpBitcast(ctx.U64, ctx.OpVectorShuffle(ctx.U32[2], ballot, ballot, 0, 1));
}

Id EmitBallotFindLsb(EmitContext& ctx, Id mask) {
    const Id value{ctx.OpCompositeConstruct(ctx.U32[4], ctx.OpBitcast(ctx.U32[2], mask),
                                            ctx.u32_zero_value, ctx.u32_zero_value)};
    return ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], SubgroupScope(ctx), value);
}

Id EmitInverseBallot(EmitContext& ctx, Id mask) {
    const Id value{ctx.OpCompositeConstruct(ctx.U32[4], ctx.OpBitcast(ctx.U32[2], mask),
                                            ctx.u32_zero_value, ctx.u32_zero_value)};
    return ctx.OpGroupNonUniformInverseBallot(ctx.U1[1], SubgroupScope(ctx), value);
}

Id EmitGroupAny(EmitContext& ctx, Id bit) {
    return ctx.OpGroupNonUniformAny(ctx.U1[1], SubgroupScope(ctx), bit);
}

} // namespace Shader::Backend::SPIRV
