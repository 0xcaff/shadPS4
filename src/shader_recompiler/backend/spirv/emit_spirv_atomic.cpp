// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/ir_emitter.h"

namespace Shader::Backend::SPIRV {

namespace {
using PointerType = EmitContext::PointerType;
using PointerSize = EmitContext::PointerSize;

std::pair<Id, Id> AtomicArgs(EmitContext& ctx) {
    const Id scope{ctx.ConstU32(static_cast<u32>(spv::Scope::Device))};
    const Id semantics{ctx.u32_zero_value};
    return {scope, semantics};
}

Id SharedAtomicU32(EmitContext& ctx, Id offset, Id value,
                   Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const Id shift_id{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], pointer, scope, semantics, value);
}

Id SharedAtomicU32IncDec(EmitContext& ctx, Id offset,
                         Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id)) {
    const Id shift_id{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], pointer, scope, semantics);
}

Id SharedAtomicU64(EmitContext& ctx, Id offset, Id value,
                   Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const Id shift_id{ctx.ConstU32(3U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U64, pointer, scope, semantics, value);
}

Id SharedAtomicU64IncDec(EmitContext& ctx, Id offset,
                         Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id)) {
    const Id shift_id{ctx.ConstU32(3U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U64, pointer, scope, semantics);
}

Id SharedAtomicU32CmpSwap(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    const Id shift_id{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.OpAtomicCompareExchange(ctx.U32[1], pointer, scope, semantics, semantics, value,
                                       cmp_value);
}

Id SharedAtomicU64CmpSwap(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    const Id shift_id{ctx.ConstU32(3U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer{ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.OpAtomicCompareExchange(ctx.U64, pointer, scope, semantics, semantics, value,
                                       cmp_value);
}

template <bool is_float = false>
Id BufferAtomicU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                   Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const auto& buffer = ctx.buffers[handle];
    const Id type = is_float ? ctx.F32[1] : ctx.U32[1];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(is_float ? PointerType::F32 : PointerType::U32);
    const Id ptr = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(type, ptr, scope, semantics, value);
}

Id BufferAtomicU32IncDec(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address,
                         Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id)) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id ptr = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], ptr, scope, semantics);
}

Id BufferAtomicU32CmpSwap(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          Id cmp_value,
                          Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id, Id, Id)) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id ptr = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], ptr, scope, semantics, semantics, value, cmp_value);
}

Id BufferAtomicU64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                   Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B64); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U64);
    const Id ptr = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U64, ptr, scope, semantics, value);
}

Id FixImageAtomicCoords(EmitContext& ctx, Id coords, AmdGpu::ImageType image_type) {
    switch (image_type) {
    case AmdGpu::ImageType::Color1D: {
        // Lowered to 2D with height 1
        const auto x = coords;
        return ctx.OpCompositeConstruct(ctx.U32[2], x, ctx.u32_zero_value);
    }
    case AmdGpu::ImageType::Color1DArray: {
        // Lowered to 2D array with height 1
        const auto x = ctx.OpCompositeExtract(ctx.U32[1], coords, 0U);
        const auto slice = ctx.OpCompositeExtract(ctx.U32[1], coords, 1U);
        return ctx.OpCompositeConstruct(ctx.U32[3], x, ctx.u32_zero_value, slice);
    }
    default:
        return coords;
    }
}

Id ImageAtomicU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                  Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id fixed_coords{FixImageAtomicCoords(ctx, coords, texture.view_type)};
    const Id pointer{
        ctx.OpImageTexelPointer(ctx.image_u32, texture.id, fixed_coords, ctx.ConstU32(0U))};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], pointer, scope, semantics, value);
}

Id ImageAtomicF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                  Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id)) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id fixed_coords{FixImageAtomicCoords(ctx, coords, texture.view_type)};
    const Id pointer{
        ctx.OpImageTexelPointer(ctx.image_f32, texture.id, fixed_coords, ctx.ConstU32(0U))};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.F32[1], pointer, scope, semantics, value);
}

Id ImageAtomicU32CmpSwap(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                         Id cmp_value,
                         Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id, Id, Id, Id)) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id fixed_coords{FixImageAtomicCoords(ctx, coords, texture.view_type)};
    const Id pointer{
        ctx.OpImageTexelPointer(ctx.image_u32, texture.id, fixed_coords, ctx.ConstU32(0U))};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], pointer, scope, semantics, semantics, value, cmp_value);
}
} // Anonymous namespace

Id EmitSharedAtomicIAdd32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitSharedAtomicIAdd64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitSharedAtomicUMax32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitSharedAtomicUMax64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitSharedAtomicSMax32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitSharedAtomicSMax64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitSharedAtomicUMin32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitSharedAtomicUMin64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitSharedAtomicSMin32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitSharedAtomicSMin64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitSharedAtomicAnd32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitSharedAtomicAnd64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitSharedAtomicOr32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicOr);
}

Id EmitSharedAtomicOr64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicOr);
}

Id EmitSharedAtomicXor32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicXor);
}

Id EmitSharedAtomicXor64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicXor);
}

Id EmitSharedAtomicISub32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU32(ctx, offset, value, &Sirit::Module::OpAtomicISub);
}

Id EmitSharedAtomicISub64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicISub);
}

Id EmitSharedAtomicCmpSwap32(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    return SharedAtomicU32CmpSwap(ctx, offset, value, cmp_value);
}

Id EmitSharedAtomicCmpSwap64(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    return SharedAtomicU64CmpSwap(ctx, offset, value, cmp_value);
}

Id EmitSharedAtomicInc32(EmitContext& ctx, Id offset) {
    return SharedAtomicU32IncDec(ctx, offset, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitSharedAtomicInc64(EmitContext& ctx, Id offset) {
    return SharedAtomicU64IncDec(ctx, offset, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitSharedAtomicDec32(EmitContext& ctx, Id offset) {
    return SharedAtomicU32IncDec(ctx, offset, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitSharedAtomicDec64(EmitContext& ctx, Id offset) {
    return SharedAtomicU64IncDec(ctx, offset, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitBufferAtomicIAdd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitBufferAtomicIAdd64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitBufferAtomicISub32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicISub);
}

Id EmitBufferAtomicSMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitBufferAtomicSMin64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitBufferAtomicUMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitBufferAtomicUMin64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitBufferAtomicFMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    if (ctx.profile.supports_buffer_fp32_atomic_min_max) {
        return BufferAtomicU32<true>(ctx, inst, handle, address, value,
                                     &Sirit::Module::OpAtomicFMin);
    }

    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    // OpSelect requires a bool condition; produce one by comparing the sign bit to 0.
    const auto sign_bit_set = ctx.OpINotEqual(
        ctx.U1[1],
        ctx.OpBitFieldUExtract(ctx.U32[1], u32_value, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);

    // FIXME this needs control flow because it currently executes both atomics
    const auto result = ctx.OpSelect(
        ctx.F32[1], sign_bit_set,
        EmitBitCastF32U32(ctx, EmitBufferAtomicUMax32(ctx, inst, handle, address, u32_value)),
        EmitBitCastF32U32(ctx, EmitBufferAtomicSMin32(ctx, inst, handle, address, u32_value)));

    return result;
}

Id EmitBufferAtomicSMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitBufferAtomicSMax64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitBufferAtomicUMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitBufferAtomicUMax64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitBufferAtomicFMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    if (ctx.profile.supports_buffer_fp32_atomic_min_max) {
        return BufferAtomicU32<true>(ctx, inst, handle, address, value,
                                     &Sirit::Module::OpAtomicFMax);
    }

    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    // OpSelect requires a bool condition; produce one by comparing the sign bit to 0.
    const auto sign_bit_set = ctx.OpINotEqual(
        ctx.U1[1],
        ctx.OpBitFieldUExtract(ctx.U32[1], u32_value, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);

    // FIXME this needs control flow because it currently executes both atomics
    const auto result = ctx.OpSelect(
        ctx.F32[1], sign_bit_set,
        EmitBitCastF32U32(ctx, EmitBufferAtomicUMin32(ctx, inst, handle, address, u32_value)),
        EmitBitCastF32U32(ctx, EmitBufferAtomicSMax32(ctx, inst, handle, address, u32_value)));

    return result;
}

Id EmitBufferAtomicInc32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return BufferAtomicU32IncDec(ctx, inst, handle, address, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitBufferAtomicDec32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return BufferAtomicU32IncDec(ctx, inst, handle, address, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitBufferAtomicAnd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitBufferAtomicOr32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicOr);
}

Id EmitBufferAtomicXor32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicXor);
}

Id EmitBufferAtomicSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitBufferAtomicCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                             Id cmp_value) {
    return BufferAtomicU32CmpSwap(ctx, inst, handle, address, value, cmp_value,
                                  &Sirit::Module::OpAtomicCompareExchange);
}

Id EmitBufferAtomicFCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                              Id cmp_value) {
    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    const auto u32_cmp = ctx.OpBitcast(ctx.U32[1], cmp_value);
    const auto result = BufferAtomicU32CmpSwap(ctx, inst, handle, address, u32_value, u32_cmp,
                                               &Sirit::Module::OpAtomicCompareExchange);
    return ctx.OpBitcast(ctx.F32[1], result);
}

Id EmitImageAtomicIAdd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitImageAtomicSMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitImageAtomicUMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitImageAtomicSMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitImageAtomicUMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitImageAtomicFMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    if (ctx.profile.supports_image_fp32_atomic_min_max) {
        return ImageAtomicF32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicFMax);
    }

    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    // OpSelect requires a bool condition; produce one by comparing the sign bit to 0.
    const auto sign_bit_set = ctx.OpINotEqual(
        ctx.U1[1],
        ctx.OpBitFieldUExtract(ctx.U32[1], u32_value, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);

    const auto result = ctx.OpSelect(
        ctx.F32[1], sign_bit_set,
        EmitBitCastF32U32(ctx, EmitImageAtomicUMin32(ctx, inst, handle, coords, u32_value)),
        EmitBitCastF32U32(ctx, EmitImageAtomicSMax32(ctx, inst, handle, coords, u32_value)));

    return result;
}

Id EmitImageAtomicFMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    if (ctx.profile.supports_image_fp32_atomic_min_max) {
        return ImageAtomicF32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicFMin);
    }

    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    // OpSelect requires a bool condition; produce one by comparing the sign bit to 0.
    const auto sign_bit_set = ctx.OpINotEqual(
        ctx.U1[1],
        ctx.OpBitFieldUExtract(ctx.U32[1], u32_value, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);

    const auto result = ctx.OpSelect(
        ctx.F32[1], sign_bit_set,
        EmitBitCastF32U32(ctx, EmitImageAtomicUMax32(ctx, inst, handle, coords, u32_value)),
        EmitBitCastF32U32(ctx, EmitImageAtomicSMin32(ctx, inst, handle, coords, u32_value)));

    return result;
}

Id EmitImageAtomicInc32(EmitContext&, IR::Inst*, u32, Id, Id) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitImageAtomicDec32(EmitContext&, IR::Inst*, u32, Id, Id) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitImageAtomicAnd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitImageAtomicOr32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicOr);
}

Id EmitImageAtomicXor32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicXor);
}

Id EmitImageAtomicExchange32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitImageAtomicCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                            Id cmp_value) {
    return ImageAtomicU32CmpSwap(ctx, inst, handle, coords, value, cmp_value,
                                 &Sirit::Module::OpAtomicCompareExchange);
}

Id EmitDataAppend(EmitContext& ctx, Id gds_dw_offset, Id exec) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitDataConsume(EmitContext& ctx, Id gds_dw_offset, Id exec) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitGdsOrderedCount(EmitContext& ctx, Id gds_dw_offset, Id value, Id wave) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

void EmitGdsOrderedSignal(EmitContext& ctx, Id wave) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

namespace {
Id OrderedTicket(EmitContext& ctx, u32 handle, u32 counter) {
    const auto [id, pointer_type] = ctx.buffers[handle].Alias(PointerType::U32);
    return ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value,
                             ctx.ConstU32(GdsOrderedTicketOffset / 4u + counter));
}

Id ScopeId(EmitContext& ctx, spv::Scope scope) {
    return ctx.ConstU32(static_cast<u32>(scope));
}

Id SemanticsId(EmitContext& ctx, spv::MemorySemanticsMask semantics) {
    return ctx.ConstU32(static_cast<u32>(semantics));
}

Id IsFirstActiveLane(EmitContext& ctx) {
    const Id subgroup = ScopeId(ctx, spv::Scope::Subgroup);
    const Id active = ctx.OpGroupNonUniformBallot(ctx.U32[4], subgroup, ctx.true_value);
    const Id first_lane = ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], subgroup, active);
    return ctx.OpIEqual(ctx.U1[1], first_lane,
                        ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id));
}

template <typename Func>
void SpinUntil(EmitContext& ctx, Func&& is_ready) {
    const Id loop_label = ctx.OpLabel();
    const Id body_label = ctx.OpLabel();
    const Id continue_label = ctx.OpLabel();
    const Id ready_label = ctx.OpLabel();
    ctx.OpBranch(loop_label);
    ctx.AddLabel(loop_label);
    ctx.OpLoopMerge(ready_label, continue_label, spv::LoopControlMask::MaskNone);
    ctx.OpBranch(body_label);
    ctx.AddLabel(body_label);
    ctx.OpBranchConditional(is_ready(), ready_label, continue_label);
    ctx.AddLabel(continue_label);
    ctx.OpBranch(loop_label);
    ctx.AddLabel(ready_label);
}

void WaitForTurn(EmitContext& ctx, Id ticket, Id wave) {
    const Id device = ScopeId(ctx, spv::Scope::Device);
    const Id acquire = SemanticsId(ctx, spv::MemorySemanticsMask::Acquire |
                                            spv::MemorySemanticsMask::UniformMemory);
    SpinUntil(ctx, [&] {
        const Id current = ctx.OpAtomicLoad(ctx.U32[1], ticket, device, acquire);
        return ctx.OpUGreaterThanEqual(ctx.U1[1], current, wave);
    });
}

void PassTurn(EmitContext& ctx, Id ticket, Id wave) {
    const Id acquire_release = SemanticsId(ctx, spv::MemorySemanticsMask::AcquireRelease |
                                                    spv::MemorySemanticsMask::UniformMemory);
    const Id acquire = SemanticsId(ctx, spv::MemorySemanticsMask::Acquire |
                                            spv::MemorySemanticsMask::UniformMemory);
    ctx.OpAtomicCompareExchange(ctx.U32[1], ticket, ScopeId(ctx, spv::Scope::Device),
                                acquire_release, acquire,
                                ctx.OpIAdd(ctx.U32[1], wave, ctx.u32_one_value), wave);
}

// True when every lane of every host subgroup of the wave below `subgroup` has gone past the
// current op: it has ended, or it waits at a barrier this lane has not reached.
Id LowerSubgroupsPassed(EmitContext& ctx, Id local_wave, Id subgroup) {
    const Id workgroup = ScopeId(ctx, spv::Scope::Workgroup);
    const Id acquire = SemanticsId(ctx, spv::MemorySemanticsMask::Acquire |
                                            spv::MemorySemanticsMask::WorkgroupMemory);
    const u32 num_subgroups = ctx.OrderedSubgroupsPerWave();
    const Id subgroup_lanes = ctx.ConstU32(ctx.profile.subgroup_size);
    const Id next_barrier_lanes = ctx.OpIMul(
        ctx.U32[1],
        ctx.OpIAdd(ctx.U32[1], ctx.OpLoad(ctx.U32[1], ctx.ordered_barriers), ctx.u32_one_value),
        subgroup_lanes);
    Id passed = ctx.true_value;
    for (u32 lower = 0; lower + 1 < num_subgroups; ++lower) {
        const Id finished_lanes = ctx.OpAtomicLoad(
            ctx.U32[1],
            ctx.OrderedWaveWord(local_wave, ctx.ConstU32(OrderedCountFinishedLanes + lower)),
            workgroup, acquire);
        const Id barrier_lanes = ctx.OpAtomicLoad(
            ctx.U32[1],
            ctx.OrderedWaveWord(local_wave,
                                ctx.ConstU32(OrderedCountFinishedLanes + num_subgroups + lower)),
            workgroup, acquire);
        const Id lower_passed =
            ctx.OpLogicalOr(ctx.U1[1], ctx.OpIEqual(ctx.U1[1], finished_lanes, subgroup_lanes),
                            ctx.OpUGreaterThanEqual(ctx.U1[1], barrier_lanes, next_barrier_lanes));
        const Id is_lower = ctx.OpULessThan(ctx.U1[1], ctx.ConstU32(lower), subgroup);
        passed = ctx.OpLogicalAnd(
            ctx.U1[1], passed,
            ctx.OpLogicalOr(ctx.U1[1], ctx.OpLogicalNot(ctx.U1[1], is_lower), lower_passed));
    }
    return passed;
}
} // Anonymous namespace

Id EmitBufferOrderedCount(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          Id wave) {
    const u32 flags = inst->Flags<u32>();
    const bool is_swap = (flags & IR::IREmitter::OrderedCountSwap) != 0;
    const bool is_release = (flags & IR::IREmitter::OrderedCountRelease) != 0;
    const u32 counter_index = flags >> IR::IREmitter::OrderedCountCounterShift;
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id counter = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address);
    const Id ticket = OrderedTicket(ctx, handle, counter_index);
    const Id workgroup = ScopeId(ctx, spv::Scope::Workgroup);
    const Id relaxed = ctx.u32_zero_value;
    const Id acquire = SemanticsId(ctx, spv::MemorySemanticsMask::Acquire |
                                            spv::MemorySemanticsMask::WorkgroupMemory);
    const Id local_wave = ctx.OrderedLocalWave();
    const Id elected = IsFirstActiveLane(ctx);
    const Id op_label = ctx.OpLabel();
    const Id skip_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(elected, op_label, skip_label);
    ctx.AddLabel(op_label);
    const Id previous_round =
        ctx.OpAtomicIAdd(ctx.U32[1], ctx.OrderedRoundWord(), workgroup, relaxed, ctx.u32_one_value);
    const Id round = ctx.OpIAdd(ctx.U32[1], previous_round, ctx.u32_one_value);
    const Id result_slot =
        ctx.OpBitwiseAnd(ctx.U32[1], round, ctx.ConstU32(OrderedCountResults - 1));
    const Id result_word = ctx.OrderedWaveWord(
        local_wave,
        ctx.OpIAdd(ctx.U32[1], ctx.ConstU32(static_cast<u32>(OrderedCountWord::Results)),
                   result_slot));
    const Id posted_word = ctx.OrderedWaveWord(local_wave, OrderedCountWord::Posted);
    // The value of the wave's first active lane is added, so the lowest host subgroup of the wave
    // that reaches a round runs it. A higher one takes that result, or runs the round itself once
    // every lower subgroup of the wave has ended without reaching it.
    Id runs = ctx.true_value;
    if (ctx.OrderedSubgroupsPerWave() > 1) {
        const Id subgroup = ctx.OrderedWaveSubgroup();
        SpinUntil(ctx, [&] {
            const Id posted = ctx.OpAtomicLoad(ctx.U32[1], posted_word, workgroup, acquire);
            return ctx.OpLogicalOr(ctx.U1[1], ctx.OpUGreaterThanEqual(ctx.U1[1], posted, round),
                                   LowerSubgroupsPassed(ctx, local_wave, subgroup));
        });
        const Id posted = ctx.OpAtomicLoad(ctx.U32[1], posted_word, workgroup, acquire);
        runs = ctx.OpULessThan(ctx.U1[1], posted, round);
    }
    const Id run_label = ctx.OpLabel();
    const Id wait_label = ctx.OpLabel();
    const Id done_label = ctx.OpLabel();
    ctx.OpSelectionMerge(done_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(runs, run_label, wait_label);
    ctx.AddLabel(run_label);
    WaitForTurn(ctx, ticket, wave);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    const Id result = is_swap ? ctx.OpAtomicExchange(ctx.U32[1], counter, scope, semantics, value)
                              : ctx.OpAtomicIAdd(ctx.U32[1], counter, scope, semantics, value);
    ctx.OpAtomicStore(result_word, workgroup, relaxed, result);
    if (is_release) {
        ctx.OpAtomicOr(ctx.U32[1], ctx.OrderedWaveWord(local_wave, OrderedCountWord::Released),
                       workgroup, relaxed, ctx.ConstU32(1u << counter_index));
        PassTurn(ctx, ticket, wave);
    }
    ctx.OpAtomicUMax(ctx.U32[1], posted_word, workgroup,
                     SemanticsId(ctx, spv::MemorySemanticsMask::Release |
                                          spv::MemorySemanticsMask::WorkgroupMemory),
                     round);
    const Id run_end_label = ctx.last_label;
    ctx.OpBranch(done_label);
    ctx.AddLabel(wait_label);
    const Id posted_result = ctx.OpAtomicLoad(ctx.U32[1], result_word, workgroup, relaxed);
    const Id wait_end_label = ctx.last_label;
    ctx.OpBranch(done_label);
    ctx.AddLabel(done_label);
    const Id wave_result =
        ctx.OpPhi(ctx.U32[1], result, run_end_label, posted_result, wait_end_label);
    ctx.OpBranch(merge_label);
    ctx.AddLabel(skip_label);
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
    const Id lane_result =
        ctx.OpPhi(ctx.U32[1], wave_result, done_label, ctx.u32_zero_value, skip_label);
    return ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], ScopeId(ctx, spv::Scope::Subgroup),
                                               lane_result);
}

void EmitBufferOrderedSignal(EmitContext& ctx, u32 handle, Id wave) {
    const Id workgroup = ScopeId(ctx, spv::Scope::Workgroup);
    const Id acquire_release = SemanticsId(ctx, spv::MemorySemanticsMask::AcquireRelease |
                                                    spv::MemorySemanticsMask::UniformMemory |
                                                    spv::MemorySemanticsMask::WorkgroupMemory);
    const Id local_wave = ctx.OrderedLocalWave();
    if (ctx.OrderedSubgroupsPerWave() > 1) {
        const Id finished_lanes = ctx.OrderedWaveWord(
            local_wave, ctx.OpIAdd(ctx.U32[1], ctx.ConstU32(OrderedCountFinishedLanes),
                                   ctx.OrderedWaveSubgroup()));
        ctx.OpAtomicIAdd(ctx.U32[1], finished_lanes, workgroup, acquire_release, ctx.u32_one_value);
    }
    const Id finished =
        ctx.OpAtomicIAdd(ctx.U32[1], ctx.OrderedWaveWord(local_wave, OrderedCountWord::Finished),
                         workgroup, acquire_release, ctx.u32_one_value);
    const Id is_last = ctx.OpIEqual(ctx.U1[1], ctx.OpIAdd(ctx.U32[1], finished, ctx.u32_one_value),
                                    ctx.OrderedWaveLanes(local_wave));
    const Id pass_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(is_last, pass_label, merge_label);
    ctx.AddLabel(pass_label);
    // The last lane of a wave passes on the turn of every counter the wave has not released.
    const Id released =
        ctx.OpAtomicLoad(ctx.U32[1], ctx.OrderedWaveWord(local_wave, OrderedCountWord::Released),
                         workgroup, ctx.u32_zero_value);
    for (u32 counter_index = 0; counter_index < GdsOrderedCounters; ++counter_index) {
        const u32 counter_bit = 1u << counter_index;
        if ((ctx.info.ordered_count_counters & counter_bit) == 0) {
            continue;
        }
        const Id unreleased = ctx.OpIEqual(
            ctx.U1[1], ctx.OpBitwiseAnd(ctx.U32[1], released, ctx.ConstU32(counter_bit)),
            ctx.u32_zero_value);
        const Id release_label = ctx.OpLabel();
        const Id next_label = ctx.OpLabel();
        ctx.OpSelectionMerge(next_label, spv::SelectionControlMask::MaskNone);
        ctx.OpBranchConditional(unreleased, release_label, next_label);
        ctx.AddLabel(release_label);
        const Id ticket = OrderedTicket(ctx, handle, counter_index);
        WaitForTurn(ctx, ticket, wave);
        PassTurn(ctx, ticket, wave);
        ctx.OpBranch(next_label);
        ctx.AddLabel(next_label);
    }
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

} // namespace Shader::Backend::SPIRV
