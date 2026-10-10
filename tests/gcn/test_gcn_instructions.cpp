// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cmath>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <half.hpp>
#include <spirv/unified1/spirv.hpp11>

#include "gcn_test_runner.hpp"
#include "instructions.hpp"
#include "shader_recompiler/info.h"
#include "translator.hpp"

class GcnTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    static void TearDownTestSuite() {
        gcn_test::Runner::DestroyInstance();
    }
};

struct F32x2 {
    float a;
    float b;
};

struct FrontFaceSpirvInfo {
    bool has_front_facing{};
    u32 select_count{};
    u32 true_value{};
    u32 false_value{};
};

FrontFaceSpirvInfo InspectFrontFaceSpirv(const std::vector<u32>& spirv) {
    FrontFaceSpirvInfo info{};
    if (spirv.size() < 5U) {
        ADD_FAILURE() << "SPIR-V header is truncated";
        return info;
    }

    std::unordered_map<u32, u32> constants;
    u32 true_value_id{};
    u32 false_value_id{};
    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 word_count = instruction >> 16;
        const auto opcode = static_cast<spv::Op>(instruction & 0xffffU);
        if (word_count == 0U || offset + word_count > spirv.size()) {
            ADD_FAILURE() << "Malformed SPIR-V instruction at word " << offset;
            return info;
        }

        if (opcode == spv::Op::OpDecorate && word_count >= 4U &&
            spirv[offset + 2] == static_cast<u32>(spv::Decoration::BuiltIn) &&
            spirv[offset + 3] == static_cast<u32>(spv::BuiltIn::FrontFacing)) {
            info.has_front_facing = true;
        } else if (opcode == spv::Op::OpConstant && word_count == 4U) {
            constants.emplace(spirv[offset + 2], spirv[offset + 3]);
        } else if (opcode == spv::Op::OpSelect && word_count == 6U) {
            ++info.select_count;
            true_value_id = spirv[offset + 4];
            false_value_id = spirv[offset + 5];
        }
        offset += word_count;
    }

    if (info.select_count == 1U && constants.contains(true_value_id) &&
        constants.contains(false_value_id)) {
        info.true_value = constants.at(true_value_id);
        info.false_value = constants.at(false_value_id);
    }
    return info;
}

struct PullModelSpirvInfo {
    u32 bary_coord_khr_count{};
    u32 frag_coord_count{};
    u32 pull_model_amd_count{};
    u32 fragment_barycentric_khr_count{};
    u32 fmul_count{};
};

PullModelSpirvInfo InspectPullModelSpirv(const std::vector<u32>& spirv) {
    PullModelSpirvInfo info{};
    if (spirv.size() < 5U) {
        ADD_FAILURE() << "SPIR-V header is truncated";
        return info;
    }

    for (size_t offset = 5; offset < spirv.size();) {
        const u32 instruction = spirv[offset];
        const u32 word_count = instruction >> 16;
        const auto opcode = static_cast<spv::Op>(instruction & 0xffffU);
        if (word_count == 0U || offset + word_count > spirv.size()) {
            ADD_FAILURE() << "Malformed SPIR-V instruction at word " << offset;
            return info;
        }

        if (opcode == spv::Op::OpCapability && word_count >= 2U &&
            spirv[offset + 1] == static_cast<u32>(spv::Capability::FragmentBarycentricKHR)) {
            ++info.fragment_barycentric_khr_count;
        } else if (opcode == spv::Op::OpDecorate && word_count >= 4U &&
                   spirv[offset + 2] == static_cast<u32>(spv::Decoration::BuiltIn)) {
            const auto builtin = static_cast<spv::BuiltIn>(spirv[offset + 3]);
            switch (builtin) {
            case spv::BuiltIn::BaryCoordKHR:
                ++info.bary_coord_khr_count;
                break;
            case spv::BuiltIn::FragCoord:
                ++info.frag_coord_count;
                break;
            case spv::BuiltIn::BaryCoordPullModelAMD:
                ++info.pull_model_amd_count;
                break;
            default:
                break;
            }
        } else if (opcode == spv::Op::OpFMul) {
            ++info.fmul_count;
        }
        offset += word_count;
    }
    return info;
}

TEST_F(GcnTest, fragment_front_face_uses_float_sign_bits) {
    const auto info = InspectFrontFaceSpirv(TranslateFragmentFrontFaceToSpirv(false));

    EXPECT_TRUE(info.has_front_facing);
    EXPECT_EQ(info.select_count, 1U);
    EXPECT_EQ(info.true_value, 0x3f800000U);
    EXPECT_EQ(info.false_value, 0xbf800000U);
}

TEST_F(GcnTest, fragment_front_face_uses_all_bits) {
    const auto info = InspectFrontFaceSpirv(TranslateFragmentFrontFaceToSpirv(true));

    EXPECT_TRUE(info.has_front_facing);
    EXPECT_EQ(info.select_count, 1U);
    EXPECT_EQ(info.true_value, 1U);
    EXPECT_EQ(info.false_value, 0U);
}

TEST_F(GcnTest, khr_barycentrics_reconstruct_pull_model) {
    const auto info = InspectPullModelSpirv(TranslateFragmentPullModelToSpirv(false));

    EXPECT_EQ(info.bary_coord_khr_count, 1U);
    EXPECT_EQ(info.frag_coord_count, 1U);
    EXPECT_EQ(info.fragment_barycentric_khr_count, 1U);
    EXPECT_EQ(info.pull_model_amd_count, 0U);
    EXPECT_EQ(info.fmul_count, 2U);
}

TEST_F(GcnTest, amd_barycentrics_use_native_pull_model) {
    const auto info = InspectPullModelSpirv(TranslateFragmentPullModelToSpirv(true));

    EXPECT_EQ(info.pull_model_amd_count, 1U);
    EXPECT_EQ(info.frag_coord_count, 0U);
    EXPECT_EQ(info.bary_coord_khr_count, 0U);
    EXPECT_EQ(info.fmul_count, 0U);
}

TEST_F(GcnTest, interp_mov_selects_p10_p20_and_p0) {
    const auto p10 = TranslateFragmentInterpMovSelector(0, false, false);
    EXPECT_EQ(p10.attribute_indices, (std::vector<u32>{1U, 0U}));
    EXPECT_EQ(p10.fsub_count, 1U);

    const auto p20 = TranslateFragmentInterpMovSelector(1, false, false);
    EXPECT_EQ(p20.attribute_indices, (std::vector<u32>{2U, 0U}));
    EXPECT_EQ(p20.fsub_count, 1U);

    const auto p0 = TranslateFragmentInterpMovSelector(2, false, false);
    EXPECT_EQ(p0.attribute_indices, (std::vector<u32>{0U}));
    EXPECT_EQ(p0.fsub_count, 0U);
}

TEST_F(GcnTest, interp_mov_uses_vertex_values_only_for_passthrough_inputs) {
    for (const auto [flat_shade, offset5] :
         {std::pair{false, false}, std::pair{true, false}, std::pair{false, true}}) {
        const auto p10 = TranslateFragmentInterpMovSelector(0, flat_shade, offset5);
        EXPECT_EQ(p10.attribute_indices, (std::vector<u32>{1U, 0U}));
        EXPECT_EQ(p10.fsub_count, 1U);
    }

    const auto p10 = TranslateFragmentInterpMovSelector(0, true, true);
    EXPECT_EQ(p10.attribute_indices, (std::vector<u32>{1U}));
    EXPECT_EQ(p10.fsub_count, 0U);
}

// Example
// TEST_F(GcnTest, test_name) {
//     // Runner sets the vulkan context
//     auto runner = gcn_test::Runner::instance().value();
//
//     // v_add_f32 v0, v0, v1
//     auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
//
//     // run<T> tells how to interpret the result (only 32bit as of now)
//     // the second argument is templated, it can be at most 4 u32s
//     // the data is accessible by the instruction in v0-4 and s0-4 (mirrored)
//     // the result has to be placed in v0
//     auto result = runner->run<float>(spirv, F32x2{1.5f, 6.0f});
//
//     EXPECT_TRUE(result.has_value());
//     EXPECT_EQ(*result, 7.5f);
// }

TEST_F(GcnTest, add_f32) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, F32x2{1.5f, 6.0f});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 7.5f);
}

TEST_F(GcnTest, add_i32_carry_feeds_addc_u32) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 2> instructions{
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V1, SOperand9::S0, VOperand8::V1).Get(),
        VOP2(OpcodeVOP2::V_ADDC_U32, VOperand8::V0, SOperand9::S2, VOperand8::V3).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    const auto overflow = runner->run<u32>(spirv, std::array{0xffffffffU, 1U, 7U, 0U});
    ASSERT_TRUE(overflow.has_value());
    EXPECT_EQ(*overflow, 8U);

    const auto no_overflow = runner->run<u32>(spirv, std::array{2U, 1U, 7U, 0U});
    ASSERT_TRUE(no_overflow.has_value());
    EXPECT_EQ(*no_overflow, 7U);
}

TEST_F(GcnTest, add_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, F32x2{1.0f, std::numeric_limits<float>::quiet_NaN()});

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(std::isnan(*result));
}

using half = half_float::half;

struct F16x2 {
    half a;
    half b = half(0.0f);

    bool operator==(const F16x2& rhs) const = default;
};

static_assert(sizeof(F16x2) == sizeof(float));

TEST_F(GcnTest, add_f16) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_ADD_F16, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, F16x2{half(2.0f)});
}

TEST_F(GcnTest, add_f16_clamp) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetClamp(true).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, F16x2{half(1.0f)}); //confirmed with neo
}

TEST_F(GcnTest, add_f16_neg) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetNeg({true, true, false}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ((*result).a, half(-2.0f)); //confirmed with neo
}

TEST_F(GcnTest, add_f16_opsel_hi) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({true, true, false, true}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(1.0f), half(2.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ((*result).a, half(1.0f));
    EXPECT_EQ((*result).b, half(4.0f));
}

TEST_F(GcnTest, sub_f16) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_SUB_F16, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(0.0f)}, F16x2{half(1.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, F16x2{half(-1.0f)}); //confirmed with neo
}

TEST_F(GcnTest, mul_legacy_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MUL_LEGACY_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<u32>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
}

TEST_F(GcnTest, mul_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MUL_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(std::isnan(*result));
}

TEST_F(GcnTest, min_legacy_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MIN_LEGACY_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<u32>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x7fc00000);
}

TEST_F(GcnTest, min_nan) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP2(OpcodeVOP2::V_MIN_F32, VOperand8::V0, SOperand9::V0, VOperand8::V1).Get());
    auto result = runner->run<float>(spirv, std::array{u32(0), u32(0x7fc00000)});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
}

TEST_F(GcnTest, add3_u32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array{0, 1, 2});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 3);
}

TEST_F(GcnTest, add3_u32_2) {
    auto runner = gcn_test::Runner::instance().value();
    auto big = 2000000000;

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array{big, big, big});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x65A0BC00);
}

TEST_F(GcnTest, add3_u32_3) {
    auto runner = gcn_test::Runner::instance().value();
    auto big = 2000000000;

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetClamp(true).Get());
    auto result = runner->run<u32>(spirv, std::array{big, big, big});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x65A0BC00);
}

TEST_F(GcnTest, add3_u32_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ADD3_U32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array{0, 1, 2});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x80000003);
}

TEST_F(GcnTest, or3_u32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xF0F0F0F0, 0x07070707, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xF7F7F7F7);
}

TEST_F(GcnTest, or3_u32_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array{0x07070707, 0x11111111, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x57575757);
}

TEST_F(GcnTest, or3_u32_3) {
    auto runner = gcn_test::Runner::instance().value();
    auto big = 2000000000;

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetClamp(true).Get());
    auto result = runner->run<u32>(spirv, std::array{0x07070707, 0x11111111, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x57575757);
}

TEST_F(GcnTest, or3_u32_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_OR3_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({0,0,1}).Get());
    auto result = runner->run<u32>(spirv, std::array{0x07070707, 0x11111111, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xD7575757);
}

TEST_F(GcnTest, and_or_b32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xF0F0F0F0, 0x07070707, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x11111111);
}

TEST_F(GcnTest, and_or_b32_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOmod(Omod::Mul2).Get());
    auto result = runner->run<u32>(spirv, std::array{0x40404040, 0x40404040, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x40404040);
}

TEST_F(GcnTest, and_or_b32_3) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetClamp(true).Get());
    auto result = runner->run<u32>(spirv, std::array{0x40404040, 0x40404040, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x40404040);
}

TEST_F(GcnTest, and_or_b32_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0x07070707, 0x11111111, 0xF0F0F0F0});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xF1F1F1F1);
}

TEST_F(GcnTest, and_or_b32_5) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetNeg({1,0,0}).SetAbs({1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0x77777777, 0xB0B0B0B0, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xB1313131);
}

TEST_F(GcnTest, and_or_b32_6) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOmod(Omod::Mul2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0x40404040, 0xB0B0B0B0, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x11111111);
}

TEST_F(GcnTest, and_or_b32_7) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOmod(Omod::Div2).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xB0B0B0B0, 0x77777777, 0x40404040});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x70707070);
}

TEST_F(GcnTest, and_or_b32_8) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_AND_OR_B32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetAbs({1,1,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,3>{0xB0B0B0B0, 0x11111111, 0x11111111});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x11111111);
}

TEST_F(GcnTest, mad_mix_f32_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIX_F32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<float>(spirv, std::array{2.0f, 3.0f, 4.0f});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 10.0f);
}

TEST_F(GcnTest, mad_mix_f32_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIX_F32, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({1,1,0}).SetOpSel({1,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<float>(spirv, std::array<u32,3>{
        std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(4.0f)}
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 26.0f);
}

TEST_F(GcnTest, mad_mixlo_f16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIXLO_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({1,1,0}).SetOpSel({1,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<F16x2>(spirv, std::array<u32,3>{
        std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(4.0f)}
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(26.0f), half(0.5f)}));
}

TEST_F(GcnTest, mad_mixhi_f16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_MAD_MIXHI_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1, SOperand9::V2).SetOpSelHi({1,1,0}).SetOpSel({1,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<F16x2>(spirv, std::array<u32,3>{
        std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(F16x2{half(44.0f), half(0.5f)}), std::bit_cast<u32>(4.0f)}
    );

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(44.0f), half(26.0f)}));
}

TEST_F(GcnTest, lshrrev_b16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHRREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF2, 0x88881000});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xFFFF0400);
}

TEST_F(GcnTest, lshrrev_b16_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHRREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({0,0,0,1}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF2, 0x88881000});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x0400FFF2);
}

TEST_F(GcnTest, lshrrev_b16_3) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHRREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({0,1,0,0}).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF2, 0x88881000});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xFFFF2222);
}

TEST_F(GcnTest, lshlrev_b16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_LSHLREV_B16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0xFFFFFFF3, 0x88888888});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0xFFFF4440);
}

TEST_F(GcnTest, ashrrev_i16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3A(OpcodeVOP3::V_ASHRREV_I16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<u32>(spirv, std::array<u32,2>{0x1234FFF3, 0x88888888});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x1234F111);
}

TEST_F(GcnTest, pk_add_f16_1) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(4.0f), half(6.0f)}));
}

TEST_F(GcnTest, pk_add_f16_2) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0, SOperand9::ConstInv2Pi).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x00003118);
}

TEST_F(GcnTest, pk_add_f16_3) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0, SOperand9::ConstInv2Pi).SetOpSel({0,1,1}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
}

TEST_F(GcnTest, pk_add_f16_4) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0p5, SOperand9::Const0p5).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x3C00);
}

TEST_F(GcnTest, pk_add_f16_5) {
    auto runner = gcn_test::Runner::instance().value();

    auto inst = VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::Const0, SOperand9::ConstInv2Pi).SetOpSelHi({0,0,0}).Get();
    auto spirv = TranslateToSpirv(inst);
    auto result = runner->run<u32>(spirv, 0U);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0x31183118);
}

TEST_F(GcnTest, pk_add_f16_neg_lo) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetNeg({1,1,0}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(-4.0f), half(6.0f)}));
}

TEST_F(GcnTest, pk_add_f16_neg_hi) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetNegHi({1,1,0}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(4.0f), half(-6.0f)}));
}

TEST_F(GcnTest, pk_add_f16_op_sel_reversed) {
    auto runner = gcn_test::Runner::instance().value();

    auto spirv = TranslateToSpirv(VOP3P(OpcodeVOP3P::V_PK_ADD_F16, VOperand8::V0, SOperand9::V0, SOperand9::V1).SetOpSel({1,1,1}).SetOpSelHi({0,0,0}).Get());
    auto result = runner->run<F16x2>(spirv, std::array{F16x2{half(1.0f), half(2.0f)}, F16x2{half(3.0f), half(4.0f)}});

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, (F16x2{half(6.0f), half(4.0f)}));
}

TEST_F(GcnTest, addk_i32_overflow_scc) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 3> instructions{
        SOPK(OpcodeSOPK::S_ADDK_I32, SOperand7::S0, 0xffff).Get(),
        SOP2(OpcodeSOP2::S_CSELECT_B32, SOperand7::S0, SOperand8::Const1, SOperand8::Const0).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S0).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);
    auto overflow = runner->run<u32>(spirv, std::array{0x80000000U, 0U, 0U, 0U});
    ASSERT_TRUE(overflow.has_value());
    EXPECT_EQ(*overflow, 1U);

    auto no_overflow = runner->run<u32>(spirv, std::array{5U, 0U, 0U, 0U});
    ASSERT_TRUE(no_overflow.has_value());
    EXPECT_EQ(*no_overflow, 0U);
}

TEST_F(GcnTest, bitcmp1_b64_bit32) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 3> instructions{
        SOPC(OpcodeSOPC::S_BITCMP1_B64, SOperand8::S0, SOperand8::S2).Get(),
        SOP2(OpcodeSOP2::S_CSELECT_B32, SOperand7::S0, SOperand8::Const1, SOperand8::Const0).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S0).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    auto result = runner->run<u32>(spirv, std::array{0U, 1U, 32U, 0U});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1U);
}

TEST_F(GcnTest, subb_u32_clears_vcc) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 3> instructions{
        VOP2(OpcodeVOP2::V_SUB_I32, VOperand8::V1, SOperand9::S0, VOperand8::V1).Get(),
        VOP2(OpcodeVOP2::V_SUBB_U32, VOperand8::V1, SOperand9::S2, VOperand8::V3).Get(),
        VOP2(OpcodeVOP2::V_ADDC_U32, VOperand8::V0, SOperand9::Const0, VOperand8::V3).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    auto result = runner->run<u32>(spirv, std::array{0U, 1U, 5U, 0U});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0U);
}

TEST_F(GcnTest, subb_u32_scc_wrap) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 4> instructions{
        SOP2(OpcodeSOP2::S_ADD_U32, SOperand7::S3, SOperand8::S0, SOperand8::S1).Get(),
        SOP2(OpcodeSOP2::S_SUBB_U32, SOperand7::S3, SOperand8::Const0, SOperand8::S2).Get(),
        SOP2(OpcodeSOP2::S_CSELECT_B32, SOperand7::S0, SOperand8::Const1, SOperand8::Const0).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S0).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    auto result = runner->run<u32>(spirv, std::array{0xffffffffU, 1U, 0xffffffffU, 0U});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1U);
}

TEST_F(GcnTest, addc_u32_clears_scc) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 4> instructions{
        SOP2(OpcodeSOP2::S_ADD_U32, SOperand7::S3, SOperand8::S0, SOperand8::S1).Get(),
        SOP2(OpcodeSOP2::S_ADDC_U32, SOperand7::S3, SOperand8::Const0, SOperand8::Const0).Get(),
        SOP2(OpcodeSOP2::S_CSELECT_B32, SOperand7::S0, SOperand8::Const1, SOperand8::Const0).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S0).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    auto result = runner->run<u32>(spirv, std::array{0xffffffffU, 1U, 0U, 0U});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0U);
}

TEST_F(GcnTest, addc_u32_result_uses_scc) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 3> instructions{
        SOP2(OpcodeSOP2::S_ADD_U32, SOperand7::S1, SOperand8::S0, SOperand8::S1).Get(),
        SOP2(OpcodeSOP2::S_ADDC_U32, SOperand7::S0, SOperand8::S2, SOperand8::S3).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S0).Get(),
    };
    const auto spirv = TranslateToSpirv(instructions);

    auto overflow = runner->run<u32>(spirv, std::array{0xffffffffU, 1U, 7U, 0U});
    ASSERT_TRUE(overflow.has_value());
    EXPECT_EQ(*overflow, 8U);

    auto no_overflow = runner->run<u32>(spirv, std::array{2U, 1U, 7U, 0U});
    ASSERT_TRUE(no_overflow.has_value());
    EXPECT_EQ(*no_overflow, 7U);
}

TEST_F(GcnTest, floor_f64_literal_is_high_dword) {
    auto runner = gcn_test::Runner::instance().value();
    if (!runner->supports_float64()) {
        GTEST_SKIP() << "shaderFloat64 is not supported";
    }
    const u64 floor_literal =
        VOP1(OpcodeVOP1::V_FLOOR_F64, VOperand8::V0, SOperand9::LiteralConstant).Get() |
        (u64{0xc0040000U} << 32);
    const std::array<u64, 1> low{floor_literal};
    const std::array<u64, 2> high{
        floor_literal,
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::V1).Get(),
    };

    auto result_lo = runner->run<u32>(TranslateToSpirv(low), std::array{0U, 0U, 0U, 0U});
    auto result_hi = runner->run<u32>(TranslateToSpirv(high), std::array{0U, 0U, 0U, 0U});
    ASSERT_TRUE(result_lo.has_value());
    ASSERT_TRUE(result_hi.has_value());
    EXPECT_EQ(*result_lo, 0U);
    EXPECT_EQ(*result_hi, 0xc0080000U);
}

TEST_F(GcnTest, tg_size_sgpr_holds_wave_count_and_order) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 1> instructions{
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S5).Get(),
    };
    const ComputeTestConfig config{
        .workgroup_size = {128, 1, 1},
        .tgid_enable = {true, false, false},
        .tg_size_enable = true,
        .ordered_append = true,
        .store_per_invocation = true,
    };
    const auto spirv = TranslateToSpirv(instructions, config);

    constexpr u32 NumGroups = 3;
    const auto result = runner->run<std::array<u32, NumGroups * 128>>(
        spirv, std::array{0U, 0U, 0U, 0U}, {NumGroups, 1, 1});
    ASSERT_TRUE(result.has_value());
    for (u32 group = 0; group < NumGroups; ++group) {
        for (u32 thread = 0; thread < 128; ++thread) {
            const u32 wave = thread / 64;
            const u32 first_wave = wave == 0 ? 1U << 31 : 0U;
            const u32 expected = first_wave | ((group * 2 + wave) << 6) | 2U;
            EXPECT_EQ((*result)[group * 128 + thread], expected)
                << "group " << group << " thread " << thread;
        }
    }
}

TEST_F(GcnTest, tg_size_sgpr_holds_wave_in_group_without_ordered_append) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 1> instructions{
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S5).Get(),
    };
    const ComputeTestConfig config{
        .workgroup_size = {128, 1, 1},
        .tgid_enable = {true, false, false},
        .tg_size_enable = true,
        .store_per_invocation = true,
    };
    const auto spirv = TranslateToSpirv(instructions, config);

    constexpr u32 NumGroups = 3;
    const auto result = runner->run<std::array<u32, NumGroups * 128>>(
        spirv, std::array{0U, 0U, 0U, 0U}, {NumGroups, 1, 1});
    ASSERT_TRUE(result.has_value());
    for (u32 group = 0; group < NumGroups; ++group) {
        for (u32 thread = 0; thread < 128; ++thread) {
            const u32 wave = thread / 64;
            const u32 first_wave = wave == 0 ? 1U << 31 : 0U;
            const u32 expected = first_wave | (wave << 6) | 2U;
            EXPECT_EQ((*result)[group * 128 + thread], expected)
                << "group " << group << " thread " << thread;
        }
    }
}

constexpr u32 GdsTicketDword = Shader::GdsOrderedTicketOffset / 4;

std::vector<u32> MakeGds() {
    return std::vector<u32>(GdsTicketDword + Shader::GdsOrderedCounters);
}

u64 OrderedCount(VOperand8 vdst, VOperand8 value, u32 counter, bool release, bool done,
                 bool swap = false) {
    const u32 offset1 = (release ? 0x1U : 0U) | (done ? 0x2U : 0U) | (swap ? 0x10U : 0U);
    return DS(OpcodeDS::DS_ORDERED_COUNT, vdst, value)
        .SetOffsets(static_cast<u8>(counter << 2), static_cast<u8>(offset1))
        .SetGds(true)
        .Get();
}

TEST_F(GcnTest, ds_ordered_count_add_returns_previous_value) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 2> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::M0, SOperand8::Const0).Get(),
        OrderedCount(VOperand8::V0, VOperand8::V1, 1, true, true),
    };
    auto gds = MakeGds();
    gds[1] = 7;

    const auto result = runner->run<u32>(TranslateToSpirv(instructions),
                                         std::array{0xdeadbeefU, 5U, 0U, 0U}, {},
                                         std::as_writable_bytes(std::span{gds}));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 7U);
    EXPECT_EQ(gds[1], 12U);
    EXPECT_EQ(gds[GdsTicketDword], 0U);
    EXPECT_EQ(gds[GdsTicketDword + 1], 1U);
}

TEST_F(GcnTest, ds_ordered_count_swap_returns_previous_value) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 2> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::M0, SOperand8::Const0).Get(),
        OrderedCount(VOperand8::V0, VOperand8::V1, 2, true, true, true),
    };
    auto gds = MakeGds();
    gds[2] = 7;

    const auto result = runner->run<u32>(TranslateToSpirv(instructions),
                                         std::array{0xdeadbeefU, 5U, 0U, 0U}, {},
                                         std::as_writable_bytes(std::span{gds}));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 7U);
    EXPECT_EQ(gds[2], 5U);
}

TEST_F(GcnTest, ds_ordered_count_runs_waves_in_order) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 2> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::M0, SOperand8::Const0).Get(),
        OrderedCount(VOperand8::V0, VOperand8::V1, 0, true, true),
    };
    const ComputeTestConfig config{.store_per_invocation = true};
    auto gds = MakeGds();

    constexpr u32 NumGroups = 64;
    const auto result = runner->run<std::array<u32, NumGroups>>(
        TranslateToSpirv(instructions, config), std::array{0U, 3U, 0U, 0U}, {NumGroups, 1, 1},
        std::as_writable_bytes(std::span{gds}));
    ASSERT_TRUE(result.has_value());
    for (u32 group = 0; group < NumGroups; ++group) {
        EXPECT_EQ((*result)[group], group * 3) << "group " << group;
    }
    EXPECT_EQ(gds[0], NumGroups * 3);
    EXPECT_EQ(gds[GdsTicketDword], NumGroups);
}

TEST_F(GcnTest, ds_ordered_count_runs_once_per_wave) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 2> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::M0, SOperand8::Const0).Get(),
        OrderedCount(VOperand8::V0, VOperand8::V1, 0, true, true),
    };
    const ComputeTestConfig config{
        .workgroup_size = {128, 1, 1},
        .store_per_invocation = true,
    };
    auto gds = MakeGds();

    constexpr u32 NumGroups = 4;
    const auto result = runner->run<std::array<u32, NumGroups * 128>>(
        TranslateToSpirv(instructions, config), std::array{0U, 1U, 0U, 0U}, {NumGroups, 1, 1},
        std::as_writable_bytes(std::span{gds}));
    ASSERT_TRUE(result.has_value());
    for (u32 group = 0; group < NumGroups; ++group) {
        for (u32 thread = 0; thread < 128; ++thread) {
            EXPECT_EQ((*result)[group * 128 + thread], group * 2 + thread / 64)
                << "group " << group << " thread " << thread;
        }
    }
    EXPECT_EQ(gds[0], NumGroups * 2);
    EXPECT_EQ(gds[GdsTicketDword], NumGroups * 2);
}

TEST_F(GcnTest, ds_ordered_count_passes_turn_at_end_of_program) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 3> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::M0, SOperand8::Const0).Get(),
        OrderedCount(VOperand8::V0, VOperand8::V1, 0, false, false),
        SOPP(OpcodeSOPP::S_ENDPGM).Get(),
    };
    const ComputeTestConfig config{
        .workgroup_size = {96, 1, 1},
        .store_per_invocation = true,
    };
    auto gds = MakeGds();

    constexpr u32 NumGroups = 4;
    const auto result = runner->run<std::array<u32, NumGroups * 96>>(
        TranslateToSpirv(instructions, config), std::array{0U, 1U, 0U, 0U}, {NumGroups, 1, 1},
        std::as_writable_bytes(std::span{gds}));
    ASSERT_TRUE(result.has_value());
    for (u32 group = 0; group < NumGroups; ++group) {
        for (u32 thread = 0; thread < 96; ++thread) {
            EXPECT_EQ((*result)[group * 96 + thread], group * 2 + thread / 64)
                << "group " << group << " thread " << thread;
        }
    }
    EXPECT_EQ(gds[0], NumGroups * 2);
    EXPECT_EQ(gds[GdsTicketDword], NumGroups * 2);
}

TEST_F(GcnTest, ds_ordered_count_orders_each_counter) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 3> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::M0, SOperand8::Const0).Get(),
        OrderedCount(VOperand8::V2, VOperand8::V1, 0, true, false),
        OrderedCount(VOperand8::V0, VOperand8::V2, 1, true, true),
    };
    const ComputeTestConfig config{
        .workgroup_size = {128, 1, 1},
        .store_per_invocation = true,
    };
    auto gds = MakeGds();

    constexpr u32 NumGroups = 16;
    constexpr u32 NumWaves = NumGroups * 2;
    const auto result = runner->run<std::array<u32, NumGroups * 128>>(
        TranslateToSpirv(instructions, config), std::array{0U, 1U, 0U, 0U}, {NumGroups, 1, 1},
        std::as_writable_bytes(std::span{gds}));
    ASSERT_TRUE(result.has_value());
    for (u32 group = 0; group < NumGroups; ++group) {
        for (u32 thread = 0; thread < 128; ++thread) {
            const u32 wave = group * 2 + thread / 64;
            EXPECT_EQ((*result)[group * 128 + thread], wave * (wave - 1) / 2)
                << "group " << group << " thread " << thread;
        }
    }
    EXPECT_EQ(gds[0], NumWaves);
    EXPECT_EQ(gds[1], NumWaves * (NumWaves - 1) / 2);
    EXPECT_EQ(gds[GdsTicketDword], NumWaves);
    EXPECT_EQ(gds[GdsTicketDword + 1], NumWaves);
}

// The second wave of a group must pass op 1 before the barrier, while the first wave has only
// done op 1: the release on op 1 passes that counter's turn on.
TEST_F(GcnTest, ds_ordered_count_passes_turn_at_release) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 4> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::M0, SOperand8::Const0).Get(),
        OrderedCount(VOperand8::V2, VOperand8::V1, 0, true, false),
        SOPP(OpcodeSOPP::S_BARRIER).Get(),
        OrderedCount(VOperand8::V0, VOperand8::V2, 1, true, true),
    };
    const ComputeTestConfig config{
        .workgroup_size = {128, 1, 1},
        .store_per_invocation = true,
    };
    auto gds = MakeGds();

    constexpr u32 NumGroups = 16;
    constexpr u32 NumWaves = NumGroups * 2;
    const auto result = runner->run<std::array<u32, NumGroups * 128>>(
        TranslateToSpirv(instructions, config), std::array{0U, 1U, 0U, 0U}, {NumGroups, 1, 1},
        std::as_writable_bytes(std::span{gds}));
    ASSERT_TRUE(result.has_value());
    for (u32 group = 0; group < NumGroups; ++group) {
        for (u32 thread = 0; thread < 128; ++thread) {
            const u32 wave = group * 2 + thread / 64;
            EXPECT_EQ((*result)[group * 128 + thread], wave * (wave - 1) / 2)
                << "group " << group << " thread " << thread;
        }
    }
    EXPECT_EQ(gds[0], NumWaves);
    EXPECT_EQ(gds[1], NumWaves * (NumWaves - 1) / 2);
    EXPECT_EQ(gds[GdsTicketDword], NumWaves);
    EXPECT_EQ(gds[GdsTicketDword + 1], NumWaves);
}

// Each lane passes its index + 1. The value of the wave's first active lane is added, also when
// the host runs the wave as more than one subgroup. The 1 comes from the user data: the runner
// binds the GDS buffer at binding 2, after the user data, so the shader has to read the user data.
TEST_F(GcnTest, ds_ordered_count_adds_first_active_lane_value) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 3> instructions{
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::M0, SOperand8::Const0).Get(),
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V1, SOperand9::S1, VOperand8::V4).Get(),
        OrderedCount(VOperand8::V0, VOperand8::V1, 0, true, true),
    };
    const ComputeTestConfig config{
        .workgroup_size = {64, 1, 1},
        .store_per_invocation = true,
        .local_index_in_v4 = true,
    };
    auto gds = MakeGds();

    constexpr u32 NumGroups = 16;
    const auto result = runner->run<std::array<u32, NumGroups * 64>>(
        TranslateToSpirv(instructions, config), std::array{0U, 1U, 0U, 0U}, {NumGroups, 1, 1},
        std::as_writable_bytes(std::span{gds}));
    ASSERT_TRUE(result.has_value());
    for (u32 group = 0; group < NumGroups; ++group) {
        for (u32 thread = 0; thread < 64; ++thread) {
            EXPECT_EQ((*result)[group * 64 + thread], group)
                << "group " << group << " thread " << thread;
        }
    }
    EXPECT_EQ(gds[0], NumGroups);
    EXPECT_EQ(gds[GdsTicketDword], NumGroups);
}

// v_cmpx leaves exec set for lanes 5-20 (lane - 5 < 16, unsigned). v_readfirstlane then reads the
// first of them, lane 5, whose v1 is 6, and every lane gets that value.
TEST_F(GcnTest, readfirstlane_reads_first_lane_with_exec_bit) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 5> instructions{
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V1, SOperand9::Const1, VOperand8::V4).Get(),
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V2, SOperand9::ConstNeg5, VOperand8::V4).Get(),
        VOPC(OpcodeVOPC::V_CMPX_GT_U32, SOperand9::Const16, VOperand8::V2).Get(),
        VOP1(OpcodeVOP1::V_READFIRSTLANE_B32, VOperand8::V12, SOperand9::V1).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S12).Get(),
    };
    const ComputeTestConfig config{
        .workgroup_size = {32, 1, 1},
        .store_per_invocation = true,
        .local_index_in_v4 = true,
    };
    const auto result = runner->run<std::array<u32, 32>>(TranslateToSpirv(instructions, config),
                                                         std::array{0U, 0U, 0U, 0U});
    ASSERT_TRUE(result.has_value());
    for (u32 lane = 0; lane < 32; ++lane) {
        EXPECT_EQ((*result)[lane], 6U) << "lane " << lane;
    }
}

// With no exec bit set v_readfirstlane reads lane 0.
TEST_F(GcnTest, readfirstlane_reads_lane_0_without_exec_bits) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 4> instructions{
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V1, SOperand9::Const1, VOperand8::V4).Get(),
        VOPC(OpcodeVOPC::V_CMPX_GT_U32, SOperand9::Const0, VOperand8::V4).Get(),
        VOP1(OpcodeVOP1::V_READFIRSTLANE_B32, VOperand8::V12, SOperand9::V1).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::S12).Get(),
    };
    const ComputeTestConfig config{
        .workgroup_size = {32, 1, 1},
        .store_per_invocation = true,
        .local_index_in_v4 = true,
    };
    const auto result = runner->run<std::array<u32, 32>>(TranslateToSpirv(instructions, config),
                                                         std::array{0U, 0U, 0U, 0U});
    ASSERT_TRUE(result.has_value());
    for (u32 lane = 0; lane < 32; ++lane) {
        EXPECT_EQ((*result)[lane], 1U) << "lane " << lane;
    }
}

// v_writelane sets one lane of v1 and every other lane keeps its value. v1 = lane + 1, then lane 2
// takes -15 = 0xfffffff1 (VOP3 form, constant lane select) and lane 31 takes s1 (VOP2 form: the
// lane select is an SGPR, s2, in the vsrc1 field).
TEST_F(GcnTest, writelane_sets_selected_lane_and_keeps_other_lanes) {
    auto runner = gcn_test::Runner::instance().value();
    const std::array<u64, 5> instructions{
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V1, SOperand9::Const1, VOperand8::V4).Get(),
        VOP3A(OpcodeVOP3::V_WRITELANE_B32, VOperand8::V1, SOperand9::ConstNeg15, SOperand9::Const2)
            .Get(),
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S2, SOperand8::Const31).Get(),
        VOP2(OpcodeVOP2::V_WRITELANE_B32, VOperand8::V1, SOperand9::S1, VOperand8::V2).Get(),
        VOP1(OpcodeVOP1::V_MOV_B32, VOperand8::V0, SOperand9::V1).Get(),
    };
    const ComputeTestConfig config{
        .workgroup_size = {32, 1, 1},
        .store_per_invocation = true,
        .local_index_in_v4 = true,
    };
    const auto result = runner->run<std::array<u32, 32>>(TranslateToSpirv(instructions, config),
                                                         std::array{0U, 0xc0de0031U, 0U, 0U});
    ASSERT_TRUE(result.has_value());
    for (u32 lane = 0; lane < 32; ++lane) {
        const u32 expected = lane == 2 ? 0xfffffff1U : lane == 31 ? 0xc0de0031U : lane + 1;
        EXPECT_EQ((*result)[lane], expected) << "lane " << lane;
    }
}
