// src/quant/x86/dot_avxvnni.cpp - the Avx2 contract on CPUs with AVX-VNNI (VEX: Alder Lake and later Intel, Zen 5): the same ported ggml code as
// dot_avx2.cpp, built so that its integer dot products of four bytes take VPDPBUSD, one instruction for maddubs + madd (ggml's __AVXVNNI__
// branch of mul_sum_us8_pairs_float, and mul_sum_i8_quads for the 4-bit grid formats).  The integer sums are exact either way and the float
// order is the AVX2 one, so the bits are the AVX2 tier's.  Only called when caps() says AVX-VNNI (Isa::AvxVnni); only the formats whose dot
// changes are registered.  The idea is Strata 0.2.18's (Niko1221/Strata, MIT, commit bde1682d1 "hy4: add AVX-VNNI + S8_K fast CPU kernels").
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__AVXVNNI__)
#define __AVXVNNI__ 1   // MSVC has the intrinsics under /arch:AVX2 but no macro for them
#endif
#include "tier_common.hpp"

#include "../generic.hpp"
#include "../registry.hpp"
#include "strata/quant/dot.hpp"

#if !defined(__AVXVNNI__) || !defined(__AVX2__)
#error "dot_avxvnni.cpp needs AVX2 and AVX-VNNI"
#endif

namespace strata::quant {
namespace avxvnni {

#include "x86_helpers.inc"
#include "x86_dots.inc"

// Experimental K-quant paths, derived from the optimized AVX2 kernels.
// VPMADDUBSW is safe: unsigned weight max 31/63 times signed int8,
// even a pair of extremes fits in int16. VPDPWSSD fuses VPMADDWD and
// VPADDD, non-saturating, keeping exactly the eight int32 lanes.
// Q6 integer addition is associative modulo 2^32; all valid block sums
// also fit int32. Float block order and Q5 minimum correction unchanged.
void dot_q5_K_q8_K_vnni(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q5_K* STRATA_RESTRICT x = (const block_q5_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m16 = _mm256_set1_epi8(16);
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const uint8_t* STRATA_RESTRICT q5 = x[i].qs;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        const float d = y[i].d * h2f(x[i].d);
        const float dmin = -y[i].d * h2f(x[i].dmin);

        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_castsi256_si128(mins_and_scales);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

#define Q5K_VNNI_STEP(J, QH0, QH1) do { \
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * (J) + 0)); \
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * (J) + 1)); \
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) (q5 + 32 * (J))); \
            const __m256i q5_0 = _mm256_or_si256(_mm256_and_si256(q5bits, m4), (QH0)); \
            const __m256i q5_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4), (QH1)); \
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * (J))); \
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 64 * (J) + 32)); \
            sumi0 = _mm256_dpwssd_avx_epi32(sumi0, scale_0, _mm256_maddubs_epi16(q5_0, q8_0)); \
            sumi1 = _mm256_dpwssd_avx_epi32(sumi1, scale_1, _mm256_maddubs_epi16(q5_1, q8_1)); \
        } while (0)

        Q5K_VNNI_STEP(0, _mm256_and_si256(_mm256_slli_epi16(hbits, 4), m16), _mm256_and_si256(_mm256_slli_epi16(hbits, 3), m16));
        Q5K_VNNI_STEP(1, _mm256_and_si256(_mm256_slli_epi16(hbits, 2), m16), _mm256_and_si256(_mm256_slli_epi16(hbits, 1), m16));
        Q5K_VNNI_STEP(2, _mm256_and_si256(hbits, m16),                        _mm256_and_si256(_mm256_srli_epi16(hbits, 1), m16));
        Q5K_VNNI_STEP(3, _mm256_and_si256(_mm256_srli_epi16(hbits, 2), m16), _mm256_and_si256(_mm256_srli_epi16(hbits, 3), m16));
#undef Q5K_VNNI_STEP

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi0, sumi1)), acc);
    }

    *s = hsum_float_8(acc) + summs;
}

void dot_q6_K_q8_K_vnni(int n, float* STRATA_RESTRICT s, const void* STRATA_RESTRICT vx, const void* STRATA_RESTRICT vy) {
    assert(n % QK_K == 0);
    const block_q6_K* STRATA_RESTRICT x = (const block_q6_K*) vx;
    const block_q8_K* STRATA_RESTRICT y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m192 = _mm256_set1_epi8(-64);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (i + 2 < nb) {
            _mm_prefetch((const char*) &x[i + 2], _MM_HINT_T0);
            _mm_prefetch((const char*) &y[i + 2], _MM_HINT_T0);
        }
        const float d = y[i].d * h2f(x[i].d);
        const uint8_t* STRATA_RESTRICT q4 = x[i].ql;
        const uint8_t* STRATA_RESTRICT qh = x[i].qh;
        const int8_t*  STRATA_RESTRICT q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*) x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        int is = 0;

        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) q4);
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) (q4 + 32));
            q4 += 64;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m12), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, m48);
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, m192), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) (q8 +  0));
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 64));
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) (q8 + 96));
            q8 += 128;

            const __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            const __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            const __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            const __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            sumi0 = _mm256_dpwssd_avx_epi32(sumi0, _mm256_cvtepi8_epi16(scale_0), p16_0);
            sumi0 = _mm256_dpwssd_avx_epi32(sumi0, _mm256_cvtepi8_epi16(scale_1), p16_1);
            sumi1 = _mm256_dpwssd_avx_epi32(sumi1, _mm256_cvtepi8_epi16(scale_2), p16_2);
            sumi1 = _mm256_dpwssd_avx_epi32(sumi1, _mm256_cvtepi8_epi16(scale_3), p16_3);
        }

        const __m256i sumi = _mm256_sub_epi32(_mm256_add_epi32(sumi0, sumi1), q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);
}

}  // namespace avxvnni

namespace {
#define ENTRY(T, ACT, NAME) {avxvnni::dot_##NAME, Type::T, Type::ACT, Arith::Avx2, Isa::AvxVnni, "avxvnni_" #NAME}
const DotKernel kTable[] = {
    ENTRY(Q4_0, Q8_0, q4_0_q8_0), ENTRY(Q4_1, Q8_1, q4_1_q8_1),     ENTRY(Q5_0, Q8_0, q5_0_q8_0),   ENTRY(Q5_1, Q8_1, q5_1_q8_1),
    ENTRY(Q8_0, Q8_0, q8_0_q8_0), ENTRY(IQ4_NL, Q8_0, iq4_nl_q8_0), ENTRY(MXFP4, Q8_0, mxfp4_q8_0), ENTRY(NVFP4, Q8_0, nvfp4_q8_0),
};
#undef ENTRY
}  // namespace

// Separate table: deliberately excluded from default dot dispatch.
const DotKernel* avxvnni_experimental_k_dot(Type w) noexcept {
    static const DotKernel q5{avxvnni::dot_q5_K_q8_K_vnni, Type::Q5_K, Type::Q8_K,
                              Arith::Avx2, Isa::AvxVnni, "avxvnni_q5_K_q8_K_experimental"};
    static const DotKernel q6{avxvnni::dot_q6_K_q8_K_vnni, Type::Q6_K, Type::Q8_K,
                              Arith::Avx2, Isa::AvxVnni, "avxvnni_q6_K_q8_K_experimental"};
    return w == Type::Q5_K ? &q5 : w == Type::Q6_K ? &q6 : nullptr;
}

DotTable avxvnni_dot_table() noexcept { return {kTable, (int) (sizeof kTable / sizeof kTable[0])}; }

}  // namespace strata::quant
