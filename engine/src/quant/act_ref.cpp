// src/quant/act_ref.cpp - the reference activation quantizers (Generic contract) and the Q4_0, Q5_0 and IQ4_NL
// requantizers.
//
// Derived from ggml's `quantize_row_q8_0_ref`, `quantize_row_q8_1_ref`, `quantize_row_q8_K_ref`, `quantize_row_q4_0_ref`,
// `quantize_row_q5_0_ref` and `quantize_row_iq4_nl_ref` (llama.cpp
// ggml/src/ggml-quants.c at 3cf03257f219afbe7334045ff7c6a06ac68c627d, Copyright (c) 2023-2026 The ggml authors, MIT).  Q8_0 / Q8_1 round half
// away from zero (roundf) here; ggml's AVX / AVX2 quantizers round half to even (x86/act_avx*.cpp).  Q8_K is ggml's scalar code on every x86
// flavour, and its scale is -127/max, not -128/max (the comment in the loop says why): do not "fix" that.
#define STRATA_QUANT_TABLES
#include "strata/quant/act.hpp"

#include <cassert>

#include "internal.hpp"
#include "registry.hpp"

#define GGML_UNUSED(x) (void) (x)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace strata::quant {
namespace {

void q8_0_ref(const float* STRATA_RESTRICT x, block_q8_0* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

void q8_1_ref(const float* STRATA_RESTRICT x, block_q8_1* STRATA_RESTRICT y, int64_t k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_1; j++) {
            const float v = x[i*QK8_1 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        int sum = 0;

        for (int j = 0; j < QK8_1/2; ++j) {
            const float v0 = x[i*QK8_1           + j]*id;
            const float v1 = x[i*QK8_1 + QK8_1/2 + j]*id;

            y[i].qs[          j] = roundf(v0);
            y[i].qs[QK8_1/2 + j] = roundf(v1);

            sum += y[i].qs[          j];
            sum += y[i].qs[QK8_1/2 + j];
        }

        y[i].s = fp32_to_fp16(sum*d);
    }
}

void q8_K_ref(const float* STRATA_RESTRICT x, block_q8_K* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

void q4_0_ref(const float* STRATA_RESTRICT x, block_q4_0* STRATA_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void q5_0_ref(const float* STRATA_RESTRICT x, block_q5_0* STRATA_RESTRICT y, int64_t k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = fp32_to_fp16(d);

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j]  = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}

static constexpr float kGroupMaxEps = 1e-15f;

static inline void get_scale_min_k4_ref(int j, const uint8_t* STRATA_RESTRICT q, uint8_t* STRATA_RESTRICT d,
                                        uint8_t* STRATA_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static float make_qx_quants_ref(int n, int nmax, const float* STRATA_RESTRICT x, int8_t* STRATA_RESTRICT L,
                                int rmse_type) {
    float max = 0, amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < kGroupMaxEps) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = MAX(-nmax, MIN(nmax - 1, nearest_int(iscale * x[i])));
        L[i] = (int8_t) (l + nmax);
        float w = rmse_type == 1 ? x[i] * x[i] : 1.0f;
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.2f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = MAX(-nmax, MIN(nmax - 1, nearest_int(iscale * x[i])));
            float w = rmse_type == 1 ? x[i] * x[i] : 1.0f;
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i)
                L[i] = (int8_t) (nmax + MAX(-nmax, MIN(nmax - 1, nearest_int(iscale * x[i]))));
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

static float make_qkx2_quants_ref(int n, int nmax, const float* STRATA_RESTRICT x, const float* STRATA_RESTRICT weights,
                                  uint8_t* STRATA_RESTRICT L, float* STRATA_RESTRICT the_min,
                                  uint8_t* STRATA_RESTRICT Laux, float rmin, float rdelta, int nstep) {
    float min = x[0], max = x[0];
    float sum_w = weights[0], sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min);
    float scale = 1.0f / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = MAX(0, MIN(nmax, nearest_int(iscale * (x[i] - min))));
        L[i] = (uint8_t) l;
        float diff = scale * l + min - x[i];
        best_error += weights[i] * diff * diff;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = MAX(0, MIN(nmax, nearest_int(iscale * (x[i] - min))));
            Laux[i] = (uint8_t) l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                cur_error += weights[i] * diff * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

void q5_K_ref(const float* STRATA_RESTRICT x, block_q5_K* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    uint8_t L[QK_K], Laux[32];
    float mins[QK_K / 32], scales[QK_K / 32], weights[32];
    for (int64_t i = 0; i < nb; ++i) {
        float max_scale = 0, max_min = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
            float av_x = sqrtf(sum_x2 / 32.0f);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
            scales[j] = make_qkx2_quants_ref(32, 31, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.5f, 0.3f, 5);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }
        float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = (uint8_t) MIN(63, MAX(0, nearest_int(inv_scale * scales[j])));
            uint8_t lm = (uint8_t) MIN(63, MAX(0, nearest_int(inv_min * mins[j])));
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j + 4] = lm;
            } else {
                y[i].scales[j + 4] = (uint8_t) ((ls & 0xF) | ((lm & 0xF) << 4));
                y[i].scales[j - 4] |= (uint8_t) ((ls >> 4) << 6);
                y[i].scales[j - 0] |= (uint8_t) ((lm >> 4) << 6);
            }
        }
        y[i].d = fp32_to_fp16(max_scale / 63.f);
        y[i].dmin = fp32_to_fp16(max_min / 63.f);
        uint8_t sc, m;
        for (int j = 0; j < QK_K / 32; ++j) {
            get_scale_min_k4_ref(j, y[i].scales, &sc, &m);
            const float d = fp16_to_fp32(y[i].d) * sc;
            if (!d) {
                for (int ii = 0; ii < 32; ++ii) L[32 * j + ii] = 0;
                continue;
            }
            const float dm = fp16_to_fp32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = MAX(0, MIN(31, nearest_int((x[32 * j + ii] + dm) / d)));
                L[32 * j + ii] = (uint8_t) l;
            }
        }
        uint8_t* qh = y[i].qh;
        uint8_t* ql = y[i].qs;
        memset(qh, 0, QK_K / 8);
        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) { l1 -= 16; qh[j] |= m1; }
                int l2 = L[n + j + 32];
                if (l2 > 15) { l2 -= 16; qh[j] |= m2; }
                ql[j] = (uint8_t) (l1 | (l2 << 4));
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }
        x += QK_K;
    }
}

void q6_K_ref(const float* STRATA_RESTRICT x, block_q6_K* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    int8_t L[QK_K];
    float scales[QK_K / 16];
    for (int64_t i = 0; i < nb; ++i) {
        float max_scale = 0, max_abs_scale = 0;
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            const float scale = make_qx_quants_ref(16, 32, x + 16 * ib, L + 16 * ib, 1);
            scales[ib] = scale;
            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }
        if (max_abs_scale < kGroupMaxEps) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = fp32_to_fp16(0.f);
            x += QK_K;
            continue;
        }
        const float iscale = -128.f / max_scale;
        y[i].d = fp32_to_fp16(1.f / iscale);
        for (int ib = 0; ib < QK_K / 16; ++ib)
            y[i].scales[ib] = (int8_t) MAX(-128, MIN(127, nearest_int(iscale * scales[ib])));
        for (int j = 0; j < QK_K / 16; ++j) {
            const float d = fp16_to_fp32(y[i].d) * y[i].scales[j];
            if (!d) {
                for (int ii = 0; ii < 16; ++ii) L[16 * j + ii] = 32;
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = MAX(-32, MIN(31, nearest_int(x[16 * j + ii] / d)));
                L[16 * j + ii] = (int8_t) (l + 32);
            }
        }
        uint8_t* ql = y[i].ql;
        uint8_t* qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = (uint8_t) (L[j + l + 0] & 0xF);
                const uint8_t q2 = (uint8_t) (L[j + l + 32] & 0xF);
                const uint8_t q3 = (uint8_t) (L[j + l + 64] & 0xF);
                const uint8_t q4 = (uint8_t) (L[j + l + 96] & 0xF);
                ql[l + 0] = (uint8_t) (q1 | (q3 << 4));
                ql[l + 32] = (uint8_t) (q2 | (q4 << 4));
                qh[l] = (uint8_t) ((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                                   ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }
        x += QK_K;
    }
}

int best_index_int8(int n, const int8_t* val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

// quantize_row_iq4_nl_impl() for one block of 32 with no importance weights and ntry 7
void iq4_nl_ref(const float* STRATA_RESTRICT x, block_iq4_nl* STRATA_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    const int8_t* values = kvalues_iq4nl;
    const int ntry = 7;
    for (int64_t ib = 0; ib < k / QK4_NL; ++ib) {
        const float* xb = x + ib*QK4_NL;
        float weight[QK4_NL];
        uint8_t L[QK4_NL];
        for (int j = 0; j < QK4_NL; ++j) weight[j] = xb[j]*xb[j];
        float amax = 0, max = 0;
        for (int j = 0; j < QK4_NL; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        float d = 0;
        if (amax >= 1e-15f) {
            d = -max/values[0];
            float id = 1/d;
            float sumqx = 0, sumq2 = 0;
            for (int j = 0; j < QK4_NL; ++j) {
                float al = id*xb[j];
                int l = best_index_int8(16, values, al);
                L[j] = l;
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            d = sumqx/sumq2;
            float best = d*sumqx;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx = sumq2 = 0;
                for (int j = 0; j < QK4_NL; ++j) {
                    float al = id*xb[j];
                    int l = best_index_int8(16, values, al);
                    float q = values[l];
                    float w = weight[j];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    d = sumqx/sumq2; best = d * sumqx;
                }
            }
        }
        y[ib].d = fp32_to_fp16(d);
        const float id = d ? 1/d : 0;
        for (int j = 0; j < QK4_NL; ++j) L[j] = best_index_int8(16, values, id*xb[j]);
        for (int j = 0; j < 16; ++j) y[ib].qs[j] = L[j] | (L[16 + j] << 4);
    }
}

}  // namespace

void quantize_q8_0_ref(const float* x, void* y, int64_t n) { q8_0_ref(x, (block_q8_0*) y, n); }
void quantize_q8_1_ref(const float* x, void* y, int64_t n) { q8_1_ref(x, (block_q8_1*) y, n); }
void quantize_q8_K_ref(const float* x, void* y, int64_t n) { q8_K_ref(x, (block_q8_K*) y, n); }
void quantize_q4_0_ref(const float* x, void* y, int64_t n) { q4_0_ref(x, (block_q4_0*) y, n); }
void quantize_q5_0_ref(const float* x, void* y, int64_t n) { q5_0_ref(x, (block_q5_0*) y, n); }
void quantize_iq4_nl_ref(const float* x, void* y, int64_t n) { iq4_nl_ref(x, (block_iq4_nl*) y, n); }
void quantize_q5_K_ref(const float* x, void* y, int64_t n) { q5_K_ref(x, (block_q5_K*) y, n); }
void quantize_q6_K_ref(const float* x, void* y, int64_t n) { q6_K_ref(x, (block_q6_K*) y, n); }

namespace {
const ActKernel kGenericAct[] = {
    {quantize_q8_0_ref, Type::Q8_0, Arith::Generic, Isa::Scalar, "generic_q8_0"},
    {quantize_q8_1_ref, Type::Q8_1, Arith::Generic, Isa::Scalar, "generic_q8_1"},
    {quantize_q8_K_ref, Type::Q8_K, Arith::Generic, Isa::Scalar, "generic_q8_K"},
};
}  // namespace

ActTable generic_act_table() noexcept { return {kGenericAct, 3}; }

}  // namespace strata::quant
