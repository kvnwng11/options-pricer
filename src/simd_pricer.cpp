#include "simd_pricer.h"
#include "black_scholes.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numbers>
#include <vector>

#if PRICING_HAS_AVX2
    #include <immintrin.h>
#endif
#if PRICING_HAS_NEON
    #include <arm_neon.h>
#endif
#if PRICING_HAS_SSE4
    #include <smmintrin.h>
#endif

namespace pricing {

// ─────────────────────────────────────────────
//  BatchInput helper
// ─────────────────────────────────────────────
void BatchInput::fill(
    const std::vector<BSParams>& params,
    std::vector<float>& S_buf,
    std::vector<float>& K_buf,
    std::vector<float>& r_buf,
    std::vector<float>& q_buf,
    std::vector<float>& sigma_buf,
    std::vector<float>& T_buf,
    std::vector<int>&   type_buf)
{
    const std::size_t n = params.size();
    S_buf.resize(n);  K_buf.resize(n);  r_buf.resize(n);
    q_buf.resize(n);  sigma_buf.resize(n); T_buf.resize(n);
    type_buf.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
        S_buf[i]     = static_cast<float>(params[i].S);
        K_buf[i]     = static_cast<float>(params[i].K);
        r_buf[i]     = static_cast<float>(params[i].r);
        q_buf[i]     = static_cast<float>(params[i].q);
        sigma_buf[i] = static_cast<float>(params[i].sigma);
        T_buf[i]     = static_cast<float>(params[i].T);
        type_buf[i]  = (params[i].type == OptionType::Put) ? 1 : 0;
    }
}

// ─────────────────────────────────────────────
//  Runtime SIMD level detection
// ─────────────────────────────────────────────
const char* SIMDBatchPricer::simdLevel() {
#if PRICING_HAS_AVX2
    return "AVX2 (8 floats/cycle)";
#elif PRICING_HAS_NEON
    return "NEON (4 floats/cycle)";
#elif PRICING_HAS_SSE4
    return "SSE4.1 (scalar double)";
#else
    return "Scalar (fallback)";
#endif
}

// ─────────────────────────────────────────────
//  Scalar fallback (used for tail elements and non-SIMD builds)
// ─────────────────────────────────────────────
void SIMDBatchPricer::priceScalar(const BatchInput& in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        BSParams p{
            in.S[i], in.K[i], in.r[i], in.q[i], in.sigma[i], in.T[i],
            (in.type[i] == 0) ? OptionType::Call : OptionType::Put
        };
        out[i] = static_cast<float>(BlackScholesEngine::price(p).price);
    }
}

// ─────────────────────────────────────────────
//  AVX2 implementation — 8 options per iteration
// ─────────────────────────────────────────────
#if PRICING_HAS_AVX2

// ── Polynomial approximations ──────────────────────────────────────────────

// Abramowitz & Stegun §26.2.17 rational approximation for norm_cdf
// Max error: ~1.5e-7 — acceptable for float precision
// This avoids erfc() which has no SIMD intrinsic.
static inline __m256 avx2_norm_cdf(__m256 x) {
    const __m256 ones   = _mm256_set1_ps(1.0f);
    const __m256 zeros  = _mm256_setzero_ps();
    const __m256 p      = _mm256_set1_ps(0.2316419f);
    const __m256 b1     = _mm256_set1_ps( 0.319381530f);
    const __m256 b2     = _mm256_set1_ps(-0.356563782f);
    const __m256 b3     = _mm256_set1_ps( 1.781477937f);
    const __m256 b4     = _mm256_set1_ps(-1.821255978f);
    const __m256 b5     = _mm256_set1_ps( 1.330274429f);
    const __m256 inv_sqrt2pi = _mm256_set1_ps(0.398942280f);

    // Work with |x|, apply sign correction at end
    __m256 abs_x  = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);  // abs via sign bit clear
    __m256 neg_mask = _mm256_cmp_ps(x, zeros, _CMP_LT_OS);        // mask where x < 0

    // t = 1 / (1 + p*|x|)
    __m256 t = _mm256_rcp_ps(_mm256_fmadd_ps(p, abs_x, ones));

    // pdf(|x|) = (1/sqrt(2pi)) * exp(-x²/2)
    __m256 x2   = _mm256_mul_ps(abs_x, abs_x);
    __m256 exp_arg = _mm256_mul_ps(x2, _mm256_set1_ps(-0.5f));

    // exp approximation via: exp(x) = 2^(x/ln2), use bit manipulation
    // We use the Cephes method: split into integer and fractional parts
    const __m256 log2e  = _mm256_set1_ps(1.44269504f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 half   = _mm256_set1_ps(0.5f);

    __m256 fx = _mm256_fmadd_ps(exp_arg, log2e, half);
    fx = _mm256_floor_ps(fx);
    __m256 tmp = _mm256_fnmadd_ps(fx, ln2_hi, exp_arg);
    tmp = _mm256_fnmadd_ps(fx, ln2_lo, tmp);

    // Polynomial for exp on [0, ln2]
    const __m256 c2 = _mm256_set1_ps(0.5000000f);
    const __m256 c3 = _mm256_set1_ps(0.1666667f);
    const __m256 c4 = _mm256_set1_ps(0.0416667f);
    const __m256 c5 = _mm256_set1_ps(0.0083333f);
    __m256 y = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_fmadd_ps(c5, tmp, c4), tmp, c3), tmp, c2), tmp, ones);
    y = _mm256_fmadd_ps(y, tmp, ones);

    // Scale by 2^fx via exponent field manipulation
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);
    __m256 exp_val = _mm256_mul_ps(y, pow2n);

    __m256 pdf_val = _mm256_mul_ps(inv_sqrt2pi, exp_val);

    // Horner evaluation of polynomial tail: t*(b1 + t*(b2 + t*(b3 + t*(b4 + t*b5))))
    __m256 poly = _mm256_fmadd_ps(b5, t, b4);
    poly = _mm256_fmadd_ps(poly, t, b3);
    poly = _mm256_fmadd_ps(poly, t, b2);
    poly = _mm256_fmadd_ps(poly, t, b1);
    poly = _mm256_mul_ps(poly, t);

    // N(|x|) = 1 - pdf * poly
    __m256 cdf_pos = _mm256_fnmadd_ps(pdf_val, poly, ones);

    // N(x) = N(|x|) if x >= 0, else 1 - N(|x|)
    __m256 cdf_neg = _mm256_sub_ps(ones, cdf_pos);
    return _mm256_blendv_ps(cdf_pos, cdf_neg, neg_mask);
}

// log(x) approximation via mantissa extraction + polynomial
static inline __m256 avx2_log(__m256 x) {
    // Extract exponent and mantissa
    __m256i xi     = _mm256_castps_si256(x);
    __m256i exp_i  = _mm256_srli_epi32(_mm256_and_si256(xi, _mm256_set1_epi32(0x7F800000)), 23);
    exp_i          = _mm256_sub_epi32(exp_i, _mm256_set1_epi32(127));
    __m256  exp_f  = _mm256_cvtepi32_ps(exp_i);

    // Normalise mantissa to [1, 2)
    __m256i mant_i = _mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)),
        _mm256_set1_epi32(0x3F800000));
    __m256 m = _mm256_castsi256_ps(mant_i);

    // Polynomial approximation of log on [1, 2): log(1+t) for t = m-1
    __m256 t = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    const __m256 c1 = _mm256_set1_ps(-0.5f);
    const __m256 c2 = _mm256_set1_ps( 0.333333f);
    const __m256 c3 = _mm256_set1_ps(-0.25f);
    const __m256 c4 = _mm256_set1_ps( 0.2f);

    __m256 poly = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(c4, t, c3), t, c2), t, c1);
    poly = _mm256_fmadd_ps(poly, _mm256_mul_ps(t, t), t);

    // log(x) = log(2) * exp + poly
    return _mm256_fmadd_ps(exp_f, _mm256_set1_ps(0.693147180f), poly);
}

// ── Core AVX2 Black-Scholes — prices 8 options simultaneously ──────────────
void SIMDBatchPricer::priceAVX2(const BatchInput& in, float* out, std::size_t n) {
    const std::size_t vec_n = (n / 8) * 8;  // Round down to multiple of 8

    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 ones  = _mm256_set1_ps(1.0f);
    const __m256 zeros = _mm256_setzero_ps();

    for (std::size_t i = 0; i < vec_n; i += 8) {
        // Load 8 options
        __m256 S     = _mm256_loadu_ps(in.S     + i);
        __m256 K     = _mm256_loadu_ps(in.K     + i);
        __m256 r     = _mm256_loadu_ps(in.r     + i);
        __m256 q     = _mm256_loadu_ps(in.q     + i);
        __m256 sigma = _mm256_loadu_ps(in.sigma  + i);
        __m256 T     = _mm256_loadu_ps(in.T     + i);

        // sqrtT = sqrt(T)
        __m256 sqrtT = _mm256_sqrt_ps(T);

        // d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrtT)
        __m256 log_SK   = avx2_log(_mm256_div_ps(S, K));
        __m256 sigma2   = _mm256_mul_ps(sigma, sigma);
        __m256 rq_drift = _mm256_sub_ps(r, q);
        __m256 drift    = _mm256_fmadd_ps(half, sigma2, rq_drift);  // r-q + 0.5*sigma^2
        __m256 d1       = _mm256_div_ps(
            _mm256_fmadd_ps(drift, T, log_SK),
            _mm256_mul_ps(sigma, sqrtT));
        __m256 d2 = _mm256_sub_ps(d1, _mm256_mul_ps(sigma, sqrtT));

        // Discount factors: df = exp(-r*T), df_q = exp(-q*T)
        // We reuse the exp approximation via: exp(x) = exp(x)
        // For simplicity, use the series: exp(-r*T) ≈ computed inline
        // We encode this as avx2_log in reverse isn't clean — use scalar exp
        // for the discount factors (they're computed once per batch, not per path)
        // Actually: compute via the exp polynomial above, applied to -r*T
        __m256 neg_rT  = _mm256_mul_ps(_mm256_mul_ps(r, T), _mm256_set1_ps(-1.0f));
        __m256 neg_qT  = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));

        // Inline exp for discount factors using same Cephes method
        auto avx2_exp = [&](const __m256& arg) -> __m256 {
            const __m256 log2e  = _mm256_set1_ps(1.44269504f);
            const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
            const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
            __m256 fx = _mm256_fmadd_ps(arg, log2e, half);
            fx = _mm256_floor_ps(fx);
            __m256 tmp = _mm256_fnmadd_ps(fx, ln2_hi, arg);
            tmp = _mm256_fnmadd_ps(fx, ln2_lo, tmp);
            const __m256 c2 = _mm256_set1_ps(0.5000000f);
            const __m256 c3 = _mm256_set1_ps(0.1666667f);
            const __m256 c4 = _mm256_set1_ps(0.0416667f);
            const __m256 c5 = _mm256_set1_ps(0.0083333f);
            __m256 y = _mm256_fmadd_ps(
                _mm256_fmadd_ps(
                    _mm256_fmadd_ps(
                        _mm256_fmadd_ps(c5, tmp, c4), tmp, c3), tmp, c2), tmp, ones);
            y = _mm256_fmadd_ps(y, tmp, ones);
            __m256i emm0 = _mm256_cvttps_epi32(fx);
            emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
            emm0 = _mm256_slli_epi32(emm0, 23);
            return _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
        };

        __m256 df   = avx2_exp(neg_rT);   // e^{-rT}
        __m256 df_q = avx2_exp(neg_qT);   // e^{-qT}

        // N(d1), N(d2), N(-d1), N(-d2)
        __m256 Nd1  = avx2_norm_cdf(d1);
        __m256 Nd2  = avx2_norm_cdf(d2);
        __m256 Nd1n = _mm256_sub_ps(ones, Nd1);
        __m256 Nd2n = _mm256_sub_ps(ones, Nd2);

        // Call price = S*e^{-qT}*N(d1) - K*e^{-rT}*N(d2)
        __m256 call_price = _mm256_fmsub_ps(
            _mm256_mul_ps(S, df_q), Nd1,
            _mm256_mul_ps(_mm256_mul_ps(K, df), Nd2));

        // Put price = K*e^{-rT}*N(-d2) - S*e^{-qT}*N(-d1)
        __m256 put_price = _mm256_fmsub_ps(
            _mm256_mul_ps(K, df), Nd2n,
            _mm256_mul_ps(_mm256_mul_ps(S, df_q), Nd1n));

        // Select call or put based on type flag
        // type[i] == 0 => call, type[i] == 1 => put
        // Build a float mask: 0x00000000 for call, 0xFFFFFFFF for put
        __m256i type_i = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(in.type + i));
        // Convert int mask to float blend mask (non-zero = put)
        __m256 put_mask = _mm256_castsi256_ps(
            _mm256_cmpeq_epi32(type_i, _mm256_set1_epi32(1)));

        __m256 result = _mm256_blendv_ps(call_price, put_price, put_mask);

        _mm256_storeu_ps(out + i, result);
    }

    // Scalar tail for remaining < 8 elements
    if (vec_n < n) {
        BatchInput tail = in;
        tail.S     += vec_n;
        tail.K     += vec_n;
        tail.r     += vec_n;
        tail.q     += vec_n;
        tail.sigma += vec_n;
        tail.T     += vec_n;
        tail.type  += vec_n;
        priceScalar(tail, out + vec_n, n - vec_n);
    }
}

void SIMDBatchPricer::greeksAVX2(const BatchInput& in, BatchOutput& out, std::size_t n) {
    // Price pass first
    priceAVX2(in, out.prices, n);

    // Delta: reuse d1 — for calls: e^{-qT}*N(d1), puts: e^{-qT}*(N(d1)-1)
    // We recompute d1 here; in a production system you'd cache it from the price pass.
    for (std::size_t i = 0; i < n; ++i) {
        BSParams p{in.S[i], in.K[i], in.r[i], in.q[i], in.sigma[i], in.T[i],
                   in.type[i] == 0 ? OptionType::Call : OptionType::Put};
        const auto res = BlackScholesEngine::price(p);
        if (out.delta) out.delta[i] = static_cast<float>(res.greeks.delta);
        if (out.gamma) out.gamma[i] = static_cast<float>(res.greeks.gamma);
        if (out.vega ) out.vega [i] = static_cast<float>(res.greeks.vega);
    }
}

#endif // PRICING_HAS_AVX2

// ─────────────────────────────────────────────
//  SSE4.1 implementation — 4 options per iteration
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
//  NEON implementation — 4 float options per iteration
//
//  Runs natively on Apple Silicon (M1/M2/M3/M4) and ARM Cortex.
//  Uses the same Cephes/A&S polynomial approximations as the AVX2
//  path, ported to ARM float32x4_t intrinsics.
//
//  Naming conventions:
//    vdupq_n_f32(x)   — broadcast scalar x to all 4 lanes
//    vaddq_f32(a,b)   — a + b elementwise
//    vmulq_f32(a,b)   — a * b elementwise
//    vfmaq_f32(c,a,b) — c + a*b  (FMA, available on ARMv8)
//    vcgtq_f32(a,b)   — a > b bitmask
//    vbslq_f32(m,a,b) — blend: m ? a : b
// ─────────────────────────────────────────────
#if PRICING_HAS_NEON

// ── neon_exp: Cephes method, exp(x) = 2^(x/ln2) ────────────────────────────
// vfmaq_f32(c, a, b) = c + a*b
// Horner: ((((c5*t + c4)*t + c3)*t + c2)*t + 1)*t + 1
static inline float32x4_t neon_exp(float32x4_t arg) {
    const float32x4_t log2e  = vdupq_n_f32(1.44269504f);
    const float32x4_t ln2_hi = vdupq_n_f32(0.693359375f);
    const float32x4_t ln2_lo = vdupq_n_f32(-2.12194440e-4f);
    const float32x4_t half   = vdupq_n_f32(0.5f);
    const float32x4_t ones   = vdupq_n_f32(1.0f);

    // fx = floor(arg * log2e + 0.5)
    float32x4_t fx  = vrndmq_f32(vaddq_f32(vmulq_f32(arg, log2e), half));
    // tmp = arg - fx*ln2  (two-step for accuracy)
    float32x4_t tmp = vfmsq_f32(arg, fx, ln2_hi);   // arg - fx*ln2_hi
    tmp             = vfmsq_f32(tmp, fx, ln2_lo);   // tmp - fx*ln2_lo

    // Polynomial: 1 + t*(1 + t*(1/2 + t*(1/6 + t*(1/24 + t/120))))
    float32x4_t y = vdupq_n_f32(1.0f / 120.0f);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f),  y, tmp);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f),   y, tmp);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 2.0f),   y, tmp);
    y = vfmaq_f32(ones,                         y, tmp);
    y = vfmaq_f32(ones,                         y, tmp);

    // Scale by 2^fx: set exponent field of float
    int32x4_t emm0 = vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127));
    emm0           = vshlq_n_s32(emm0, 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(emm0));
}

// ── neon_log: atanh-based, log(x) = 2*atanh((m-1)/(m+1)) + exp_f*ln2 ──────
static inline float32x4_t neon_log(float32x4_t x) {
    const float32x4_t ones  = vdupq_n_f32(1.0f);
    const float32x4_t sqrt2 = vdupq_n_f32(1.4142135f);
    const float32x4_t ln2   = vdupq_n_f32(0.693147180f);

    // Extract biased exponent, subtract 127
    int32x4_t xi    = vreinterpretq_s32_f32(x);
    int32x4_t exp_i = vsubq_s32(
        vshrq_n_s32(vandq_s32(xi, vdupq_n_s32(0x7F800000)), 23),
        vdupq_n_s32(127));
    float32x4_t exp_f = vcvtq_f32_s32(exp_i);

    // Normalise mantissa to [1, 2)
    int32x4_t mant_i = vorrq_s32(
        vandq_s32(xi, vdupq_n_s32(0x007FFFFF)),
        vdupq_n_s32(0x3F800000));
    float32x4_t m = vreinterpretq_f32_s32(mant_i);

    // Further reduce [1,2) → [1, sqrt(2)]: if m > sqrt(2), halve it and add 1 to exp
    uint32x4_t gt = vcgtq_f32(m, sqrt2);
    m     = vbslq_f32(gt, vmulq_f32(m, vdupq_n_f32(0.5f)), m);
    exp_f = vaddq_f32(exp_f, vreinterpretq_f32_u32(vandq_u32(gt, vreinterpretq_u32_f32(ones))));

    // t = (m-1)/(m+1),  log(m) = 2*t*(1 + t²*(1/3 + t²*(1/5 + t²/7)))
    float32x4_t t  = vdivq_f32(vsubq_f32(m, ones), vaddq_f32(m, ones));
    float32x4_t t2 = vmulq_f32(t, t);

    // Horner for atanh polynomial P(t²): 1 + t²*(1/3 + t²*(1/5 + t²/7))
    float32x4_t poly = vdupq_n_f32(1.0f / 7.0f);
    poly = vfmaq_f32(vdupq_n_f32(1.0f / 5.0f), poly, t2);
    poly = vfmaq_f32(vdupq_n_f32(1.0f / 3.0f), poly, t2);
    poly = vfmaq_f32(ones,                      poly, t2);

    // log(m) = 2*t*P(t²),  log(x) = log(m) + exp_f*ln2
    float32x4_t log_m = vmulq_f32(vmulq_f32(vdupq_n_f32(2.0f), t), poly);
    return vfmaq_f32(log_m, exp_f, ln2);
}

// ── neon_norm_cdf: Abramowitz & Stegun §26.2.17 ─────────────────────────────
// t = 1/(1 + 0.2316419*|x|),  N(x) = 1 - pdf(x)*poly(t) for x >= 0
static inline float32x4_t neon_norm_cdf(float32x4_t x) {
    const float32x4_t ones        = vdupq_n_f32(1.0f);
    const float32x4_t zeros       = vdupq_n_f32(0.0f);
    const float32x4_t inv_sqrt2pi = vdupq_n_f32(0.398942280f);
    const float32x4_t p_coef      = vdupq_n_f32(0.2316419f);

    float32x4_t abs_x    = vabsq_f32(x);
    uint32x4_t  neg_mask = vcltq_f32(x, zeros);

    // Accurate t = 1 / (1 + p*|x|) via two Newton-Raphson refinements on vrecpeq
    // vrecpeq gives ~8-bit estimate; each NR step doubles precision
    float32x4_t denom = vfmaq_f32(ones, p_coef, abs_x);  // 1 + p*|x|
    float32x4_t t     = vrecpeq_f32(denom);               // ~8-bit estimate
    t = vmulq_f32(t, vfmsq_f32(vdupq_n_f32(2.0f), t, denom));  // NR step 1 -> ~16-bit
    t = vmulq_f32(t, vfmsq_f32(vdupq_n_f32(2.0f), t, denom));  // NR step 2 -> ~24-bit

    // pdf(|x|) = (1/sqrt(2pi)) * exp(-|x|²/2)
    float32x4_t exp_arg = vmulq_f32(vmulq_f32(abs_x, abs_x), vdupq_n_f32(-0.5f));
    float32x4_t pdf_v   = vmulq_f32(inv_sqrt2pi, neon_exp(exp_arg));

    // A&S polynomial in t (Horner): t*(b1 + t*(b2 + t*(b3 + t*(b4 + t*b5))))
    float32x4_t poly = vdupq_n_f32(1.330274429f);
    poly = vfmaq_f32(vdupq_n_f32(-1.821255978f), poly, t);
    poly = vfmaq_f32(vdupq_n_f32( 1.781477937f), poly, t);
    poly = vfmaq_f32(vdupq_n_f32(-0.356563782f), poly, t);
    poly = vfmaq_f32(vdupq_n_f32( 0.319381530f), poly, t);
    poly = vmulq_f32(poly, t);

    // N(|x|) = 1 - pdf * poly
    float32x4_t cdf_pos = vfmsq_f32(ones, pdf_v, poly);
    float32x4_t cdf_neg = vsubq_f32(ones, cdf_pos);
    return vbslq_f32(neg_mask, cdf_neg, cdf_pos);
}

void SIMDBatchPricer::priceNEON(const BatchInput& in, float* out, std::size_t n) {
    const std::size_t vec_n = (n / 4) * 4;

    const float32x4_t ones = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t neg1 = vdupq_n_f32(-1.0f);

    for (std::size_t i = 0; i < vec_n; i += 4) {
        float32x4_t S     = vld1q_f32(in.S     + i);
        float32x4_t K     = vld1q_f32(in.K     + i);
        float32x4_t r     = vld1q_f32(in.r     + i);
        float32x4_t q     = vld1q_f32(in.q     + i);
        float32x4_t sigma = vld1q_f32(in.sigma  + i);
        float32x4_t T     = vld1q_f32(in.T     + i);

        float32x4_t sqrtT  = vsqrtq_f32(T);
        float32x4_t sigma2 = vmulq_f32(sigma, sigma);

        // d1 = (log(S/K) + (r - q + 0.5*sigma²)*T) / (sigma*sqrtT)
        float32x4_t log_SK = neon_log(vdivq_f32(S, K));
        float32x4_t drift  = vfmaq_f32(vsubq_f32(r, q), half, sigma2);
        float32x4_t d1     = vdivq_f32(
            vfmaq_f32(log_SK, drift, T),
            vmulq_f32(sigma, sqrtT));
        float32x4_t d2 = vfmsq_f32(d1, sigma, sqrtT);

        // Discount factors
        float32x4_t df   = neon_exp(vmulq_f32(vmulq_f32(r, T), neg1));
        float32x4_t df_q = neon_exp(vmulq_f32(vmulq_f32(q, T), neg1));

        float32x4_t Nd1  = neon_norm_cdf(d1);
        float32x4_t Nd2  = neon_norm_cdf(d2);
        float32x4_t Nd1n = vsubq_f32(ones, Nd1);
        float32x4_t Nd2n = vsubq_f32(ones, Nd2);

        // Call = S*df_q*N(d1) - K*df*N(d2)
        float32x4_t call_p = vfmsq_f32(
            vmulq_f32(vmulq_f32(S, df_q), Nd1),
            vmulq_f32(K, df), Nd2);
        // Put = K*df*N(-d2) - S*df_q*N(-d1)
        float32x4_t put_p = vfmsq_f32(
            vmulq_f32(vmulq_f32(K, df), Nd2n),
            vmulq_f32(S, df_q), Nd1n);

        // Select call/put via type mask
        int32x4_t   type_i   = vld1q_s32(in.type + i);
        uint32x4_t  put_mask = vceqq_s32(type_i, vdupq_n_s32(1));
        float32x4_t result   = vbslq_f32(put_mask, put_p, call_p);

        vst1q_f32(out + i, result);
    }

    // Scalar tail
    if (vec_n < n) {
        BatchInput tail = in;
        tail.S += vec_n; tail.K += vec_n; tail.r += vec_n;
        tail.q += vec_n; tail.sigma += vec_n; tail.T += vec_n;
        tail.type += vec_n;
        priceScalar(tail, out + vec_n, n - vec_n);
    }
}

#endif // PRICING_HAS_NEON

#if PRICING_HAS_SSE4

static inline __m128 sse4_norm_cdf(__m128 x) {
    const __m128 ones  = _mm_set1_ps(1.0f);
    const __m128 zeros = _mm_setzero_ps();
    const __m128 p     = _mm_set1_ps(0.2316419f);
    const __m128 b1    = _mm_set1_ps( 0.319381530f);
    const __m128 b2    = _mm_set1_ps(-0.356563782f);
    const __m128 b3    = _mm_set1_ps( 1.781477937f);
    const __m128 b4    = _mm_set1_ps(-1.821255978f);
    const __m128 b5    = _mm_set1_ps( 1.330274429f);
    const __m128 inv_sqrt2pi = _mm_set1_ps(0.398942280f);

    __m128 abs_x   = _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
    __m128 neg_mask = _mm_cmplt_ps(x, zeros);
    __m128 t = _mm_rcp_ps(_mm_add_ps(_mm_mul_ps(p, abs_x), ones));

    __m128 x2      = _mm_mul_ps(abs_x, abs_x);
    __m128 exp_arg = _mm_mul_ps(x2, _mm_set1_ps(-0.5f));

    // Simple exp approximation for SSE (less accurate but fast)
    const __m128 log2e = _mm_set1_ps(1.44269504f);
    const __m128 half  = _mm_set1_ps(0.5f);
    __m128 fx = _mm_add_ps(_mm_mul_ps(exp_arg, log2e), half);
    fx = _mm_floor_ps(fx);  // SSE4.1 _mm_floor_ps
    __m128 tmp = _mm_sub_ps(exp_arg, _mm_mul_ps(fx, _mm_set1_ps(0.693147180f)));
    __m128 y = _mm_add_ps(_mm_add_ps(
        _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.0416667f), tmp),
                               _mm_set1_ps(0.1666667f)), tmp),
        _mm_mul_ps(_mm_set1_ps(0.5f), tmp)), ones);
    y = _mm_add_ps(_mm_mul_ps(y, tmp), ones);
    __m128i emm0 = _mm_add_epi32(_mm_cvttps_epi32(fx), _mm_set1_epi32(127));
    emm0 = _mm_slli_epi32(emm0, 23);
    __m128 exp_val = _mm_mul_ps(y, _mm_castsi128_ps(emm0));
    __m128 pdf_val = _mm_mul_ps(inv_sqrt2pi, exp_val);

    __m128 poly = _mm_add_ps(_mm_mul_ps(b5, t), b4);
    poly = _mm_add_ps(_mm_mul_ps(poly, t), b3);
    poly = _mm_add_ps(_mm_mul_ps(poly, t), b2);
    poly = _mm_add_ps(_mm_mul_ps(poly, t), b1);
    poly = _mm_mul_ps(poly, t);

    __m128 cdf_pos = _mm_sub_ps(ones, _mm_mul_ps(pdf_val, poly));
    __m128 cdf_neg = _mm_sub_ps(ones, cdf_pos);
    return _mm_blendv_ps(cdf_pos, cdf_neg, neg_mask);
}

void SIMDBatchPricer::priceSSE4(const BatchInput& in, float* out, std::size_t n) {
    // SSE4.1 float polynomial approximations accumulate too much error for
    // Black-Scholes (log/exp/norm_cdf compound to ~5-15% option price error).
    // On this path we use double-precision scalar arithmetic — the compiler's
    // auto-vectoriser will emit NEON/SSE2 from this loop anyway, and we get
    // correct results without hand-rolling double-precision SIMD intrinsics.
    for (std::size_t i = 0; i < n; ++i) {
        const double S     = static_cast<double>(in.S    [i]);
        const double K     = static_cast<double>(in.K    [i]);
        const double r     = static_cast<double>(in.r    [i]);
        const double q     = static_cast<double>(in.q    [i]);
        const double sigma = static_cast<double>(in.sigma[i]);
        const double T     = static_cast<double>(in.T    [i]);

        const double sqrtT = std::sqrt(T);
        const double d1    = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T)
                             / (sigma * sqrtT);
        const double d2    = d1 - sigma * sqrtT;

        const double df    = std::exp(-r * T);
        const double df_q  = std::exp(-q * T);
        const double Nd1   = 0.5 * std::erfc(-d1 * std::numbers::inv_sqrt2);
        const double Nd2   = 0.5 * std::erfc(-d2 * std::numbers::inv_sqrt2);

        const double price = (in.type[i] == 0)
            ? S * df_q * Nd1 - K * df * Nd2
            : K * df * (1.0 - Nd2) - S * df_q * (1.0 - Nd1);

        out[i] = static_cast<float>(price);
    }
}

#endif // PRICING_HAS_SSE4

// ─────────────────────────────────────────────
//  Public dispatch
// ─────────────────────────────────────────────
void SIMDBatchPricer::price(const BatchInput& in, float* out_prices, std::size_t n) {
#if PRICING_HAS_AVX2
    priceAVX2(in, out_prices, n);
#elif PRICING_HAS_NEON
    priceNEON(in, out_prices, n);
#elif PRICING_HAS_SSE4
    priceSSE4(in, out_prices, n);
#else
    priceScalar(in, out_prices, n);
#endif
}

void SIMDBatchPricer::priceAndGreeks(const BatchInput& in, BatchOutput& out, std::size_t n) {
#if PRICING_HAS_AVX2
    greeksAVX2(in, out, n);
#else
    // Fallback: scalar Greeks
    if (out.prices) priceScalar(in, out.prices, n);
    for (std::size_t i = 0; i < n; ++i) {
        BSParams p{in.S[i], in.K[i], in.r[i], in.q[i], in.sigma[i], in.T[i],
                   in.type[i] == 0 ? OptionType::Call : OptionType::Put};
        const auto res = BlackScholesEngine::price(p);
        if (out.delta) out.delta[i] = static_cast<float>(res.greeks.delta);
        if (out.gamma) out.gamma[i] = static_cast<float>(res.greeks.gamma);
        if (out.vega ) out.vega [i] = static_cast<float>(res.greeks.vega);
    }
#endif
}

std::vector<float> SIMDBatchPricer::price(const std::vector<BSParams>& params) {
    std::vector<float> S_buf, K_buf, r_buf, q_buf, sigma_buf, T_buf;
    std::vector<int>   type_buf;
    BatchInput::fill(params, S_buf, K_buf, r_buf, q_buf, sigma_buf, T_buf, type_buf);

    BatchInput in{S_buf.data(), K_buf.data(), r_buf.data(), q_buf.data(),
                  sigma_buf.data(), T_buf.data(), type_buf.data(), params.size()};

    std::vector<float> out(params.size());
    price(in, out.data(), params.size());
    return out;
}

} // namespace pricing