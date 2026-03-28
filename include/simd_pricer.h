#pragma once

#include "black_scholes.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

// ─────────────────────────────────────────────
//  Runtime SIMD capability detection
// ─────────────────────────────────────────────
#if defined(__AVX2__)
    #define PRICING_HAS_AVX2 1
#else
    #define PRICING_HAS_AVX2 0
#endif

#if defined(__SSE4_1__)
    #define PRICING_HAS_SSE4 1
#else
    #define PRICING_HAS_SSE4 0
#endif

#if defined(__ARM_NEON)
    #define PRICING_HAS_NEON 1
#else
    #define PRICING_HAS_NEON 0
#endif

namespace pricing {

// ─────────────────────────────────────────────
//  Batch input — Structure of Arrays (SoA)
//
//  SoA layout is critical for SIMD: we want to load 8 consecutive
//  S values, 8 consecutive K values, etc. into vector registers.
//  AoS (array of BSParams) would interleave fields and prevent
//  efficient SIMD loads.
//
//  All pointer arrays must be aligned to 32 bytes (AVX2 requirement)
//  or 16 bytes (SSE). Use BatchInput::make() for safe allocation.
// ─────────────────────────────────────────────
struct BatchInput {
    const float* S;      // Spot prices       [n]
    const float* K;      // Strike prices     [n]
    const float* r;      // Risk-free rates   [n]
    const float* q;      // Dividend yields   [n]
    const float* sigma;  // Volatilities      [n]
    const float* T;      // Times to expiry   [n]
    const int*   type;   // 0 = Call, 1 = Put [n]
    std::size_t  n;      // Number of options

    // Convenience: allocate aligned SoA arrays and fill from BSParams vector
    static void fill(
        const std::vector<BSParams>& params,
        std::vector<float>& S_buf,
        std::vector<float>& K_buf,
        std::vector<float>& r_buf,
        std::vector<float>& q_buf,
        std::vector<float>& sigma_buf,
        std::vector<float>& T_buf,
        std::vector<int>&   type_buf
    );
};

// ─────────────────────────────────────────────
//  Batch output — prices only (SoA)
// ─────────────────────────────────────────────
struct BatchOutput {
    float* prices;   // Discounted option prices [n]
    float* delta;    // Greeks (optional, may be nullptr)
    float* gamma;
    float* vega;
};

// ─────────────────────────────────────────────
//  SIMD Batch Pricing Engine
//
//  Prices N options in a single vectorised pass:
//
//  AVX2 path  (Intel Haswell 2013+ / AMD Zen 2019+):
//    8 float options per iteration via 256-bit YMM registers.
//
//  NEON path  (Apple Silicon M1+ / ARM Cortex-A55+):
//    4 float options per iteration via 128-bit NEON registers.
//    Uses Cephes polynomial log/exp and A&S norm_cdf approximations
//    ported to float32x4_t intrinsics. Runs natively — no Rosetta.
//
//  SSE4.1 path (Intel Penryn 2007+ / Apple Silicon via Rosetta):
//    Scalar double loop — correct but no vectorisation benefit.
//
//  Scalar fallback:
//    Delegates to BlackScholesEngine. Used when no SIMD is available.
//
//  Key techniques: Cephes polynomial exp, atanh-based log with range
//  reduction, A&S §26.2.17 norm_cdf, FMA, SoA layout, zero heap
//  allocation in the hot loop.
// ─────────────────────────────────────────────
class SIMDBatchPricer {
public:
    // Price a batch of options. Output arrays must be pre-allocated to size n.
    // Automatically selects AVX2 → SSE4.1 → scalar based on CPU support.
    static void price(const BatchInput& in, float* out_prices, std::size_t n);

    // Price + delta + gamma in one pass (extra cost is minimal since
    // d1/d2 are already computed for pricing)
    static void priceAndGreeks(const BatchInput& in, BatchOutput& out, std::size_t n);

    // Convenience wrapper: takes vector of BSParams, returns vector of prices
    static std::vector<float> price(const std::vector<BSParams>& params);

    // Report which SIMD path will be used at runtime
    static const char* simdLevel();

private:
#if PRICING_HAS_AVX2
    static void priceAVX2  (const BatchInput& in, float* out, std::size_t n);
    static void greeksAVX2 (const BatchInput& in, BatchOutput& out, std::size_t n);
#endif
#if PRICING_HAS_NEON
    static void priceNEON  (const BatchInput& in, float* out, std::size_t n);
#endif
#if PRICING_HAS_SSE4
    static void priceSSE4  (const BatchInput& in, float* out, std::size_t n);
#endif
    static void priceScalar(const BatchInput& in, float* out, std::size_t n);
};

} // namespace pricing