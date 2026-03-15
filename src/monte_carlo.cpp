#include "monte_carlo.h"
#include "black_scholes.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pricing {

// ─────────────────────────────────────────────
//  Validation
// ─────────────────────────────────────────────
void MCParams::validate() const {
    if (S        <= 0.0) throw std::invalid_argument("S must be > 0");
    if (K        <= 0.0) throw std::invalid_argument("K must be > 0");
    if (sigma    <= 0.0) throw std::invalid_argument("sigma must be > 0");
    if (T        <= 0.0) throw std::invalid_argument("T must be > 0");
    if (numPaths <  1  ) throw std::invalid_argument("numPaths must be >= 1");
    if (numSteps <  1  ) throw std::invalid_argument("numSteps must be >= 1");
    if (q        <  0.0) throw std::invalid_argument("q must be >= 0");

    const bool is_barrier = (exotic == ExoticType::BarrierUpOut  ||
                              exotic == ExoticType::BarrierUpIn   ||
                              exotic == ExoticType::BarrierDownOut ||
                              exotic == ExoticType::BarrierDownIn);
    if (is_barrier && barrier <= 0.0)
        throw std::invalid_argument("barrier must be > 0 for barrier options");
}

// ─────────────────────────────────────────────
//  Xorshift64 + Box-Muller normal variate
// ─────────────────────────────────────────────
double MonteCarloEngine::randNormal(uint64_t& state) {
    // Xorshift64
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    const double u1 = static_cast<double>(state >> 11) / static_cast<double>(1ULL << 53);

    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    const double u2 = static_cast<double>(state >> 11) / static_cast<double>(1ULL << 53);

    // Box-Muller transform — return one of the two normals
    return std::sqrt(-2.0 * std::log(u1 + 1e-300)) *
           std::cos(2.0 * std::numbers::pi * u2);
}

// ─────────────────────────────────────────────
//  Payoff computation from a simulated price path
// ─────────────────────────────────────────────
double MonteCarloEngine::payoff(const MCParams& p, const std::vector<double>& path) {
    const double S_T = path.back();

    switch (p.exotic) {

    case ExoticType::European: {
        return (p.type == OptionType::Call)
            ? std::max(S_T - p.K, 0.0)
            : std::max(p.K - S_T, 0.0);
    }

    case ExoticType::AsianArith: {
        // Arithmetic average of all path prices (including S_0 is convention-dependent;
        // here we average over monitoring dates only, i.e. path[1..N])
        const double avg = std::accumulate(path.begin() + 1, path.end(), 0.0) /
                           static_cast<double>(path.size() - 1);
        return (p.type == OptionType::Call)
            ? std::max(avg - p.K, 0.0)
            : std::max(p.K - avg, 0.0);
    }

    case ExoticType::AsianGeo: {
        // Geometric average: exp(mean of log prices)
        double logsum = 0.0;
        for (std::size_t i = 1; i < path.size(); ++i)
            logsum += std::log(path[i]);
        const double geo_avg = std::exp(logsum / static_cast<double>(path.size() - 1));
        return (p.type == OptionType::Call)
            ? std::max(geo_avg - p.K, 0.0)
            : std::max(p.K - geo_avg, 0.0);
    }

    case ExoticType::BarrierUpOut: {
        // Knocked out if any S_t >= barrier
        for (std::size_t i = 1; i < path.size(); ++i)
            if (path[i] >= p.barrier) return 0.0;
        return (p.type == OptionType::Call)
            ? std::max(S_T - p.K, 0.0)
            : std::max(p.K - S_T, 0.0);
    }

    case ExoticType::BarrierUpIn: {
        // Active only if S_t >= barrier at some point
        bool knocked_in = false;
        for (std::size_t i = 1; i < path.size(); ++i)
            if (path[i] >= p.barrier) { knocked_in = true; break; }
        if (!knocked_in) return 0.0;
        return (p.type == OptionType::Call)
            ? std::max(S_T - p.K, 0.0)
            : std::max(p.K - S_T, 0.0);
    }

    case ExoticType::BarrierDownOut: {
        // Knocked out if any S_t <= barrier
        for (std::size_t i = 1; i < path.size(); ++i)
            if (path[i] <= p.barrier) return 0.0;
        return (p.type == OptionType::Call)
            ? std::max(S_T - p.K, 0.0)
            : std::max(p.K - S_T, 0.0);
    }

    case ExoticType::BarrierDownIn: {
        // Active only if S_t <= barrier at some point
        bool knocked_in = false;
        for (std::size_t i = 1; i < path.size(); ++i)
            if (path[i] <= p.barrier) { knocked_in = true; break; }
        if (!knocked_in) return 0.0;
        return (p.type == OptionType::Call)
            ? std::max(S_T - p.K, 0.0)
            : std::max(p.K - S_T, 0.0);
    }

    case ExoticType::Lookback: {
        // Fixed-strike lookback: call on minimum, put on maximum
        // Call payoff = max(S_min - K, 0), Put payoff = max(K - S_max, 0)
        double S_min = path[1], S_max = path[1];
        for (std::size_t i = 2; i < path.size(); ++i) {
            S_min = std::min(S_min, path[i]);
            S_max = std::max(S_max, path[i]);
        }
        return (p.type == OptionType::Call)
            ? std::max(S_min - p.K, 0.0)
            : std::max(p.K - S_max, 0.0);
    }

    default:
        throw std::invalid_argument("Unknown ExoticType");
    }
}

// ─────────────────────────────────────────────
//  Simulate a block of paths
//
//  Generates GBM paths via Euler-Maruyama:
//    S_{t+dt} = S_t * exp((r - q - sigma²/2)*dt + sigma*sqrt(dt)*Z)
//
//  Antithetic variates: for each Z drawn, also simulate with -Z.
//  The payoff stored is the average of the two, which preserves the
//  mean but reduces variance for monotone payoffs.
// ─────────────────────────────────────────────
MonteCarloEngine::PathStats MonteCarloEngine::simulateBlock(
    const MCParams& p,
    int             blockPaths,
    uint64_t        threadSeed)
{
    const int    N     = p.numSteps;
    const double dt    = p.T / N;
    const double drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
    const double vol   = p.sigma * std::sqrt(dt);
    const double df    = std::exp(-p.r * p.T);

    const bool use_antithetic = (p.vr == VarianceReduction::Antithetic ||
                                  p.vr == VarianceReduction::AntitheticAndControl);

    uint64_t rng = threadSeed;

    std::vector<double> path(N + 1);
    std::vector<double> path_anti(N + 1);

    PathStats stats{0.0, 0.0, 0};

    const int effective_paths = use_antithetic ? blockPaths / 2 : blockPaths;

    for (int i = 0; i < effective_paths; ++i) {
        // Simulate forward path
        path[0] = p.S;
        if (use_antithetic) path_anti[0] = p.S;

        for (int t = 0; t < N; ++t) {
            const double Z = randNormal(rng);
            path[t + 1]   = path[t]   * std::exp(drift + vol * Z);
            if (use_antithetic)
                path_anti[t + 1] = path_anti[t] * std::exp(drift - vol * Z);
        }

        const double pv = use_antithetic
            ? 0.5 * (payoff(p, path) + payoff(p, path_anti))
            : payoff(p, path);

        const double discounted = df * pv;
        stats.sum   += discounted;
        stats.sumSq += discounted * discounted;
        stats.count += 1;
    }

    return stats;
}

// ─────────────────────────────────────────────
//  Main simulation entry point (parallel over path blocks)
// ─────────────────────────────────────────────
MCResult MonteCarloEngine::priceOnly(const MCParams& p) {
    p.validate();

    // Determine thread count
    int nThreads = 1;
#ifdef _OPENMP
    nThreads = omp_get_max_threads();
#endif

    const int pathsPerThread = (p.numPaths + nThreads - 1) / nThreads;

    // Per-thread accumulators
    std::vector<double> threadSum  (nThreads, 0.0);
    std::vector<double> threadSumSq(nThreads, 0.0);
    std::vector<long>   threadCount(nThreads, 0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < nThreads; ++t) {
        // Derive a unique, deterministic seed per thread
        const uint64_t threadSeed = p.seed ^ (static_cast<uint64_t>(t + 1) * 0x9E3779B97F4A7C15ULL);
        const int      blockPaths = (t == nThreads - 1)
            ? p.numPaths - t * pathsPerThread
            : pathsPerThread;

        const auto stats = simulateBlock(p, blockPaths, threadSeed);
        threadSum  [t] = stats.sum;
        threadSumSq[t] = stats.sumSq;
        threadCount[t] = stats.count;
    }

    // Reduce across threads
    double totalSum   = 0.0;
    double totalSumSq = 0.0;
    long   totalCount = 0;
    for (int t = 0; t < nThreads; ++t) {
        totalSum   += threadSum  [t];
        totalSumSq += threadSumSq[t];
        totalCount += threadCount[t];
    }

    // ── Control variate correction ──
    // Uses the European BS price as a control. We price a European with the
    // same params via MC, compare to the known BS price, and adjust.
    double cvCorrection = 0.0;
    if (p.vr == VarianceReduction::ControlVariate ||
        p.vr == VarianceReduction::AntitheticAndControl)
    {
        MCParams euro_p  = p;
        euro_p.exotic    = ExoticType::European;
        euro_p.vr        = VarianceReduction::None;
        const auto euro_mc = priceOnly(euro_p);

        BSParams bs_p{p.S, p.K, p.r, p.q, p.sigma, p.T, p.type};
        const double bs_price = BlackScholesEngine::price(bs_p).price;

        cvCorrection = bs_price - euro_mc.price;
    }

    const double mean   = totalSum / static_cast<double>(totalCount);
    const double meanSq = totalSumSq / static_cast<double>(totalCount);
    const double var    = (meanSq - mean * mean) / static_cast<double>(totalCount - 1);
    const double se     = std::sqrt(std::max(var, 0.0));

    MCResult result{};
    result.price    = mean + cvCorrection;
    result.stdError = se;
    result.confLo   = result.price - 1.96 * se;
    result.confHi   = result.price + 1.96 * se;
    result.pathsRun = totalCount;
    return result;
}

// ─────────────────────────────────────────────
//  Greeks via finite difference on priceOnly()
//
//  All bumps use central differences except theta (one-sided,
//  since we can't price with negative time to expiry).
// ─────────────────────────────────────────────
MCResult MonteCarloEngine::price(const MCParams& p) {
    p.validate();

    MCResult result = priceOnly(p);

    // ── Delta ──
    {
        const double h = p.S * 0.01;  // 1% of spot
        MCParams p_up = p; p_up.S = p.S + h; p_up.vr = VarianceReduction::None;
        MCParams p_dn = p; p_dn.S = p.S - h; p_dn.vr = VarianceReduction::None;
        result.greeks.delta = (priceOnly(p_up).price - priceOnly(p_dn).price) / (2.0 * h);
    }

    // ── Gamma ──
    {
        const double h = p.S * 0.01;
        MCParams p_up = p; p_up.S = p.S + h; p_up.vr = VarianceReduction::None;
        MCParams p_dn = p; p_dn.S = p.S - h; p_dn.vr = VarianceReduction::None;
        result.greeks.gamma = (priceOnly(p_up).price - 2.0 * result.price + priceOnly(p_dn).price)
                             / (h * h);
    }

    // ── Theta (per calendar day) ──
    {
        const double h = 2.0 / 365.0;
        MCParams p_fwd = p; p_fwd.T = p.T - h; p_fwd.vr = VarianceReduction::None;
        result.greeks.theta = (priceOnly(p_fwd).price - result.price) / h / 365.0;
    }

    // ── Vega (per 1% vol move) ──
    {
        const double h = 0.01;
        MCParams p_up = p; p_up.sigma = p.sigma + h; p_up.vr = VarianceReduction::None;
        MCParams p_dn = p; p_dn.sigma = p.sigma - h; p_dn.vr = VarianceReduction::None;
        result.greeks.vega = (priceOnly(p_up).price - priceOnly(p_dn).price) / (2.0 * h) * 0.01;
    }

    // ── Rho (per 1% rate move) ──
    {
        const double h = 0.0001;
        MCParams p_up = p; p_up.r = p.r + h; p_up.vr = VarianceReduction::None;
        MCParams p_dn = p; p_dn.r = p.r - h; p_dn.vr = VarianceReduction::None;
        result.greeks.rho = (priceOnly(p_up).price - priceOnly(p_dn).price) / (2.0 * h) / 100.0;
    }

    return result;
}

} // namespace pricing