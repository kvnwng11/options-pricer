# Options Pricing Engine

A C++20 library for pricing derivatives and computing risk sensitivities (Greeks). Implements four methods: a Black-Scholes analytical solver for European options; CRR and Trinomial tree models for American options with early exercise; a parallelised Monte Carlo simulator for path-dependent exotics (Asian, Barrier, Lookback); and an AVX2 SIMD batch pricer that processes 8 options per clock cycle achieving ~10x throughput over scalar pricing. All modules are property-based tested with deterministic randomised trials across hundreds of parameter combinations.

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bs_tests
./build/american_tests
./build/mc_tests
./build/simd_tests
./build/bs_demo        # run the demo
```