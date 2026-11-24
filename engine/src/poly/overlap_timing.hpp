// Opt-in CPU/GPU orchestration timing; no additional CUDA synchronization.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>

namespace strata::poly {

using OverlapClock = std::chrono::steady_clock;
inline double interval_ms(OverlapClock::time_point from, OverlapClock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

// Wall-clock intersection of two activities, NOT GPU kernel occupancy.
inline double concurrent_ms(OverlapClock::time_point a0, OverlapClock::time_point a1,
                            OverlapClock::time_point b0, OverlapClock::time_point b1) {
    return std::max(0.0, interval_ms(std::max(a0, b0), std::min(a1, b1)));
}

struct OverlapTiming {
    int64_t small_calls = 0, small_submits = 0, prompt_calls = 0;
    double small_cpu_ms = 0, small_submit_ms = 0, small_concurrent_ms = 0, small_join_ms = 0;
    double prompt_cpu_ms = 0, prompt_gpu_path_ms = 0, prompt_concurrent_ms = 0, prompt_join_ms = 0;

    void add(const OverlapTiming& v) {
        small_calls += v.small_calls;
        small_submits += v.small_submits;
        prompt_calls += v.prompt_calls;
        small_cpu_ms += v.small_cpu_ms;
        small_submit_ms += v.small_submit_ms;
        small_concurrent_ms += v.small_concurrent_ms;
        small_join_ms += v.small_join_ms;
        prompt_cpu_ms += v.prompt_cpu_ms;
        prompt_gpu_path_ms += v.prompt_gpu_path_ms;
        prompt_concurrent_ms += v.prompt_concurrent_ms;
        prompt_join_ms += v.prompt_join_ms;
    }

    OverlapTiming minus(const OverlapTiming& b) const {
        OverlapTiming d;
        d.small_calls = small_calls - b.small_calls;
        d.small_submits = small_submits - b.small_submits;
        d.prompt_calls = prompt_calls - b.prompt_calls;
        d.small_cpu_ms = small_cpu_ms - b.small_cpu_ms;
        d.small_submit_ms = small_submit_ms - b.small_submit_ms;
        d.small_concurrent_ms = small_concurrent_ms - b.small_concurrent_ms;
        d.small_join_ms = small_join_ms - b.small_join_ms;
        d.prompt_cpu_ms = prompt_cpu_ms - b.prompt_cpu_ms;
        d.prompt_gpu_path_ms = prompt_gpu_path_ms - b.prompt_gpu_path_ms;
        d.prompt_concurrent_ms = prompt_concurrent_ms - b.prompt_concurrent_ms;
        d.prompt_join_ms = prompt_join_ms - b.prompt_join_ms;
        return d;
    }
};

}  // namespace strata::poly
