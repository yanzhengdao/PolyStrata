// Host-only correctness checks for optional overlap diagnostics.
#include "overlap_timing.hpp"
#include <chrono>
#include <cstdio>

using namespace strata::poly;

int main() {
    using namespace std::chrono;
    const auto t = OverlapClock::time_point{};
    const auto at = [&](int n) { return t + milliseconds(n); };
    if (concurrent_ms(at(0), at(10), at(4), at(12)) != 6.0 ||
        concurrent_ms(at(0), at(2), at(4), at(12)) != 0.0 ||
        concurrent_ms(at(5), at(5), at(4), at(12)) != 0.0)
        return std::fprintf(stderr, "overlap_timing_test: bad interval overlap\n"), 1;
    OverlapTiming a;
    a.small_calls = 5; a.small_submits = 3; a.small_cpu_ms = 100; a.small_join_ms = 4;
    a.small_submit_ms = 15; a.small_concurrent_ms = 11;
    a.prompt_calls = 2; a.prompt_cpu_ms = 12; a.prompt_gpu_path_ms = 14;
    a.prompt_concurrent_ms = 10; a.prompt_join_ms = 2;
    OverlapTiming b = a;
    b.add(a);
    OverlapTiming d = b.minus(a);
    if (d.small_calls != a.small_calls || d.small_submits != a.small_submits ||
        d.small_cpu_ms != a.small_cpu_ms || d.small_join_ms != a.small_join_ms ||
        d.small_submit_ms != a.small_submit_ms || d.small_concurrent_ms != a.small_concurrent_ms ||
        d.prompt_calls != a.prompt_calls || d.prompt_cpu_ms != a.prompt_cpu_ms ||
        d.prompt_gpu_path_ms != a.prompt_gpu_path_ms ||
        d.prompt_concurrent_ms != a.prompt_concurrent_ms || d.prompt_join_ms != a.prompt_join_ms)
        return std::fprintf(stderr, "overlap_timing_test: bad aggregate/delta\n"), 1;
    std::puts("overlap_timing_test: passed");
    return 0;
}
