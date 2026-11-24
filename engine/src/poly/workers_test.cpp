// src/poly/workers_test.cpp - Workers: every task of a batch runs once, and a task that shares out work of its
// own comes back.  The rest of cpu_experts.cpp needs ggml, so CMake builds this: poly_workers_test.
#include "cpu_experts.hpp"
#include "strata/quant/types.hpp"
#include "strata/quant/dot.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <thread>

using strata::poly::Workers;

namespace {
// Exercise the actual quantized CPU expert path with repeated expert uses,
// not just the task-size formula. Q5_K uses 256-value blocks, with two
// half scales at the front. Use finite small scales and fixed random bytes.
bool task_shape_test() {
    using namespace strata::poly;
    namespace Q = strata::quant;
    const CpuExpertTaskShape baseline;
    if (!baseline.valid() || baseline.gate_max_rows != 32 || baseline.down_rows != 16 ||
        CpuExpertTaskShape{0,16}.valid() || CpuExpertTaskShape{32,6}.valid() ||
        CpuExpertTaskShape{260,16}.valid()) return false;
    for (int threads : {1, 8, 12, 15, 32})
        for (int runs : {0, 1, 2, 5, 20}) {
            int old = runs * 512 / (4 * threads);
            old = std::clamp(old - old % 4, 4, 32);
            if (baseline.gate_rows(runs, 512, threads) != old) return false;
        }
    constexpr int ne = 256, nf = 512, nr = 3;
    const int type = (int) Q::Type::Q5_K;
    std::string err;
    if (!cpu_expert_types_ok(type, type, ne, nf, err)) {
        std::fprintf(stderr, "FAIL: Q5_K task shape test: %s\n", err.c_str());
        return false;
    }
    const auto* ti = Q::type_info(type);
    std::mt19937 rng(0x51a7u);
    auto weights = [&](int cols, int rows) {
        std::vector<uint8_t> w(Q::row_bytes(type, cols) * (size_t) rows);
        for (auto& b : w) b = (uint8_t) rng();
        const uint16_t d = 0x1800, dm = 0x1400;
        for (size_t i = 0; i < w.size(); i += ti->bytes) {
            std::memcpy(w.data() + i, &d, 2);
            std::memcpy(w.data() + i + 2, &dm, 2);
        }
        return w;
    };
    auto g0 = weights(ne,nf), u0 = weights(ne,nf), d0 = weights(nf,ne);
    auto g1 = weights(ne,nf), u1 = weights(ne,nf), d1 = weights(nf,ne);
    const CpuExpert ex[] = {
        {g0.data(),u0.data(),d0.data(),0.4f,0},
        {g1.data(),u1.data(),d1.data(),0.6f,0},
        {g1.data(),u1.data(),d1.data(),1.0f,1},
        {g0.data(),u0.data(),d0.data(),0.7f,2},
        {g1.data(),u1.data(),d1.data(),0.3f,2}
    };
    std::vector<float> x(ne*nr), ref(ne*nr), out(ne*nr);
    for (auto& v : x) v = (float)((int)(rng()%201)-100) / 100.f;
    const CpuExpertTaskShape shapes[] = {{32,16},{16,8},{32,32},{64,32},{64,64},{4,4},{256,256},
                                        {8,16,true},{16,16,true}};
    const bool supported = Q::experimental_q5_rows_kernel(Q::default_arith()) != nullptr;
    bool first = true;
    for (const auto shape : shapes) {
        Workers pool(4, 20000, shape);
        CpuExpertTiming timing;
        std::fill(out.begin(), out.end(), 0.f);
        cpu_experts(pool, ex, 5, nr, type, type, type, ne, nf, x.data(), 20.f, false,
                    out.data(), nullptr, nullptr, nullptr, 0.f, 0.f, &timing);
        const int per = shape.gate_rows(2,nf,pool.threads());
        if (timing.calls != 1 || timing.expert_uses != 5 || timing.unique_experts != 2 ||
            timing.gate_tasks != 2*((nf+per-1)/per) || timing.gate_weight_rows != 2*nf ||
            timing.down_tasks != (ne+shape.down_rows-1)/shape.down_rows || timing.down_output_rows != ne ||
            timing.down_reuse_calls != (shape.down_reuse && supported ? 1 : 0) ||
            timing.down_reuse_runs != (shape.down_reuse && supported ? 2 : 0) ||
            timing.down_reuse_uses != (shape.down_reuse && supported ? 5 : 0))
            return false;
        if (first) {
            bool nonzero = false;
            for (const float v : out) {
                if (!std::isfinite(v)) return false;
                nonzero |= v != 0.f;
            }
            if (!nonzero) return false;
            ref = out;
            first = false;
        } else if (std::memcmp(out.data(), ref.data(), out.size()*sizeof(float)) != 0) return false;
    }
    return true;
}
} // namespace

int main() {
    // a pool that waits for itself never comes back: the test says so instead of standing still
    std::atomic<bool> done{false};
    std::thread watch([&] {
        for (int i = 0; i < 300 && !done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!done.load()) {
            std::fprintf(stderr, "FAIL: the pool did not come back in 30 s\n");
            std::_Exit(1);
        }
    });
    int checks = 0;
    // Exercise immediate sleeping, short spinning and the original 20k-pause behavior.
    // Every 100 rounds, leave enough idle time to enter the atomic wait path.
    for (const int spins : {0, 1024, 20000}) {
        Workers pool(8, spins);
        strata::poly::WorkerTiming timing;
        std::atomic<int> profiled{0};
        pool.run(1, [&](int) { profiled.fetch_add(1); }, &timing);
        pool.run(32, [&](int) { profiled.fetch_add(1); }, &timing);
        if (profiled.load() != 33 || timing.serial_calls != 1 || timing.batches != 1 ||
            timing.tasks != 33 || timing.lock_ms < 0 || timing.publish_ms < 0 ||
            timing.caller_drain_ms < 0 || timing.completion_ms < 0 || timing.retire_ms < 0 ||
            timing.serial_ms < 0) {
            std::fprintf(stderr, "FAIL: profiled workers (spins %d)\n", spins);
            done.store(true);
            watch.join();
            return 1;
        }
        ++checks;
        for (int round = 0; round < 2000; ++round) {
            if (round % 100 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::atomic<int> outer{0}, inner{0};
            pool.run(16, [&](int) {
                outer.fetch_add(1);
                pool.run(4, [&](int) { inner.fetch_add(1); });
            });
            if (outer.load() != 16 || inner.load() != 64) {
                std::fprintf(stderr, "FAIL: spins %d round %d ran %d of 16 tasks and %d of 64 of theirs\n",
                             spins, round, outer.load(), inner.load());
                done.store(true);
                watch.join();
                return 1;
            }
            checks += 2;
        }
    }
    if (!task_shape_test()) {
        std::fprintf(stderr, "FAIL: CPU expert task shape parity or task counters\n");
        done.store(true);
        watch.join();
        return 1;
    }
    ++checks;
    done.store(true);
    watch.join();
    std::printf("workers_test: %d checks passed (including Q5_K task shape parity)\n", checks);
    return 0;
}
