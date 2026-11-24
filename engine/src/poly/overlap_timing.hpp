// Opt-in CPU/GPU orchestration timing; no additional CUDA synchronization.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

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

// Cumulative host wait at existing router events, indexed by the model's layer id.
struct RouterLayerWait {
    int64_t calls = 0;
    double ms = 0;
};
struct RouterTiming {
    std::vector<RouterLayerWait> layers;

    void record(int layer, double ms) {
        if (layer < 0) return;
        const auto i = static_cast<std::size_t>(layer);
        if (layers.size() <= i) layers.resize(i + 1);
        ++layers[i].calls;
        layers[i].ms += ms;
    }
    void add(const RouterTiming& v) {
        if (layers.size() < v.layers.size()) layers.resize(v.layers.size());
        for (std::size_t i = 0; i < v.layers.size(); ++i) {
            layers[i].calls += v.layers[i].calls;
            layers[i].ms += v.layers[i].ms;
        }
    }
    RouterTiming minus(const RouterTiming& b) const {
        RouterTiming d = *this;
        for (std::size_t i = 0; i < std::min(d.layers.size(), b.layers.size()); ++i) {
            d.layers[i].calls -= b.layers[i].calls;
            d.layers[i].ms -= b.layers[i].ms;
        }
        return d;
    }
    RouterLayerWait total() const {
        RouterLayerWait r;
        for (const auto& l : layers) {
            r.calls += l.calls;
            r.ms += l.ms;
        }
        return r;
    }
};

// Caller-side wall times at the existing worker batch boundaries.
// completion_ms can contain useful work on other workers, not just scheduling overhead.
struct WorkerTiming {
    int64_t batches = 0, serial_calls = 0, tasks = 0;
    double lock_ms = 0, publish_ms = 0, caller_drain_ms = 0, completion_ms = 0, retire_ms = 0, serial_ms = 0;

    void add(const WorkerTiming& v) {
        batches += v.batches;
        serial_calls += v.serial_calls;
        tasks += v.tasks;
        lock_ms += v.lock_ms;
        publish_ms += v.publish_ms;
        caller_drain_ms += v.caller_drain_ms;
        completion_ms += v.completion_ms;
        retire_ms += v.retire_ms;
        serial_ms += v.serial_ms;
    }
    WorkerTiming minus(const WorkerTiming& b) const {
        WorkerTiming d = *this;
        d.batches -= b.batches;
        d.serial_calls -= b.serial_calls;
        d.tasks -= b.tasks;
        d.lock_ms -= b.lock_ms;
        d.publish_ms -= b.publish_ms;
        d.caller_drain_ms -= b.caller_drain_ms;
        d.completion_ms -= b.completion_ms;
        d.retire_ms -= b.retire_ms;
        d.serial_ms -= b.serial_ms;
        return d;
    }
};

// The CPU expert stage wall times are disjoint within a call. Thread times
// for hidden quantization and explicit late callbacks are nested in Gate/Up,
// may overlap one another, and must NOT be added to the wall times.
struct CpuExpertTiming {
    int64_t calls = 0, rows = 0, expert_uses = 0, unique_experts = 0, late_wait_calls = 0, hidden_quant_calls = 0, bell_calls = 0;
    double total_ms = 0, setup_ms = 0, input_quant_ms = 0, group_ms = 0, gate_up_activation_ms = 0, down_reduce_ms = 0, hidden_quant_thread_ms = 0, late_wait_thread_ms = 0;
    // Weight rows counted once per unique Gate/Up run; Down output rows once per call.
    int64_t gate_tasks = 0, gate_weight_rows = 0, down_tasks = 0, down_output_rows = 0;
    int64_t down_reuse_calls = 0, down_reuse_runs = 0, down_reuse_uses = 0;
    WorkerTiming workers;

    void add(const CpuExpertTiming& v) {
        calls += v.calls;
        rows += v.rows;
        expert_uses += v.expert_uses;
        unique_experts += v.unique_experts;
        late_wait_calls += v.late_wait_calls;
        hidden_quant_calls += v.hidden_quant_calls;
        bell_calls += v.bell_calls;
        total_ms += v.total_ms;
        setup_ms += v.setup_ms;
        input_quant_ms += v.input_quant_ms;
        group_ms += v.group_ms;
        gate_up_activation_ms += v.gate_up_activation_ms;
        down_reduce_ms += v.down_reduce_ms;
        hidden_quant_thread_ms += v.hidden_quant_thread_ms;
        late_wait_thread_ms += v.late_wait_thread_ms;
        gate_tasks += v.gate_tasks;
        gate_weight_rows += v.gate_weight_rows;
        down_tasks += v.down_tasks;
        down_output_rows += v.down_output_rows;
        down_reuse_calls += v.down_reuse_calls;
        down_reuse_runs += v.down_reuse_runs;
        down_reuse_uses += v.down_reuse_uses;
        workers.add(v.workers);
    }
    CpuExpertTiming minus(const CpuExpertTiming& b) const {
        CpuExpertTiming d = *this;
        d.calls -= b.calls;
        d.rows -= b.rows;
        d.expert_uses -= b.expert_uses;
        d.unique_experts -= b.unique_experts;
        d.late_wait_calls -= b.late_wait_calls;
        d.hidden_quant_calls -= b.hidden_quant_calls;
        d.bell_calls -= b.bell_calls;
        d.total_ms -= b.total_ms;
        d.setup_ms -= b.setup_ms;
        d.input_quant_ms -= b.input_quant_ms;
        d.group_ms -= b.group_ms;
        d.gate_up_activation_ms -= b.gate_up_activation_ms;
        d.down_reduce_ms -= b.down_reduce_ms;
        d.hidden_quant_thread_ms -= b.hidden_quant_thread_ms;
        d.late_wait_thread_ms -= b.late_wait_thread_ms;
        d.gate_tasks -= b.gate_tasks;
        d.gate_weight_rows -= b.gate_weight_rows;
        d.down_tasks -= b.down_tasks;
        d.down_output_rows -= b.down_output_rows;
        d.down_reuse_calls -= b.down_reuse_calls;
        d.down_reuse_runs -= b.down_reuse_runs;
        d.down_reuse_uses -= b.down_reuse_uses;
        d.workers = workers.minus(b.workers);
        return d;
    }
};

struct OverlapTiming {
    int64_t small_calls = 0, small_submits = 0, prompt_calls = 0;
    double small_cpu_ms = 0, small_submit_ms = 0, small_concurrent_ms = 0, small_join_ms = 0;
    double prompt_cpu_ms = 0, prompt_gpu_path_ms = 0, prompt_concurrent_ms = 0, prompt_join_ms = 0;
    RouterTiming router;
    CpuExpertTiming cpu_detail;

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
        router.add(v.router);
        cpu_detail.add(v.cpu_detail);
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
        d.router = router.minus(b.router);
        d.cpu_detail = cpu_detail.minus(b.cpu_detail);
        return d;
    }
};

}  // namespace strata::poly
