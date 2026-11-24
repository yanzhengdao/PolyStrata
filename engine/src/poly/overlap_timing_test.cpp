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
    // A layer can first appear after the phase boundary, or on a different card.
    OverlapTiming start, after_prompt, after_decode;
    start.router.record(0, 2);
    after_prompt = start;
    after_prompt.router.record(0, 3);
    after_prompt.router.record(2, 7);
    after_decode = after_prompt;
    after_decode.router.record(2, 11);
    after_decode.router.record(4, 13);
    const auto pre = after_prompt.minus(start).router;
    const auto dec = after_decode.minus(after_prompt).router;
    const auto all = after_decode.minus(start).router;
    if (pre.total().calls != 2 || pre.total().ms != 10 ||
        dec.total().calls != 2 || dec.total().ms != 24 ||
        all.total().calls != 4 || all.total().ms != 34 ||
        pre.layers[0].ms != 3 || pre.layers[2].ms != 7 ||
        dec.layers[0].calls != 0 || dec.layers[2].ms != 11 || dec.layers[4].ms != 13)
        return std::fprintf(stderr, "overlap_timing_test: bad router phase delta\n"), 1;
    RouterTiming combined = pre;
    combined.add(dec);
    if (combined.total().ms != all.total().ms || combined.total().calls != all.total().calls ||
        combined.layers[2].calls != 2 || combined.layers[2].ms != 18)
        return std::fprintf(stderr, "overlap_timing_test: bad router layer aggregation\n"), 1;
    OverlapTiming cpu_a, cpu_b;
    cpu_a.cpu_detail.calls = 2;
    cpu_a.cpu_detail.rows = 3;
    cpu_a.cpu_detail.expert_uses = 8;
    cpu_a.cpu_detail.unique_experts = 5;
    cpu_a.cpu_detail.bell_calls = 1;
    cpu_a.cpu_detail.total_ms = 32;
    cpu_a.cpu_detail.gate_up_activation_ms = 20;
    cpu_a.cpu_detail.down_reduce_ms = 10;
    cpu_a.cpu_detail.hidden_quant_thread_ms = 4;
    cpu_a.cpu_detail.hidden_quant_calls = 5;
    cpu_a.cpu_detail.late_wait_calls = 3;
    cpu_a.cpu_detail.late_wait_thread_ms = 7;
    cpu_a.cpu_detail.gate_tasks = 12;
    cpu_a.cpu_detail.gate_weight_rows = 384;
    cpu_a.cpu_detail.down_tasks = 16;
    cpu_a.cpu_detail.down_output_rows = 256;
    cpu_a.cpu_detail.down_reuse_calls = 2;
    cpu_a.cpu_detail.down_reuse_runs = 4;
    cpu_a.cpu_detail.down_reuse_uses = 7;
    cpu_a.cpu_detail.workers.batches = 4;
    cpu_a.cpu_detail.workers.tasks = 160;
    cpu_a.cpu_detail.workers.completion_ms = 6;
    cpu_b.add(cpu_a);
    cpu_b.add(cpu_a);
    const auto cd = cpu_b.minus(cpu_a).cpu_detail;
    if (cd.calls != 2 || cd.rows != 3 || cd.expert_uses != 8 || cd.unique_experts != 5 ||
        cd.bell_calls != 1 || cd.total_ms != 32 || cd.gate_up_activation_ms != 20 ||
        cd.down_reduce_ms != 10 || cd.hidden_quant_thread_ms != 4 || cd.hidden_quant_calls != 5 ||
        cd.late_wait_calls != 3 || cd.late_wait_thread_ms != 7 || cd.workers.batches != 4 ||
        cd.workers.tasks != 160 || cd.workers.completion_ms != 6 ||
        cd.gate_tasks != 12 || cd.gate_weight_rows != 384 || cd.down_tasks != 16 || cd.down_output_rows != 256 ||
        cd.down_reuse_calls != 2 || cd.down_reuse_runs != 4 || cd.down_reuse_uses != 7)
        return std::fprintf(stderr, "overlap_timing_test: bad CPU detail aggregate/delta\n"), 1;
    // A quant mix first seen in decode must survive deltas, and a prefill-only
    // mix must disappear. Gate+Up stage time is joint, never split by guesswork.
    CpuExpertTiming qa, qb;
    CpuQuantMix q1;
    q1.gate.weight = 12; q1.up.weight = 13; q1.down.weight = 14;
    q1.calls = 2; q1.uses = 9; q1.unique_runs = 6; q1.gate_up_ms = 5; q1.down_ms = 3;
    auto q2 = q1;
    q2.down.weight = 13; q2.down_multi = true; q2.calls = 1; q2.uses = 4;
    add_quant_mix(qa.quant_mixes, q1);
    qb = qa;
    add_quant_mix(qb.quant_mixes, q2);
    const auto only = qb.minus(qa);
    if (only.quant_mixes.size()!=1 || !only.quant_mixes[0].same(q2) ||
        only.quant_mixes[0].uses!=4 || only.quant_mixes[0].gate_up_ms!=5)
        return std::fprintf(stderr, "overlap_timing_test: bad quant mix delta\n"), 1;
    qa.add(only);
    qa.add(only);
    if (qa.quant_mixes.size()!=2 || qa.quant_mixes[1].calls!=2 ||
        qa.quant_mixes[1].uses!=8 || qa.quant_mixes[1].down_ms!=6)
        return std::fprintf(stderr, "overlap_timing_test: bad quant mix add\n"), 1;
    DecodeRounds rounds;
    rounds.record(1,false,1,0,true,2,1,0.5);
    rounds.record(4,false,2,1,true,3,1,1.0);
    rounds.record(4,true,4,3,true,0,3,0.0);
    rounds.record(4,true,2,2,false,1,3,0.5);
    if (rounds.single!=1 || rounds.model_rounds!=1 || rounds.lookup_rounds!=2 ||
        rounds.model_calls!=6 || rounds.model_offered!=3 || rounds.model_accepted!=1 ||
        rounds.lookup_offered!=6 || rounds.lookup_accepted!=5 ||
        rounds.verified_rows!=12 || rounds.kept_rows!=8 || rounds.mismatch_rounds!=1 ||
        rounds.model_draft_unused_rounds!=2 || rounds.lookup_candidates!=8 || rounds.model_draft_ms!=2)
        return std::fprintf(stderr, "overlap_timing_test: bad decode round accounting\n"), 1;
    std::puts("overlap_timing_test: passed");
    return 0;
}
