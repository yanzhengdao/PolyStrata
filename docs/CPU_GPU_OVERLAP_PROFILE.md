# CPU/GPU orchestration timing experiment

Run with `--profile-overlap` on **strata-poly** to log optional timing summaries to
stderr after each normal server request (GEN) or single `--tokens` invocation.

This feature makes no CPU-affinity, scheduling, quantization, stream-sync, or
kernel changes; it adds host-side clock reads only when explicitly requested.
It does **not** add a CUDA synchronization.

## What the measurements mean

- **Small batches**: host CPU expert calculation, time spent **submitting**
  GPU work to the CUDA stream from an auxiliary thread, overlap between those
  two host activities, and waiting for that submission thread to finish.
  **GPU enqueue time is not GPU kernel execution time**.
- **Prompt mixed calls**: CPU expert computation while a separate thread
  launches GPU work and waits for the existing staging stream completion.
  `GPU/staging completion path` includes CPU-side launches and its existing
  `cudaStreamSynchronize` wall time. `concurrent wall` is the intersection
  of that thread's wall interval and CPU expert computation, **not measured
  SM occupancy**.
- **Router wait**: existing CUDA-event synchronization time, reported for
  the request alongside the new statistics.

Totals aggregate across layers and may exceed end-to-end wall time because
different CPU/GPU work overlaps. Small-batch and prompt categories are
separate; absence of either category is normal for some inference paths.
The log is a **first-stage host orchestration diagnostic**, not a complete
Nsight GPU kernel or PCIe bandwidth profiler. CUDA graph execution and
asynchronous transfers are not individually instrumented.

With your baseline configuration unchanged, compare the same fixed requests
first without, then with `--profile-overlap` to estimate diagnostic overhead.
For performance measurements keep the flag **off**. For bottleneck diagnosis,
capture the `overlap-profile:` lines plus normal decode tok/s and run with
0 VMs and 1 VM. If timings indicate GPU execution or PCIe stalls, use
Nsight Systems or CUDA events as a separate second stage rather than adding
unconditional synchronization to the hot path.

A host-only unit test, `poly_overlap_timing_test`, checks interval intersection
and counter aggregation. It does not validate CUDA execution or throughput.


## Phase and layer breakdown (second profiling revision)

The same `--profile-overlap` flag also enables two additional summaries
for a normal single request:

- `overlap-phase: prefill` and `overlap-phase: decode` report the phase wall
  time, router wait in milliseconds and its call count, and the original CPU
  and enqueue timing fields separately. The phases are split at the actual
  Session::generate boundary, not by MoE batch size. Prefill includes the
  final prompt token, and reports newly read rather than reused tokens.
- `overlap-router-layers:` lists all observed layers as
  `zero_based_layer=total_ms/call_count`. The totals use the same existing
  host-side event waits as the aggregate router wait. They can include
  upstream stream work, transfer dependencies, and scheduling latency;
  they are **not** exclusive router kernel timings. Speculative work,
  including rejected rows, contributes to the totals.

Interleaved batch-slot requests retain the original aggregate log and do
not claim an exact phase attribution. Calls using paths with no host router
wait can have no layer entries. Layer totals aggregate across devices if
layer splitting is used, and are not necessarily disjoint wall intervals.
No new GPU events or synchronizations are inserted. Profiling off leaves
the router layer vectors empty and does not take phase snapshots.


## CPU expert detail (third profiling revision)

Add `--profile-cpu-experts` to enable the additional measurements (it implies
`--profile-overlap`). The original overlap flag alone does not enable them.
Normal single-request logs contain `cpu-expert-profile: prefill/decode` and
`cpu-worker-profile: prefill/decode` when the phase actually ran CPU experts.

- Stage wall times: scratch/setup, input activation quantization, grouping
  and task setup, fused Gate/Up plus nonlinear activation and hidden
  quantization, then Down plus weighted reduction. Their sum is the reported
  CPU expert total (before aggregating diagnostic samples).
- Worker batch caller times: mutex acquisition, batch publication and
  notification, caller task drain, waiting for outstanding tasks, retiring
  worker references, and serial tasks. These are nested within stage times.
  Completion wait can contain other workers' useful computation; it is not
  a measurement of pure thread scheduling or barrier overhead.
- Hidden quantization and explicit late-expert wait/copy callback elapsed
  times are thread sums from independent per-task/run samples, not wall
  intervals. They are nested in Gate/Up and must not be added to the total.
  Mapped-file faults during dot products remain inside the stage times.
- Expert uses, unique runs and processed rows help normalize changing
  workloads. Doorbell CPU calls are included, unlike the first overlap
  revision's small-call timing. Multi-device aggregation sums per-device
  measurements and is not necessarily a wall-clock decomposition.

The additional clock reads and sample buffers are opt-in. No SIMD kernel,
task assignment, weight residency policy or synchronization is changed.
Compare throughput with the flag disabled before drawing conclusions.


## CPU expert task granularity (experimental)

Two opt-in CLI parameters, independent of the profiling flags:

- `--cpu-gate-max-rows N`: dynamic Gate/Up task row cap (default 32).
  The actual rows/task remain aligned to four, between 4 and the cap,
  targeting roughly four tasks per worker: floor(unique_runs * n_ff /
  (4 * threads)), rounded down to four and clamped to that range.
- `--cpu-down-rows N`: Down task row count (default 16).

Both accept multiples of four from 4 through 256; invalid input is rejected.
Four-row alignment and reduction order remain unchanged. These options are
stored in the CPU expert worker pool; staging workers keep their defaults.
They apply to small, mixed-prompt and doorbell CPU expert calls.

With `--profile-cpu-experts`, `cpu-task-profile:` reports gate tasks,
gate weight rows (one set per unique expert run, not per token use),
down tasks and down output rows, plus each stage's mean actual rows/task.
A larger gate cap can be ineffective when the dynamic formula is smaller;
compare the actual rows/task before interpreting throughput.

Suggested first tests at 12 threads and otherwise identical settings:
32/16 (baseline), 32/32 (Down only), 16/16 (Gate/Up only). If these show
a meaningful gain or loss, extend with 16/8 and 64/32 or 64/64. Keep the
same prompt, cache state and workload; compare CPU ms per expert use,
per unique run, worker completion wait and throughput. Keep the VM load
consistent. `poly_workers_test` verifies the default sizing formula,
validation, task counts and bitwise Q5_K CPU output parity across seven
shapes with repeated uses. No new kernels or synchronization changes.


## Experimental Q5_K Down reuse (kernel phase 1)

`--cpu-down-reuse` is off by default. It selects a new AVX2 Q5_K/Q8_K
multi-activation dot only for repeated uses of the same CPU Down expert.
Gate/Up, all other weight types, the single-use path, and the original
expert reduction order are unchanged. The kernel decodes each weight
block and scale vector once per group of 2–4 activations; it retains the
existing AVX2 dot's accumulation order. Groups of up to eight uses are
handled in the existing task, with singleton tails using a one-row dot.
It is intentionally NOT installed in the default DotRows dispatch:
the upstream multi-activation registry has IQ3_XXS, IQ2_XXS, IQ2_S and
Q3_K but no Q5_K/Q4_K. This experiment targets Q5_K only.

`cpu-down-reuse-profile:` reports calls where the new kernel is available
with more than one input row, runs with repeated uses, and the uses
actually covered by multi-activation groups (singleton remainders excluded).
The denominator includes all CPU expert uses, not just Q5_K Down.
It is not a per-kernel invocation counter. No additional GPU synchronization.

Build `poly_cpu_expert_bench` in the same Release build as strata-poly.
Run `poly_cpu_expert_bench --check` for bitwise dot and full-expert parity
(AVX2; reports skip if unsupported). The actual dot tests cover 1–9
activations and 256–4096 columns. Run `poly_cpu_expert_bench 12 8 16 100`
for three paired A/B rounds with fixed seed, 2048/512 dimensions,
eight rotating banks, one/two/four/eight/nine shared activations and a
mixed four-token routing pattern. This isolates the CPU kernel from
MTP acceptance, GPU, and context. Both modes include identical profiling;
no two worker pools run simultaneously. It prints total and Down wall
us/use, throughput deltas and reuse coverage. The synthetic weight
distribution and reuse rate are not a model end-to-end speed prediction.
The existing poly_workers_test also checks Q5_K output parity and counters
with the experiment on and off. Keep the flag off in normal usage until
bitwise parity and paired benchmark results are verified on the target CPU.


## Decode stability / quantization diagnosis

`--profile-decode-window N` (0/off by default; valid 0..65536) enables
CPU and overlap profiling and emits a compact `decode-window:` line
after a completed generation round once at least N additional tokens
have been emitted (and a final partial interval if present).
Use N=256. The interval can exceed N by a speculative window; the first
and final emitted token need not be fed back into the model. No extra
CUDA synchronization, no policy or sampling changes. Only non-interleaved
Session::generate requests are attributed. The request counter is local
to this engine session and restarts on engine restart.

* `start_unix_ms/end_unix_ms`: system-clock milliseconds since Unix
  epoch; use these to join external nvidia-smi data in local time.
  Throughput uses steady-clock time, including sampling and host work.
  Logging and snapshot bookkeeping can contribute to adjacent intervals.
* `model_drafts accepted/offered` (MTP for Qwen), `lookup_drafts accepted/offered`:
  only drafts selected for verification, split by actual selected source.
  `model_calls` counts calls to engine.draft including unused proposals;
  `model_draft_ms` is summed host-observed time in those calls.
  `lookup_candidates` counts candidate length after the window cap, not
  verified drafts. `model_unused_rounds` counts model-draft calls made
  in rounds that instead use lookup or a single-token eval.
* `verify_rows/kept_rows/discarded_rows` count speculative window rows,
  including the known anchor; discarded includes a terminal accepted row
  not kept. `mismatch_rounds` counts observed mismatch, not truncation
  at EOS/cancel/length. The final output-only token may have no eval.
* Router, CPU and worker timings are deltas of existing counters.
  GPU/CPU uses, batch cached/staged, swap and ahead counters are existing
  Engine::Stats deltas (including speculative work), not a universal
  cache hit rate. Resident counts are snapshots at interval boundaries.
* `cpu-quant-profile:` appears per prefill/decode phase with observed
  gate/up/down weight and activation format combinations, selected base
  dot and eligible x4/multi-activation kernels, calls, uses and combined
  Gate/Up and Down wall times. Gate and Up timing cannot be separated
  from each other or activation. Eligible x4/rows does not assert they
  executed on every row. The Down experimental flag reflects availability
  for that call, not actual grouped uses; see cpu-down-reuse-profile.
* `cpu-isa-profile:` prints effective capability flags and arithmetic
  contract at startup; a supported ISA is not necessarily selected.

Do not enable `--cpu-down-reuse` for baseline stability tests.
Keep external GPU telemetry separate. Full CPU profiling (including
quant mix aggregation) is diagnostic and may add small host overhead.
