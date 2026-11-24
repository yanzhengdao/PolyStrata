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


## Experimental Q5_K / Q6_K single-activation AVX-VNNI

The 2026-11-24 Tiel UD-Q5_K_XL CPU profile showed Q5_K gate/up and Q6_K
down as the main hot formats, with AVX2 dots despite effective AVX-VNNI
support. The new opt-in `--cpu-q5-vnni` and `--cpu-q6-vnni` flags select
experimental single-activation kernels only for the corresponding CPU
expert weight type and only when its baseline dot uses AVX2, the
AVX-VNNI tier is built, and the effective CPU caps allow it. They may be
enabled independently or together. They do not modify default dot
dispatch, other formats, activation quantization, routing or GPU kernels.
The phase `cpu-quant-profile` records actual selected dot names.

The new kernels retain the optimized AVX2 unpack, scale/minimum
correction, int32 lane structure, FP32 accumulation order and reduction,
but replace VPMADDWD + VPADDD by VEX VPDPWSSD (non-saturating).
VPMADDUBSW remains: unsigned Q5/Q6 values fit int16 pair products
even at the signed Q8 endpoints. Q6 integer additions are regrouped
without changing sums. This is an experiment in instruction reduction,
not a guarantee of faster execution on a particular CPU.

Run `poly_cpu_expert_bench --vnni-check` to test bitwise parity of the
two dots against the existing AVX2 baseline, including valid extreme
Q8_K bytes with consistent block sums, and the full Q5_K/Q5_K/Q6_K
expert output and actual dispatch of all three flag combinations.
The test is also registered as `poly_cpu_k_vnni_test`.
Run `poly_cpu_expert_bench --vnni 12 8 16 2000` (or 5000) to measure
three paired, alternating-order rounds for each candidate. Cases:
single, sparse-4 (two of sixteen uses share one expert), mixed-4 (four
of sixteen uses share one expert), shared-2, shared-8. Eight rotating
banks, 2048/512 dimensions, identical input/weights per pair. Per-mode
baseline is measured separately. No GPU or model required.
The previous Q5_K Down reuse benchmark and test remain unchanged in
purpose. Do not combine the Down reuse flag with the VNNI experiment
for initial performance tests. Keep both new flags off for normal
inference until target-machine parity and performance pass.


## Experimental Q5_K / Q6_K four-weight-row AVX2

The 5000-call VNNI benchmark on the 285HX showed small Q5 improvements
and consistent Q6 Down regressions. The next experiment uses
`--cpu-q5-x4` and `--cpu-q6-x4` independently. Both default off.
Four weight rows share each Q8_K activation block load and its sums.
The block-major AVX2 kernels keep four independent FP32 accumulators,
the original integer lane arithmetic, Q5 scalar minimum correction and
per-row block/reduction order. No AVX-VNNI instructions are required.
Register pressure/spills and weight access patterns may offset reuse
benefits; the target-machine benchmark is the gate, not instruction count.

Only CPU expert dispatch is changed, and only when the baseline uses
AVX2. The existing Gate/Up and Down four-row paths are reused; incomplete
row groups still use the original single-dot kernel. If the corresponding
VNNI flag selects a VNNI dot, that selection takes precedence over x4.
Do not combine these flags or Down reuse for the first A/B test.
`cpu-quant-profile` shows the selected x4 kernel name, separately from
the scalar-dot name. Eligibility is not a count of actual x4 invocations.

`poly_cpu_expert_bench --x4-check` checks both formats against the
baseline, seven lengths, 256 trials each, packed/padded row strides,
extreme valid Q8_K bytes and full mixed-quant expert output/dispatch.
`poly_cpu_k_x4_test` registers the same check with CTest.
`poly_cpu_expert_bench --x4 12 8 16 2000` or 5000 runs the same
rotating-bank mixed-quant benchmark as --vnni, comparing baseline
against Q5-only, Q6-only and both. Keep the existing --vnni and --check
modes. These are CPU stage measurements, not end-to-end inference.


## GPU FDEC q8 experiments (Blackwell / SM120 target)

After the unsuccessful K-quant x4 CPU experiment, isolate the CUDA
attention path. Upstream main was still 36e654b1 (already merged) at
this review. The existing FDEC kernel uses 4 warps, 64 keys per tile,
256 or 1024 keys per split depending on scratch, and MT=1/2 (16/32
query rows). The current q8 kernel loads all V fragments before QK;
the merge computes identical softmax split coefficients for each output
dimension. The opt-in environment variable STRATA_FDEC_EXPERIMENT:
0 baseline (default), 1 late V fragment load (16 keys at a time during
PV), 2 shared split coefficients during merge, 3 both. Only q8 FDEC
is changed, not F16, prefill, split selection, quantization or draft
policy. The shared merge is bounded to 4096 splits (32 KiB dynamic
shared); otherwise it falls back. No production CUDA events or host
synchronization. Potential tradeoff: late V reduces live registers but
may lose memory latency hiding; shared merge adds CTA barriers. Neither
is assumed faster until measured. Runtime mode is read once per process.

Build the existing CUDA target `poly_fattn_test`; run
`poly_fattn_test dec-ab --repeats 20` (GQA8, heads16/KV2), or add
`gqa16` for heads32/KV2. It prints device attributes and static kernel
resource usage (registers, local bytes, shared bytes, theoretical
active blocks/warps per SM, not measured occupancy); then pairs each
candidate against the baseline in alternating order with CUDA events.
The reported time includes FDEC attention and merge launches together,
not separate kernel durations or end-to-end token speed. Cases cover
4K-262K positions, n=1/2/4/8, device position/room, whole mask, a
strided cache and reduced scratch. It checks bitwise output and softmax
max/sum against baseline for each mode; the existing `dec` numerical
reference test remains available (run with the chosen environment mode).
Device-position launches exercise recorded-launch sizing, but this
benchmark does not capture/replay CUDA graphs. Keep all CPU experiment
flags off. First test only the standalone benchmark; then enable the
winning mode in the engine's process environment, if any. No defaults
are changed without target-machine parity and performance evidence.
