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
