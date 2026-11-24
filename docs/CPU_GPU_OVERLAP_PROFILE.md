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
