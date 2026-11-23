# CPU expert worker idle-wait experiment (Windows / Linux)

Previous opt-in Windows `--cpu-affinity p-cores` experiment was reverted because
it reduced real decode throughput sharply. Default Windows CPU scheduling is unchanged.

New knob: `--worker-idle-spins N` (0..1,000,000). This tunes only the CPU
expert `Workers` pool. It is the number of `_mm_pause` loop iterations before
idle threads block on `std::atomic::wait` until a new batch arrives.
It does **not** affect GPU helpers, the `Aside` thread, expert stagers, CPU
affinity, task partitioning, or numerical inference output.

The default 20,000 is **exactly the previous behavior**. Suggested A/B values:

- omit the flag, or `--worker-idle-spins 20000`: baseline
- `--worker-idle-spins 1024`: short spin before sleep
- `--worker-idle-spins 0`: sleep as soon as no work is available

Measure warmed-up median decode tok/s, prefill tok/s, CPU usage and VM
responsiveness with the same GGUF, prompt, and all other engine options.
Shorter spinning may reduce CPU contention but costs more wakeups and can
reduce throughput. There is no presumed winner; retain the baseline if
the new settings do not improve real workloads.

`poly_workers_test` runs 2,000 nested-work batches in each of the three
modes and exercises the idle-to-sleep transition. These tests check
correctness/liveness, not throughput.
