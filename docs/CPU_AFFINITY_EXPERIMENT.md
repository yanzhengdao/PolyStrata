# CPU expert P-core affinity experiment (Windows)

This is an opt-in experiment for strata-poly on Windows 10/11 hybrid CPUs. It does
not change the default behavior.

- **--cpu-affinity default** (the default): leave Windows scheduling unchanged.
- **--cpu-affinity p-cores**: choose the available CPU Sets in the highest
  EfficiencyClass reported by Windows. On a hybrid CPU this normally identifies
  its performance cores. If Windows cannot distinguish CPU classes, the engine
  exits with an error rather than silently running a misleading benchmark.

The engine assigns these CPU Sets to its CPU expert Workers threads. The
request/calling thread also computes CPU expert tasks; it temporarily selects
the same CPU Sets **during cpu_experts() only**, restoring its previous
selection immediately afterward. Expert prefetch threads, staging threads,
the asynchronous GPU-launch helper and the rest of the process are not pinned.
Windows may still schedule freely *within* the selected CPU Sets.

This setting does not reduce the number of threads. Start with your known good
--threads setting and compare the same model, prompt and output length
against default before changing any other option. Test with no VM, one VM
and two VMs. Record prefill and decode speed, first-token latency and VM
responsiveness. The extra CPU Set calls on the request thread may outweigh
any scheduling improvement; speedup is **not** assumed.

The implementation uses Windows
[GetSystemCpuSetInformation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemcpusetinformation),
[SetThreadSelectedCpuSets](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadselectedcpusets)
and [GetThreadSelectedCpuSets](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getthreadselectedcpusets).
A CPU Set does not override process-level affinity restrictions.
