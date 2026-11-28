// src/poly/bell.cuh - the card's side of the doorbell (ExpertBell, kernels.hpp): its loads and stores of the words
// the host writes and reads while a kernel runs, and its wait between two looks.  CUDA keeps its volatile accesses
// and __threadfence_system(), as before.  On HIP the words are read and written with system-scope atomics (a
// kernel on an RX 9070 XT under WSL2 saw the host's writes to pinned memory live this way, and the host its writes:
// a round trip of 1.3 to 2.8 us), and the bell's memory is coherent (Engine::hmalloc(): hipHostMallocCoherent), so
// no cache between the card and the host holds a stale copy.
#pragma once

#include "strata/poly/kernels.hpp"

#include <cstdint>

namespace strata::poly {

/// The word *p as the host last wrote it.
__device__ __forceinline__ uint32_t bell_peek(const volatile uint32_t* p) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __hip_atomic_load(const_cast<uint32_t*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *p;
#endif
}

/// *p = v for the host to see; what this thread wrote before (and fenced at system scope) is the host's first.
__device__ __forceinline__ void bell_poke(volatile uint32_t* p, uint32_t v) {
#if defined(__HIP_DEVICE_COMPILE__)
    __hip_atomic_store(const_cast<uint32_t*>(p), v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    *p = v;
#endif
}

/// A pause between two looks at the bell, so that a waiting block does not keep the memory path busy (a spinning
/// card draws power that a laptop's CPU, computing the experts meanwhile, would rather have): 0.5 us on CUDA; on AMD
/// s_sleep 8, 8 x 64 clocks (~0.2 us at 2.5 GHz).
__device__ __forceinline__ void bell_pause() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_s_sleep(8);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(500);
#endif
}

/// One thread: wait until the bell's ack is ring r or the host gave up (abort).  The caller's block then meets at a
/// barrier and fences at system scope before it reads what the host wrote.
__device__ __forceinline__ void bell_wait(const ExpertBell* bell, uint32_t r) {
    while (bell_peek(&bell->ack) != r && !bell_peek(&bell->abort)) bell_pause();
#if defined(__HIP_DEVICE_COMPILE__)
    // the acquire that pairs with the host's release before its ack: the host's sum is this thread's to share
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
#endif
}

}  // namespace strata::poly
