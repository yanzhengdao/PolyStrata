// src/poly/bell_test.cu - the doorbell (bell_ring(), bell_add(), ExpertBell) between a running card and this host,
// as Engine::bell_serve() answers it: the picks the card lacks and the token's input reach the host while the card
// goes on, and the card adds the host's answer only once it is there.  Cases: no miss (no ring); a ring answered
// while the card waits, many times over with new numbers each time (a stale read of the answer or of the input
// shows); 48 layers of rings in one recording, answered as they come, alone and between ~1,500 other kernels with
// large arguments in two recordings (a token's size: the host's launches must not wait for room behind a card that
// waits for the host); an answer 3 s late; an abort, which must end the card's wait.
// The memory is the engine's (pinned; on HIP coherent, as Engine::hmalloc() gives the doorbell).
//
// usage: poly_bell_test   (exit 0: every check passed; 77: no card)
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

using namespace strata::poly;
using Clock = std::chrono::steady_clock;

constexpr int kExperts = 64, kUsed = 8, kEmbd = 2304, kLayers = 48;
int failed = 0, checks = 0;
int delay_us = 0;   // serve(): the host waits this long before it answers a ring (a card that does not wait shows)

void expect(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++failed;
        std::printf("FAIL %s\n", what);
    }
}

void ck(cudaError_t e) {
    if (e == cudaSuccess) return;
    ++failed;
    std::printf("FAIL CUDA: %s\n", cudaGetErrorString(e));
}

void* host_live(size_t bytes) {
    void* p = nullptr;
#if defined(STRATA_USE_HIP)
    if (hipHostMalloc(&p, bytes, hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) return nullptr;
#else
    if (cudaMallocHost(&p, bytes) != cudaSuccess) return nullptr;
#endif
    std::memset(p, 0, bytes);
    return p;
}

struct Rig {
    ExpertBell* bell = nullptr;
    float *hx = nullptr, *hy = nullptr;            // pinned: the token's input, the host's answer
    uint32_t* seq = nullptr;                       // the card's
    void** slots = nullptr;                        // per layer: an expert's slot, null outside VRAM
    int32_t* ids = nullptr;                        // per layer: the token's picks
    float *w = nullptr, *x = nullptr, *acc = nullptr;
    std::vector<int32_t> h_ids;                    // kLayers x kUsed
    std::vector<float> h_w;
    std::vector<uint8_t> missing;                  // kLayers x kExperts
    cudaStream_t st = nullptr;
    cudaEvent_t done = nullptr;
    uint32_t seen = 0;
};

// Each layer's picks: distinct experts, about a third of them outside VRAM (skip: none outside).
void make_layers(Rig& g, std::mt19937& rng, bool none_missing) {
    g.h_ids.assign((size_t) kLayers * kUsed, 0);
    g.h_w.assign(g.h_ids.size(), 0.f);
    g.missing.assign((size_t) kLayers * kExperts, 0);
    std::vector<void*> slots((size_t) kLayers * kExperts);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (int il = 0; il < kLayers; ++il) {
        for (int e = 0; e < kExperts; ++e) {
            const bool miss = !none_missing && u(rng) < 0.33f;
            g.missing[(size_t) il * kExperts + e] = miss;
            slots[(size_t) il * kExperts + e] = miss ? nullptr : (void*) (uintptr_t) (0x1000 + e);   // never read
        }
        std::vector<int> pool(kExperts);
        for (int e = 0; e < kExperts; ++e) pool[e] = e;
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int k = 0; k < kUsed; ++k) {
            g.h_ids[(size_t) il * kUsed + k] = pool[k];
            g.h_w[(size_t) il * kUsed + k] = u(rng);
        }
    }
    ck(cudaMemcpy(g.slots, slots.data(), slots.size() * sizeof(void*), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(g.ids, g.h_ids.data(), g.h_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(g.w, g.h_w.data(), g.h_w.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void launch_layer(Rig& g, int il) {
    bell_ring(g.slots + (size_t) il * kExperts, g.ids + (size_t) il * kUsed, g.w + (size_t) il * kUsed, kUsed, kUsed,
              il, g.x + (size_t) il * kEmbd, g.hx, kEmbd, g.bell, g.seq, g.st);
    bell_add(g.bell, g.seq, g.hy, g.acc, kEmbd, g.st);
}

// A stand-in for a layer's other kernels: an argument as large as the engine's larger ones, one write.
struct Big {
    float* out;
    int pad[60];
};
__global__ void filler_kernel(Big b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) b.out[0] = (float) b.pad[0];
}

// The host's answer to layer il's ring: its jobs checked against the picks, its input against x, and an answer
// that depends on the layer and the round (so a stale one shows).
float answer(int il, int round, int i) { return (float) ((il * 7 + round * 13 + i) % 97) * 0.25f - 3.f; }

// Engine::bell_serve()'s loop: answer each ring until the card is past `done`.  abort_after: rings answered before
// the host gives up instead (abort), -1 never.  False: the card did not finish within 10 s.
bool serve(Rig& g, const std::vector<float>& h_x, int round, int abort_after, int* answered, bool* jobs_ok) {
    ck(cudaEventRecord(g.done, g.st));
    (void) cudaStreamQuery(g.st);   // as bell_serve(): under WSL2 a launch may wait for the host's next HIP call
    const Clock::time_point t0 = Clock::now();
    *answered = 0;
    for (;;) {
        const uint32_t r = g.bell->ring;
        if (r != g.seen) {
            std::atomic_thread_fence(std::memory_order_acquire);
            g.seen = r;
            if (abort_after >= 0 && *answered >= abort_after) {
                g.bell->abort = 1;
                ck(cudaStreamSynchronize(g.st));
                g.bell->abort = 0;
                g.seen = g.bell->ring;
                return true;
            }
            const int il = g.bell->layer, n = g.bell->n;
            // the jobs: the picks outside VRAM, in pick order
            std::vector<int> want;
            for (int k = 0; k < kUsed; ++k) {
                const int e = g.h_ids[(size_t) il * kUsed + k];
                if (g.missing[(size_t) il * kExperts + e]) want.push_back(k);
            }
            bool ok = n == (int) want.size();
            for (int j = 0; ok && j < n; ++j)
                ok = g.bell->e[j] == g.h_ids[(size_t) il * kUsed + want[j]] &&
                     g.bell->w[j] == g.h_w[(size_t) il * kUsed + want[j]];
            for (int i = 0; ok && i < kEmbd; ++i) ok = g.hx[i] == h_x[(size_t) il * kEmbd + i];
            *jobs_ok = *jobs_ok && ok;
            if (delay_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
            for (int i = 0; i < kEmbd; ++i) g.hy[i] = answer(il, round, i);
            std::atomic_thread_fence(std::memory_order_release);
            g.bell->ack = r;
            ++*answered;
            continue;
        }
        const cudaError_t q = cudaEventQuery(g.done);
        if (q == cudaSuccess) return true;
        if (q != cudaErrorNotReady) {
            std::printf("CUDA: %s\n", cudaGetErrorString(q));
            return false;
        }
        if (Clock::now() - t0 > std::chrono::seconds(10)) {
            g.bell->abort = 1;   // end the card's wait so that the test can go on
            ck(cudaStreamSynchronize(g.st));
            g.bell->abort = 0;
            g.seen = g.bell->ring;
            return false;
        }
    }
}

bool rings_at(const Rig& g, int il) {
    for (int k = 0; k < kUsed; ++k)
        if (g.missing[(size_t) il * kExperts + g.h_ids[(size_t) il * kUsed + k]]) return true;
    return false;
}

// acc after layers [0, layers) of a round: each ringing layer adds its answer
std::vector<float> want_acc(const Rig& g, const std::vector<float>& acc0, int layers, int round) {
    std::vector<float> a = acc0;
    for (int il = 0; il < layers; ++il) {
        if (!rings_at(g, il)) continue;
        for (int i = 0; i < kEmbd; ++i) a[i] += answer(il, round, i);
    }
    return a;
}

std::vector<float> down(const float* d, size_t n) {
    std::vector<float> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

}  // namespace

int main() {
    int devices = 0;
    const cudaError_t de = cudaGetDeviceCount(&devices);
#if defined(STRATA_USE_HIP)
    const bool none = de == hipErrorNoDevice;
#else
    const bool none = de == cudaErrorNoDevice;
#endif
    if (none || (de == cudaSuccess && devices == 0)) {
        std::printf("no card\n");
        return 77;
    }
    if (de != cudaSuccess) {
        std::printf("FAIL the runtime: %s\n", cudaGetErrorString(de));
        return 1;
    }
    Rig g;
    g.bell = (ExpertBell*) host_live(sizeof(ExpertBell));
    g.hx = (float*) host_live(kEmbd * sizeof(float));
    g.hy = (float*) host_live(kEmbd * sizeof(float));
    if (!g.bell || !g.hx || !g.hy) {
        std::printf("FAIL pinned memory\n");
        return 1;
    }
    ck(cudaMalloc(&g.seq, 2 * sizeof(uint32_t)));
    ck(cudaMemset(g.seq, 0, 2 * sizeof(uint32_t)));
    ck(cudaMalloc(&g.slots, (size_t) kLayers * kExperts * sizeof(void*)));
    ck(cudaMalloc(&g.ids, (size_t) kLayers * kUsed * sizeof(int32_t)));
    ck(cudaMalloc(&g.w, (size_t) kLayers * kUsed * sizeof(float)));
    ck(cudaMalloc(&g.x, (size_t) kLayers * kEmbd * sizeof(float)));
    ck(cudaMalloc(&g.acc, kEmbd * sizeof(float)));
    ck(cudaStreamCreateWithFlags(&g.st, cudaStreamNonBlocking));
    ck(cudaEventCreateWithFlags(&g.done, cudaEventDisableTiming));
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h_x((size_t) kLayers * kEmbd), acc0(kEmbd);
    const auto fresh = [&] {
        for (float& v : h_x) v = nd(rng);
        for (float& v : acc0) v = nd(rng);
        ck(cudaMemcpy(g.x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(g.acc, acc0.data(), acc0.size() * sizeof(float), cudaMemcpyHostToDevice));
    };

    // 1. no expert outside VRAM: no ring, nothing added
    {
        make_layers(g, rng, true);
        fresh();
        const uint32_t ring0 = g.bell->ring;
        launch_layer(g, 0);
        int answered = 0;
        bool jobs_ok = true;
        expect(serve(g, h_x, 0, -1, &answered, &jobs_ok), "no miss: the card finishes");
        expect(answered == 0 && g.bell->ring == ring0, "no miss: no ring");
        expect(down(g.acc, kEmbd) == acc0, "no miss: acc unchanged");
    }

    // 2. one layer a call, rung and answered while the card waits: 2,000 rounds with new numbers each
    {
        make_layers(g, rng, false);
        int il = 0;
        while (il + 1 < kLayers && !rings_at(g, il)) ++il;
        expect(rings_at(g, il), "one layer: a layer with a miss");
        bool all_ok = true, jobs_ok = true;
        double worst_us = 0, sum_us = 0;
        const int rounds = 2000;
        for (int round = 0; round < rounds; ++round) {
            for (int i = 0; i < kEmbd; ++i) h_x[(size_t) il * kEmbd + i] = nd(rng);
            for (float& v : acc0) v = nd(rng);
            ck(cudaMemcpyAsync(g.x + (size_t) il * kEmbd, h_x.data() + (size_t) il * kEmbd, kEmbd * sizeof(float),
                               cudaMemcpyHostToDevice, g.st));
            ck(cudaMemcpyAsync(g.acc, acc0.data(), kEmbd * sizeof(float), cudaMemcpyHostToDevice, g.st));
            ck(cudaStreamSynchronize(g.st));
            const Clock::time_point t0 = Clock::now();
            launch_layer(g, il);
            int answered = 0;
            delay_us = round % 10 == 9 ? 200 + 200 * (round % 7) : 0;   // some rounds answer late
            const bool fin = serve(g, h_x, round, -1, &answered, &jobs_ok);
            delay_us = 0;
            const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            worst_us = std::max(worst_us, us), sum_us += us;
            std::vector<float> want = acc0;
            for (int i = 0; i < kEmbd; ++i) want[i] += answer(il, round, i);
            if (!fin || answered != 1 || down(g.acc, kEmbd) != want) {
                if (all_ok) std::printf("  round %d: finished %d, answered %d\n", round, (int) fin, answered);
                all_ok = false;
            }
        }
        expect(jobs_ok, "one layer: the jobs and the input as the card had them");
        expect(all_ok, "one layer: acc + the host's answer, every round");
        std::printf("  one layer a call: %d rounds, %.1f us a call on average, worst %.0f us (launch, ring, answer, add, "
                    "the card done)\n", rounds, sum_us / rounds, worst_us);
    }

    // 3. 48 layers in one recording, answered as they ring (bell_serve() after a whole pass), 20 rounds
    {
        make_layers(g, rng, false);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        bool rec = cudaStreamBeginCapture(g.st, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        for (int il = 0; il < kLayers && rec; ++il) launch_layer(g, il);
        rec = rec && cudaStreamEndCapture(g.st, &graph) == cudaSuccess &&
              cudaGraphInstantiateWithFlags(&exec, graph, 0) == cudaSuccess;
        expect(rec, "recording: 48 layers recorded");
        int rings = 0;
        for (int il = 0; il < kLayers; ++il) rings += rings_at(g, il);
        bool all_ok = rec, jobs_ok = true;
        double sum_us = 0;
        const int rounds = 20;
        for (int round = 0; round < rounds && rec; ++round) {
            fresh();
            const Clock::time_point t0 = Clock::now();
            ck(cudaGraphLaunch(exec, g.st));
            int answered = 0;
            const bool fin = serve(g, h_x, round, -1, &answered, &jobs_ok);
            sum_us += std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            if (!fin || answered != rings || down(g.acc, kEmbd) != want_acc(g, acc0, kLayers, round)) {
                if (all_ok) std::printf("  round %d: finished %d, answered %d of %d\n", round, (int) fin, answered, rings);
                all_ok = false;
            }
        }
        expect(jobs_ok, "recording: the jobs and the inputs as the card had them");
        expect(all_ok, "recording: acc + every ringing layer's answer, every round");
        std::printf("  recording: %d of %d layers ring, %.0f us a pass on average\n", rings, kLayers, sum_us / rounds);
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }

    // 3b. the token's size: per layer 15 kernels between the ring and the add and 15 after, in two recordings (4
    //     layers, then 44), as Engine::run() launches a whole pass, then served
    {
        make_layers(g, rng, false);
        float* sink = nullptr;
        ck(cudaMalloc(&sink, sizeof(float)));
        Big b{};
        b.out = sink;
        cudaGraph_t graph[2] = {};
        cudaGraphExec_t exec[2] = {};
        bool rec = true;
        for (int part = 0; part < 2 && rec; ++part) {
            rec = cudaStreamBeginCapture(g.st, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
            for (int il = part ? 4 : 0; il < (part ? kLayers : 4) && rec; ++il) {
                bell_ring(g.slots + (size_t) il * kExperts, g.ids + (size_t) il * kUsed, g.w + (size_t) il * kUsed, kUsed,
                          kUsed, il, g.x + (size_t) il * kEmbd, g.hx, kEmbd, g.bell, g.seq, g.st);
                for (int k = 0; k < 15; ++k) filler_kernel<<<1, 64, 0, g.st>>>(b);
                bell_add(g.bell, g.seq, g.hy, g.acc, kEmbd, g.st);
                for (int k = 0; k < 15; ++k) filler_kernel<<<1, 64, 0, g.st>>>(b);
            }
            rec = rec && cudaStreamEndCapture(g.st, &graph[part]) == cudaSuccess &&
                  cudaGraphInstantiateWithFlags(&exec[part], graph[part], 0) == cudaSuccess;
        }
        expect(rec, "token size: 48 layers x 32 kernels recorded");
        int rings = 0;
        for (int il = 0; il < kLayers; ++il) rings += rings_at(g, il);
        bool all_ok = rec, jobs_ok = true;
        double sum_us = 0;
        const int rounds = 20;
        for (int round = 0; round < rounds && rec; ++round) {
            fresh();
            const Clock::time_point t0 = Clock::now();
            ck(cudaGraphLaunch(exec[0], g.st));
            ck(cudaGraphLaunch(exec[1], g.st));
            int answered = 0;
            delay_us = round % 2 ? 300 : 0;
            const bool fin = serve(g, h_x, round, -1, &answered, &jobs_ok);
            delay_us = 0;
            sum_us += std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            if (!fin || answered != rings || down(g.acc, kEmbd) != want_acc(g, acc0, kLayers, round)) {
                if (all_ok) std::printf("  round %d: finished %d, answered %d of %d\n", round, (int) fin, answered, rings);
                all_ok = false;
            }
        }
        expect(jobs_ok && all_ok, "token size: every ring answered, acc right, every round");
        std::printf("  token size: %d kernels a pass, %d rings, %.0f us a pass on average\n", kLayers * 32, rings,
                    sum_us / rounds);
        for (int part = 0; part < 2; ++part) {
            if (exec[part]) cudaGraphExecDestroy(exec[part]);
            if (graph[part]) cudaGraphDestroy(graph[part]);
        }
        cudaFree(sink);
    }

    // 3c. an answer 3 s late (the card waits in one kernel that long: a cold first token's experts from the disk)
    {
        make_layers(g, rng, false);
        fresh();
        int il = 0;
        while (il + 1 < kLayers && !rings_at(g, il)) ++il;
        launch_layer(g, il);
        int answered = 0;
        bool jobs_ok = true;
        delay_us = 3000000;
        const bool fin = serve(g, h_x, 7, -1, &answered, &jobs_ok);
        delay_us = 0;
        std::vector<float> want = acc0;
        for (int i = 0; i < kEmbd; ++i) want[i] += answer(il, 7, i);
        expect(fin && answered == 1 && jobs_ok && down(g.acc, kEmbd) == want, "a 3 s wait: the card waits, then adds");
        // and the card still works after it
        fresh();
        launch_layer(g, il);
        expect(serve(g, h_x, 8, -1, &answered, &jobs_ok) && answered == 1, "after the 3 s wait: a ring answered");
    }

    // 4. an abort after 3 answers: the card's waits end at once and the stream finishes
    {
        make_layers(g, rng, false);
        fresh();
        for (int il = 0; il < kLayers; ++il) launch_layer(g, il);
        int answered = 0;
        bool jobs_ok = true;
        const Clock::time_point t0 = Clock::now();
        const bool fin = serve(g, h_x, 0, 3, &answered, &jobs_ok);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        expect(fin && answered == 3 && cudaStreamQuery(g.st) == cudaSuccess, "abort: the stream ends");
        expect(ms < 5000, "abort: within 5 s");
        // the bell works again after an abort
        make_layers(g, rng, false);
        fresh();
        for (int il = 0; il < kLayers; ++il) launch_layer(g, il);
        jobs_ok = true;
        expect(serve(g, h_x, 1, -1, &answered, &jobs_ok) && jobs_ok &&
                   down(g.acc, kEmbd) == want_acc(g, acc0, kLayers, 1),
               "after an abort: 48 layers answered again");
    }

    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess && e != cudaErrorNotReady) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        ++failed;
    }
    std::printf("%d checks, %d failed\n", checks, failed);
    return failed ? 1 : 0;
}
