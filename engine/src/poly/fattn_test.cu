// src/poly/fattn_test.cu - fattn_q8() (llama.cpp's flash-attention kernel, fattn.hpp) against attend_batch() on the
// same q8_0 cache, Qwen3.6's shape (heads of 256 values, 16 heads, 2 KV heads), and both against the arithmetic in
// doubles for three tokens of each batch.  Also with a scratch that holds only a few thousand keys, so that the
// keys go in many blocks and are merged.  Then the same cases with fattn_f16() on an fp16 cache of the same values.
//
// llama.cpp's kernel sums the values' products in halves on NVIDIA cards (T_C_VKQ in fattn_mma.cuh, upstream's
// choice, not one of our changes; attend_batch() sums in floats), so it is the further from the doubles, and by how
// much depends on the card: an RTX 3070 (sm_86) and an RTX 4070 (sm_89) give the same 1.06e-2 in one case where an
// RTX 5090 (sm_120) gives 4.85e-3.  So the bounds scale with the largest result and are about twice the most sm_86
// gives (9.4e-3 of the largest result from attend_batch(), 3.4e-3 from the doubles).  They catch a wrong mask or
// merge, which is off by tenths.
//
// The inputs are drawn from mt19937's raw output (draw()), not std::normal_distribution, whose numbers differ between
// standard libraries, so that every platform tests the same data.
//
// Each case's line ends with a digest of the kernel's results ("bits"): a default build (fattn_mma_d256.cuh) and a
// -DSTRATA_POLY_MMQ=ON one (fattn_mma.cuh) print the same digests on one card when their sums are the same.
//
// usage: poly_fattn_test   (exit 0: every case within kRoomBatch and kRoomDoubles, times max(1, largest result), of
//        attend_batch() and the doubles)
//        poly_fattn_test [dec] gqa16   the same checks with Ornith 122B's 32 heads / 2 KV heads
//        poly_fattn_test dec   fdec_q8() and fdec_f16() (the decode kernel) against attend() and the doubles, up
//                              to 170,000 rows; a token alone and with others, and from a position on the card,
//                              to the bit; in Qwen35's scratch and in a quarter of it (the splits of 1,024 keys)
#include "fattn.hpp"
#include "strata/poly/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kDim = 256, kKvHeads = 2, kWidth = kKvHeads * kDim, kMaxRows = 131072, kBatch = 1024;
int kHeads = 16;   // gqa16 selects 32; the existing group-8 geometry remains the default
constexpr size_t kScore = size_t(16) << 20;   // Qwen35's score_ (kScoreFloats)
constexpr float kRoomBatch = 2e-2f, kRoomDoubles = 7.5e-3f;   // the bounds, as shares of the largest result (at least 1)

// the same numbers from the same seed on every standard library: uniform in (0, 1) from the raw 32 bits, and a
// normal one by Box-Muller on two of them
double uniform(std::mt19937& rng) { return (rng() + 0.5) / 4294967296.0; }
float draw(std::mt19937& rng) {
    const double u = uniform(rng), v = uniform(rng);
    return (float) (std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v));
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

struct Cache {
    std::vector<int8_t> codes;
    std::vector<uint16_t> scales;
    std::vector<uint16_t> halves;   // the same values in fp16 (fp16())
    float at(size_t row, int i) const {
        if (!halves.empty()) return __half2float(__ushort_as_half(halves[row * kWidth + i]));
        return (float) codes[row * kWidth + i] * __half2float(__ushort_as_half(scales[row * (kWidth / 32) + i / 32]));
    }
    void fp16() {
        std::vector<uint16_t> h(codes.size());
        for (size_t i = 0; i < h.size(); ++i) h[i] = __half_as_ushort(__float2half(at(i / kWidth, (int) (i % kWidth))));
        halves = std::move(h);
    }
};

Cache make(std::mt19937& rng, size_t rows = kMaxRows) {
    Cache c;
    c.codes.resize(rows * kWidth);
    c.scales.resize(rows * (kWidth / 32));
    for (auto& x : c.codes) x = (int8_t) std::clamp(std::lround(40.f * draw(rng)), -127L, 127L);
    for (auto& x : c.scales) x = __half_as_ushort(__float2half((float) (0.5 + uniform(rng)) / 40.f));
    return c;
}

// kh and vh: the fp16 caches (fattn_f16()), or null for the q8 ones (fattn_q8())
int check(cublasHandle_t blas, const Cache& K, const Cache& V, Kvq8 kq, Kvq8 vq, const uint16_t* kh,
          const uint16_t* vh, const std::vector<float>& q, const float* dq, float* score, float* awork,
          uint16_t* kwork, float* out, long long pos, int n, size_t fattn_floats) {
    const size_t cells = size_t(n) * kHeads * kDim;
    const float scale = 1.f / 16.f;
    const size_t halves = size_t(2048) * kWidth;
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0), cudaEventCreate(&e1), cudaEventCreate(&e2), cudaEventCreate(&e3);
    cudaEventRecord(e0);
    if (kh)
        attend_batch(blas, dq, kh, vh, pos, n, score, kScore, awork, out, kDim, kHeads, kKvHeads, scale, nullptr, 0, 0,
                     false, Kvq8{}, Kvq8{}, nullptr, nullptr, 0);
    else
        attend_batch(blas, dq, nullptr, nullptr, pos, n, score, kScore, awork, out, kDim, kHeads, kKvHeads, scale,
                     nullptr, 0, 0, false, kq, vq, kwork, kwork + halves, halves);
    cudaEventRecord(e1);
    const std::vector<float> ref = download(out, cells);
    cudaMemset(out, 0, cells * sizeof(float));
    cudaEventRecord(e2);
    if (!(kh ? fattn_f16(dq, kh, vh, pos, n, out, kDim, kHeads, kKvHeads, scale, score, fattn_floats, nullptr)
             : fattn_q8(dq, kq, vq, pos, n, out, kDim, kHeads, kKvHeads, scale, score, fattn_floats, nullptr))) {
        std::printf("FAIL pos %6lld rows %4d: fattn refused the scratch\n", pos, n);
        return 1;
    }
    cudaEventRecord(e3);
    const std::vector<float> fa = download(out, cells);
    float ms_ref = 0.f, ms_fa = 0.f;
    cudaEventElapsedTime(&ms_ref, e0, e1);
    cudaEventElapsedTime(&ms_fa, e2, e3);
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
        std::printf("FAIL pos %6lld rows %4d: CUDA %s\n", pos, n, cudaGetErrorString(e));
        return 1;
    }
    float most = 0.f, big = 0.f;
    double sum = 0;
    for (size_t i = 0; i < cells; ++i) {
        const float d = std::fabs(fa[i] - ref[i]);
        most = std::max(most, std::isnan(d) ? INFINITY : d);
        sum += d;
        big = std::max(big, std::fabs(ref[i]));
    }
    // three tokens in doubles: the first, the middle and the last
    float most_fa = 0.f, most_ref = 0.f;
    for (int t : {0, n / 2, n - 1})
        for (int h = 0; h < kHeads; ++h) {
            const int g = h / (kHeads / kKvHeads);
            const float* qh = q.data() + (size_t(t) * kHeads + h) * kDim;
            const long long keys = pos + t + 1;
            std::vector<double> w(keys);
            double top = -1e300;
            for (long long r = 0; r < keys; ++r) {
                double s = 0;
                for (int i = 0; i < kDim; ++i) s += (double) qh[i] * K.at(r, g * kDim + i);
                w[r] = s * scale;
                top = std::max(top, w[r]);
            }
            double z = 0;
            for (auto& x : w) z += x = std::exp(x - top);
            for (int i = 0; i < kDim; ++i) {
                double o = 0;
                for (long long r = 0; r < keys; ++r) o += w[r] * V.at(r, g * kDim + i);
                o /= z;
                const size_t at = (size_t(t) * kHeads + h) * kDim + i;
                most_fa = std::max(most_fa, (float) std::fabs(fa[at] - o));
                most_ref = std::max(most_ref, (float) std::fabs(ref[at] - o));
            }
        }
    const float room = std::max(1.f, big);
    const bool ok = most <= kRoomBatch * room && most_fa <= kRoomDoubles * room;
    // FNV-1a of the results' bits: the same digest from two builds on one card is the same sums
    unsigned long long digest = 14695981039346656037ull;
    for (const float x : fa) {
        uint32_t u;
        std::memcpy(&u, &x, 4);
        for (int b = 0; b < 32; b += 8) digest = (digest ^ ((u >> b) & 0xff)) * 1099511628211ull;
    }
    std::printf("%s %s pos %6lld rows %4d scratch %5.1f MiB: %.2e from attend_batch (mean %.1e, results up to %.2f); "
                "against doubles: fattn %.2e, attend_batch %.2e; %.2f ms against %.2f; bits %016llx\n",
                ok ? "ok  " : "FAIL", kh ? "f16" : "q8 ", pos, n, fattn_floats * 4.0 / (1 << 20), most, sum / cells, big, most_fa,
                most_ref, ms_fa, ms_ref, digest);
    return ok ? 0 : 1;
}

// The doubles' result of token t's head h attending the rows 0 to `keys` - 1.
std::vector<double> exact(const Cache& K, const Cache& V, const float* qh, int g, long long keys, float scale) {
    std::vector<double> w(keys), o(kDim, 0.0);
    double top = -1e300, z = 0;
    for (long long r = 0; r < keys; ++r) {
        double s = 0;
        for (int i = 0; i < kDim; ++i) s += (double) qh[i] * K.at(r, g * kDim + i);
        top = std::max(top, w[r] = s * scale);
    }
    for (auto& x : w) z += x = std::exp(x - top);
    for (long long r = 0; r < keys; ++r)
        for (int i = 0; i < kDim; ++i) o[i] += w[r] * V.at(r, g * kDim + i);
    for (double& x : o) x /= z;
    return o;
}

int decode_main() {
    constexpr int kRows = 172032, kMaxN = 9;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(size_t(kMaxN) * kHeads * kDim);
    for (float& x : q) x = 2.f * nd(rng);
    const Cache K = make(rng, kRows), V = make(rng, kRows);
    int8_t *kc = upload(K.codes), *vc = upload(V.codes);
    uint16_t *ks = upload(K.scales), *vs = upload(V.scales);
    const Kvq8 kq{kc, ks}, vq{vc, vs};
    // the same values as an fp16 cache
    std::vector<uint16_t> kf(K.codes.size()), vf(V.codes.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        kf[i] = __half_as_ushort(__float2half(K.at(i / kWidth, int(i % kWidth))));
        vf[i] = __half_as_ushort(__float2half(V.at(i / kWidth, int(i % kWidth))));
    }
    uint16_t *kh = upload(kf), *vh = upload(vf);
    bool f16 = false;
    float* dq = upload(q);
    float *score, *out;
    long long* at;
    const size_t cells = size_t(kMaxN) * kHeads * kDim;
    cudaMalloc(&score, kScore * sizeof(float));
    cudaMalloc(&out, cells * sizeof(float));
    cudaMalloc(&at, sizeof(long long));
    const float scale = 1.f / 16.f;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    auto old = [&](long long pos, int n) {
        // The reference writes chunk * heads * keys scores. Group 16 exceeds kScore at long context with 4.
        const int chunk = std::min<size_t>(4, kScore / (size_t(kHeads) * (pos + n)));
        attend(dq, f16 ? kh : nullptr, f16 ? vh : nullptr, pos, n, score, chunk, out, kDim, kHeads, kKvHeads, scale, nullptr,
               0, 0, false, nullptr, 0, 0, f16 ? Kvq8{} : kq, f16 ? Kvq8{} : vq);
    };
    // a quarter of Qwen35's scratch holds the splits of 1,024 keys but not those of 256 for the cases from 120,000
    // rows on: they take the wider splits (split_keys(), as a --max-context of 262,144 does)
    size_t floats = kScore;
    auto dec = [&](long long pos, int n, bool on_card) {
        int room = 1024;
        while (room < pos + n) room *= 2;
        if (on_card) cudaMemcpy(at, &pos, sizeof pos, cudaMemcpyHostToDevice);
        const long long* a = on_card ? at : nullptr;
        return f16 ? fdec_f16(dq, kh, vh, pos, n, a, std::min(room, kRows), out, kDim, kHeads, kKvHeads, scale, score,
                              floats, nullptr)
                   : fdec_q8(dq, kq, vq, pos, n, a, std::min(room, kRows), out, kDim, kHeads, kKvHeads, scale, score,
                             floats, nullptr);
    };
    auto time = [&](auto&& run) {
        run();
        cudaEventRecord(e0);
        for (int i = 0; i < 10; ++i) run();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / 10;
    };
    int bad = 0;
    struct Case { long long pos; int n; };
    const Case cases[] = {{0, 1}, {5, 3}, {1022, 2}, {1023, 4}, {1024, 1}, {2047, 5}, {30000, 2},
                          {120000, 3}, {140000, 1}, {140000, 3}, {170000, 4}, {170000, 6}, {170000, 9}};
    for (const size_t scratch : {kScore, kScore / 4})
    for (const bool half_cache : {false, true})
    for (const Case& c : cases) {
        f16 = half_cache;
        floats = scratch;
        const size_t n_cells = size_t(c.n) * kHeads * kDim;
        old(c.pos, c.n);
        const std::vector<float> ref = download(out, n_cells);
        cudaMemset(out, 0, cells * sizeof(float));
        if (!dec(c.pos, c.n, false)) {
            std::printf("FAIL %s pos %6lld rows %d: refused\n", f16 ? "f16" : "q8 ", c.pos, c.n);
            ++bad;
            continue;
        }
        const std::vector<float> fd = download(out, n_cells);
        // from the card's position, and each token alone at its own position: the same bits
        dec(c.pos, c.n, true);
        const std::vector<float> fd_at = download(out, n_cells);
        bool same = fd_at == fd;
        for (int t = 0; t < c.n; ++t) {
            // token t alone: its query moved to the front
            cudaMemcpy(dq, q.data() + size_t(t) * kHeads * kDim, size_t(kHeads) * kDim * sizeof(float),
                       cudaMemcpyHostToDevice);
            dec(c.pos + t, 1, true);
            const std::vector<float> one = download(out, size_t(kHeads) * kDim);
            same = same && std::equal(one.begin(), one.end(), fd.begin() + size_t(t) * kHeads * kDim);
            cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
            std::printf("FAIL pos %6lld rows %d: CUDA %s\n", c.pos, c.n, cudaGetErrorString(e));
            return 1;
        }
        float most = 0.f, big = 0.f, most_fd = 0.f, most_ref = 0.f;
        for (size_t i = 0; i < n_cells; ++i) {
            const float d = std::fabs(fd[i] - ref[i]);
            most = std::max(most, std::isnan(d) ? INFINITY : d);
            big = std::max(big, std::fabs(ref[i]));
        }
        for (int t : {0, c.n - 1})
            for (int h = 0; h < kHeads; h += c.pos > 50000 ? 5 : 1) {
                const std::vector<double> o =
                    exact(K, V, q.data() + (size_t(t) * kHeads + h) * kDim, h / (kHeads / kKvHeads), c.pos + t + 1, scale);
                for (int i = 0; i < kDim; ++i) {
                    const size_t a = (size_t(t) * kHeads + h) * kDim + i;
                    most_fd = std::max(most_fd, (float) std::fabs(fd[a] - o[i]));
                    most_ref = std::max(most_ref, (float) std::fabs(ref[a] - o[i]));
                }
            }
        const float ms_old = time([&] { old(c.pos, c.n); }), ms_dec = time([&] { dec(c.pos, c.n, false); });
        const bool ok = same && most <= 1e-2f && most_fd <= 1e-2f;
        bad += !ok;
        std::printf("%s %s %s pos %6lld rows %d: %.2e from attend() (results up to %.2f); against doubles: fdec %.2e, "
                    "attend %.2e; %s; %.3f ms against %.3f\n", ok ? "ok  " : "FAIL", f16 ? "f16" : "q8 ",
                    floats == kScore ? "scratch 1/1" : "scratch 1/4", c.pos, c.n, most, big, most_fd, most_ref,
                    same ? "same bits alone and on the card" : "BITS DIFFER", ms_dec, ms_old);
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}

// Standalone Q8 FDEC A/B. Test exactly the same split layout, input and
// position for every mode. Timings are CUDA event intervals for both kernels
// combined; no timing events are added to the production engine.
int decode_ab(int repeats) {
    constexpr int rows = 262144, max_n = 9;
    cudaDeviceProp prop{};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 1;
    if (!fdec_fits(kDim, kHeads, kKvHeads, rows, kScore)) {
        std::fprintf(stderr, "FDEC unavailable on this device / scratch\n");
        return 1;
    }
    std::printf("FDEC A/B %s sm_%d%d SMs=%d heads=%d kv_heads=%d repeats=%d\n", prop.name,
                prop.major, prop.minor, prop.multiProcessorCount, kHeads, kKvHeads, repeats);
    for (int mt : {1,2}) for (bool late : {false,true}) {
        FdecKernelInfo info;
        if (!fdec_q8_kernel_info(kHeads/kKvHeads, mt, late, info)) {
            std::fprintf(stderr, "FAIL kernel attributes\n"); return 1;
        }
        std::printf("fdec-kernel-info: MT=%d late_v=%d regs/thread=%d local_bytes/thread=%zu static_shared=%zu "
                    "max_blocks/SM=%d threads/block=%d theoretical_warps/SM=%d\n",
                    mt, (int)late, info.registers, info.local_bytes, info.shared_bytes,
                    info.blocks_per_sm, info.threads_per_block, info.blocks_per_sm*info.threads_per_block/32);
    }
    auto check_cuda = [](cudaError_t e) {
        if (e == cudaSuccess) return;
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    };
    // Fast deterministic synthetic valid cache (no slow normal draws for 2*134M bytes).
    std::mt19937 rng(119);
    auto cache = [&] {
        Cache a;
        a.codes.resize(size_t(rows)*kWidth);
        a.scales.resize(size_t(rows)*(kWidth/32));
        for (auto& x:a.codes) x=(int8_t)((int)(rng()%255)-127);
        for (auto& x:a.scales) x=__half_as_ushort(__float2half((float)(0.5+uniform(rng))/80.f));
        return a;
    };
    Cache K=cache(), V=cache();
    int8_t *kc=upload(K.codes), *vc=upload(V.codes);
    uint16_t *ks=upload(K.scales), *vs=upload(V.scales);
    check_cuda(cudaGetLastError());
    std::vector<float> q(size_t(max_n)*kHeads*kDim);
    for (auto& x:q) x=2.f*draw(rng);
    float* dq=upload(q);
    float *scratch=nullptr, *out=nullptr, *ml=nullptr;
    long long* at=nullptr;
    check_cuda(cudaMalloc(&scratch,kScore*sizeof(float)));
    check_cuda(cudaMalloc(&out,q.size()*sizeof(float)));
    check_cuda(cudaMalloc(&ml,size_t(max_n)*kHeads*2*sizeof(float)));
    check_cuda(cudaMalloc(&at,sizeof(long long)));
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream));
    cudaEvent_t e0,e1;
    check_cuda(cudaEventCreate(&e0)); check_cuda(cudaEventCreate(&e1));
    const float scale=1.f/16.f;
    struct Case { long long pos; int n, room; bool whole; size_t stride, sf; };
    std::vector<Case> cases;
    // Exact-position dispatch, recorded-position dispatch, both sides of
    // split boundaries; 256K scratch fallback to 1024-key splits.
    for (long long pos:{4095LL,16383LL,32767LL,65535LL,131071LL,262135LL})
        for (int n:{1,2,4,8}) cases.push_back({pos,n,0,false,0,kScore});
    for (int n:{1,3,4,9}) cases.push_back({65527,n,131072,false,0,kScore});
    cases.push_back({32763,4,65536,true,0,kScore});
    cases.push_back({32763,4,65536,false,2*size_t(kWidth),kScore});
    cases.push_back({120003,3,131072,false,0,kScore/4});
    int bad=0;
    auto same = [](const std::vector<float>& a,const std::vector<float>& b) {
        return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
    };
    for (const auto& c:cases) {
        const Kvq8 kq{kc,ks,c.stride}, vq{vc,vs,c.stride};
        if (c.room) check_cuda(cudaMemcpyAsync(at,&c.pos,sizeof(c.pos),cudaMemcpyHostToDevice,stream));
        const size_t cells=size_t(c.n)*kHeads*kDim, meta=size_t(c.n)*kHeads*2;
        auto run=[&](int mode) {
            if(!fdec_q8(dq,kq,vq,c.pos,c.n,c.room?at:nullptr,c.room,out,kDim,kHeads,kKvHeads,scale,
                        scratch,c.sf,stream,c.whole,ml,mode)) {
                std::fprintf(stderr,"FAIL refused pos=%lld n=%d mode=%d\n",c.pos,c.n,mode);
                std::exit(1);
            }
        };
        auto read=[&](float* p,size_t n) {
            std::vector<float> v(n);
            check_cuda(cudaMemcpyAsync(v.data(),p,n*sizeof(float),cudaMemcpyDeviceToHost,stream));
            check_cuda(cudaStreamSynchronize(stream));
            return v;
        };
        run(0);
        const auto ref=read(out,cells), rm=read(ml,meta);
        bool ok=std::all_of(ref.begin(),ref.end(),[](float x){return std::isfinite(x);}) &&
                std::all_of(rm.begin(),rm.end(),[](float x){return std::isfinite(x);});
        for(int mode=1;mode<=3;++mode) {
            run(mode);
            const auto test=read(out,cells), tm=read(ml,meta);
            if(!same(ref,test) || !same(rm,tm)) {
                float err=0, em=0;
                for(size_t i=0;i<cells;++i) err=std::max(err,std::fabs(ref[i]-test[i]));
                for(size_t i=0;i<meta;++i) em=std::max(em,std::fabs(rm[i]-tm[i]));
                std::printf("FAIL bitwise mode=%d pos=%lld n=%d output_abs=%g meta_abs=%g\n",mode,c.pos,c.n,err,em);
                ok=false;
            }
        }
        if(!ok) { ++bad; continue; }
        // Paired GPU event intervals in alternating order. Warm each mode;
        // do not include host readback, first launch, or cache allocation.
        auto time=[&](int mode) {
            for(int i=0;i<3;++i) run(mode);
            check_cuda(cudaEventRecord(e0,stream));
            for(int i=0;i<repeats;++i) run(mode);
            check_cuda(cudaEventRecord(e1,stream)); check_cuda(cudaEventSynchronize(e1));
            float ms=0; check_cuda(cudaEventElapsedTime(&ms,e0,e1));
            return double(ms)*1000/repeats;
        };
        std::printf("fdec-ab: pos=%lld n=%d room=%d whole=%d stride=%zu scratch_MiB=%.0f parity PASS",
                    c.pos,c.n,c.room,(int)c.whole,c.stride,c.sf*4.0/(1<<20));
        for(int mode=1;mode<=3;++mode) {
            double a=0,b=0;
            for(int r=0;r<3;++r) {
                if(r%2==0) { a+=time(0); b+=time(mode); }
                else { b+=time(mode); a+=time(0); }
            }
            std::printf("; mode%d %.2f -> %.2f us (%+.2f%%)",mode,a/3,b/3,100*(a/b-1));
        }
        std::printf("\n"); std::fflush(stdout);
    }
    check_cuda(cudaEventDestroy(e0)); check_cuda(cudaEventDestroy(e1));
    check_cuda(cudaStreamDestroy(stream));
    for(void* p:{(void*)kc,(void*)vc,(void*)ks,(void*)vs,(void*)dq,(void*)scratch,(void*)out,(void*)ml,(void*)at})
        check_cuda(cudaFree(p));
    std::printf("FDEC A/B: %s (%d failed cases)\n",bad?"FAIL":"PASS",bad);
    return bad?1:0;
}

}  // namespace

int main(int argc, char** argv) {
    bool decode = false, ab = false;
    int repeats = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "dec") decode = true;
        else if (arg == "dec-ab") ab = true;
        else if (arg == "gqa16") kHeads = 32;
        else if (arg == "--repeats" && i+1<argc) {
            char* end=nullptr;
            const char* raw=argv[++i];
            const long n=std::strtol(raw,&end,10);
            if(end==raw || *end || n<1 || n>1000) return 2;
            repeats=(int)n;
        }
        else {
            std::fprintf(stderr, "usage: poly_fattn_test [dec|dec-ab] [gqa16] [--repeats N]\n");
            return 2;
        }
    }
    std::printf("query heads %d, KV heads %d, group %d\n", kHeads, kKvHeads, kHeads / kKvHeads);
    if (fattn_fits(kDim, 24, kKvHeads, kBatch, kScore) ||
        fdec_fits(kDim, 24, kKvHeads, 1024, kScore) ||
        fattn_fits(128, kHeads, kKvHeads, kBatch, kScore) ||
        fdec_fits(128, kHeads, kKvHeads, 1024, kScore) ||
        fattn_fits(kDim, kHeads, 0, kBatch, kScore) ||
        fdec_fits(kDim, kHeads, 0, 1024, kScore) ||
        fattn_fits(kDim, kHeads, kKvHeads, kBatch, 0) ||
        fdec_fits(kDim, kHeads, kKvHeads, 1024, 0)) {
        std::printf("FAIL: unsupported shape or insufficient scratch accepted\n");
        return 1;
    }
    if (ab) return decode_ab(repeats);
    if (decode) {
        if (!fdec_fits(kDim, kHeads, kKvHeads, 172032, kScore)) {
            std::printf("no decode kernel for this card\n");
            return 1;
        }
        return decode_main();
    }
    if (!fattn_fits(kDim, kHeads, kKvHeads, kBatch, kScore)) {
        std::printf("no kernel for this card\n");
        return 1;
    }
    // a batch whose results alone fill Qwen35's score_ (--prompt-batch 4096) is refused up front, not call by call
    if (fattn_fits(kDim, kHeads, kKvHeads, 4096, kScore)) {
        std::printf("FAIL: fattn_fits() takes a batch of 4096 rows in a scratch that cannot hold its results\n");
        return 1;
    }
    std::mt19937 rng(7);
    std::vector<float> q(size_t(kBatch) * kHeads * kDim);
    for (float& x : q) x = 2.f * draw(rng);   // spread scores, so the softmax has a few large weights
    Cache K = make(rng), V = make(rng);
    int8_t *kc = upload(K.codes), *vc = upload(V.codes);
    uint16_t *ks = upload(K.scales), *vs = upload(V.scales);
    const Kvq8 kq{kc, ks}, vq{vc, vs};
    float* dq = upload(q);
    float *score, *awork, *out;
    uint16_t* kwork;
    const size_t heads = size_t(kBatch) * kHeads * kDim;
    cudaMalloc(&score, kScore * sizeof(float));
    cudaMalloc(&awork, (heads + heads / 2 + 1 + 3 * size_t(kBatch) * kHeads) * sizeof(float));
    cudaMalloc(&kwork, size_t(2) * 2048 * kWidth * sizeof(uint16_t));
    cudaMalloc(&out, heads * sizeof(float));
    cublasHandle_t blas;
    cublasCreate(&blas);
    cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH);
    int bad = 0;
    struct Case { long long pos; int n; size_t floats; };
    // a few thousand keys a block: the batch's results (n * 4096 floats) and its metadata, then rows of 512 floats
    auto few = [](int n, int rows) {
        const int tile_tokens = 64 / (kHeads / kKvHeads);
        return size_t(n) * kHeads * (kDim + 4) + 64 + size_t(n + tile_tokens - 1) / tile_tokens + size_t(rows) * kWidth;
    };
    const Case cases[] = {{0, 16, kScore},          {0, 64, kScore},          {0, kBatch, kScore},
                          {5, 37, kScore},          {1000, 517, kScore},      {4095, kBatch, kScore},
                          {30000, kBatch, kScore},  {30000, kBatch, few(kBatch, 4096)},
                          {30001, 100, few(100, 160)}, {120000, kBatch, kScore}};
    for (const Case& c : cases)
        bad += check(blas, K, V, kq, vq, nullptr, nullptr, q, dq, score, awork, kwork, out, c.pos, c.n, c.floats);
    // the fp16 cache in place of the q8 one, which is freed first: the card may be shared with a running server
    cudaFree(kc), cudaFree(vc), cudaFree(ks), cudaFree(vs);
    K.fp16(), V.fp16();
    uint16_t *kh = upload(K.halves), *vh = upload(V.halves);
    for (const Case& c : cases)
        bad += check(blas, K, V, Kvq8{}, Kvq8{}, kh, vh, q, dq, score, awork, kwork, out, c.pos, c.n, c.floats);
    cublasDestroy(blas);
    cudaFree(kh), cudaFree(vh), cudaFree(dq), cudaFree(score), cudaFree(awork);
    cudaFree(kwork), cudaFree(out);
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
