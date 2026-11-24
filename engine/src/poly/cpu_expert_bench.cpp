// Deterministic Q5_K CPU expert A/B microbenchmark. No GPU or GGUF required.
#include "cpu_experts.hpp"
#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/types.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace Q = strata::quant;
using namespace strata::poly;
namespace {
constexpr int type = (int) Q::Type::Q5_K;
constexpr int ne = 2048, nf = 512;
using Bytes = std::vector<uint8_t>;
struct Weights { Bytes g,u,d; };

Bytes weights(int cols, int rows, std::mt19937& rng) {
    const auto* ti = Q::type_info(type);
    Bytes w(Q::row_bytes(type,cols)*(size_t)rows);
    for (auto& v : w) v = (uint8_t)rng();
    for (size_t i=0; i<w.size(); i+=ti->bytes) {
        // Small finite half scales. Randomize magnitudes and signs to exercise correction.
        const uint16_t d = (uint16_t)(0x1400u + rng()%0x600u) | (uint16_t)((rng()&1u)<<15);
        const uint16_t dm = (uint16_t)(0x1000u + rng()%0x600u);
        std::memcpy(w.data()+i,&d,2);
        std::memcpy(w.data()+i+2,&dm,2);
    }
    return w;
}

bool equal(const std::vector<float>& a,const std::vector<float>& b) {
    if (a.size()!=b.size()) return false;
    for (float v:a) if (!std::isfinite(v)) return false;
    return std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
}

bool dot_check(Q::DotRowsFn multi) {
    const auto* dot=Q::dot_kernel(Q::Type::Q5_K,Q::default_arith());
    const auto* act=Q::act_kernel(Q::Type::Q8_K,Q::default_arith());
    if (!dot || !act) return false;
    std::mt19937 rng(0xa752u);
    for (int n:{256,512,1024,2048,2304,4096}) {
        std::vector<float> x((size_t)n), ref(9), out(9);
        const size_t stride=Q::row_bytes((int)Q::Type::Q8_K,n);
        Bytes aq(9*stride);
        const void* a[9];
        for (int trial=0;trial<24;++trial) {
            const Bytes w=weights(n,1,rng);
            for (int k=0;k<9;++k) {
                for (auto& v:x) v=((int)(rng()%2049)-1024)/128.f;
                act->fn(x.data(),aq.data()+k*stride,n);
                a[k]=aq.data()+k*stride;
                dot->fn(n,&ref[k],w.data(),a[k]);
            }
            for (int m=1;m<=9;++m) {
                multi(n,m,out.data(),w.data(),a);
                if (std::memcmp(ref.data(),out.data(),(size_t)m*sizeof(float))!=0) {
                    std::fprintf(stderr,"FAIL Q5_K dot parity n=%d activations=%d trial=%d\n",n,m,trial);
                    return false;
                }
            }
        }
    }
    return true;
}

struct Sample { double wall=0, down=0; int64_t uses=0, reused=0; };
struct Case {
    int rows;
    bool mixed;
    const char* name;
};
constexpr Case cases[]={{1,false,"single"},{2,false,"shared-2"},{4,false,"shared-4"},
                        {8,false,"shared-8"},{9,false,"shared-9-tail"},{4,true,"mixed-4"}};
struct Input {
    std::vector<float> x;
    std::vector<CpuExpert> ex;
};

Input input_for(const std::vector<Weights>& bank, int base, const Case& c, std::mt19937& rng) {
    Input in;
    in.x.resize((size_t)c.rows*ne);
    for (auto& v:in.x) v=((int)(rng()%201)-100)/100.f;
    for (int row=0;row<c.rows;++row) {
        // Four uses per token. Mixed case: one common expert, three unique per token.
        for(int e=0;e<4;++e) {
            const int idx=base+(c.mixed && e>0 ? 1+3*row+(e-1) : e);
            const auto& w=bank[(size_t)idx];
            in.ex.push_back({w.g.data(),w.u.data(),w.d.data(),0.25f,row});
        }
    }
    return in;
}

void run(Workers& pool,const Input& in,int rows,std::vector<float>& out,CpuExpertTiming* timing=nullptr) {
    cpu_experts(pool,in.ex.data(),(int)in.ex.size(),rows,type,type,type,ne,nf,in.x.data(),20.f,false,
                out.data(),nullptr,nullptr,nullptr,0.f,0.f,timing);
}

Sample measure(const std::vector<Input>& inputs,const Case& c,int threads,CpuExpertTaskShape shape,int calls) {
    Workers pool(threads,20000,shape);
    std::vector<float> out((size_t)c.rows*ne);
    // Warm each bank; a second worker pool is never alive during measurement.
    for (const auto& in:inputs) run(pool,in,c.rows,out);
    CpuExpertTiming p;
    const auto t=OverlapClock::now();
    for (int i=0;i<calls;++i) run(pool,inputs[(size_t)i%inputs.size()],c.rows,out,&p);
    const double wall=interval_ms(t,OverlapClock::now());
    return {wall,p.down_reduce_ms,p.expert_uses,p.down_reuse_uses};
}

int arg(const char* raw,int low,int high) {
    char* end=nullptr;
    const long n=std::strtol(raw,&end,10);
    return end!=raw && *end=='\0' && n>=low && n<=high ? (int)n : -1;
}
} // namespace

int main(int argc,char** argv) {
    const bool check=argc==2 && std::strcmp(argv[1],"--check")==0;
    int threads=12, gate=8, down=16, calls=100;
    if (!check && argc!=1) {
        if(argc!=5) { std::fprintf(stderr,"usage: poly_cpu_expert_bench [threads gate_rows down_rows calls] | --check\n"); return 2; }
        threads=arg(argv[1],1,64); gate=arg(argv[2],4,256); down=arg(argv[3],4,256); calls=arg(argv[4],8,10000);
    }
    if(check) threads=4;
    CpuExpertTaskShape old{gate,down,false}, experiment{gate,down,true};
    if(threads<1 || calls<1 || !old.valid()) return 2;
    const auto multi=Q::experimental_q5_rows_kernel(Q::default_arith());
    const auto* dot=Q::dot_kernel(Q::Type::Q5_K,Q::default_arith());
    if(!multi || !dot) {
        std::printf("Q5_K AVX2 experimental path unavailable; %s\n",check?"SKIP parity":"cannot benchmark");
        return check?0:2;
    }
    if(!dot_check(multi)) return 1;
    std::string err;
    if(!cpu_expert_types_ok(type,type,ne,nf,err)) { std::fprintf(stderr,"%s\n",err.c_str()); return 1; }
    const int banks=check?1:8, per_bank=13;
    std::mt19937 rng(0x619u);
    std::vector<Weights> bank;
    bank.reserve((size_t)banks*per_bank);
    for(int i=0;i<banks*per_bank;++i) bank.push_back({weights(ne,nf,rng),weights(ne,nf,rng),weights(nf,ne,rng)});
    std::printf("Q5_K CPU expert bench: threads=%d gate=%d down=%d calls=%d banks=%d dot=%s (bitwise dot parity PASS)\n",
                threads,gate,down,calls,banks,dot->name);
    for(const auto& c:cases) {
        std::vector<Input> inputs;
        for(int b=0;b<banks;++b) inputs.push_back(input_for(bank,b*per_bank,c,rng));
        std::vector<std::vector<float>> refs(inputs.size(),std::vector<float>((size_t)c.rows*ne));
        { Workers pool(threads,20000,old);
          for(size_t i=0;i<inputs.size();++i) run(pool,inputs[i],c.rows,refs[i]); }
        { Workers pool(threads,20000,experiment);
          std::vector<float> out((size_t)c.rows*ne);
          for(size_t i=0;i<inputs.size();++i) {
              CpuExpertTiming p;
              run(pool,inputs[i],c.rows,out,&p);
              const int expected=c.mixed?4:4*(c.rows-(c.rows%4==1?1:0));
              if(!equal(refs[i],out) || p.down_reuse_uses!=expected) {
                  std::fprintf(stderr,"FAIL expert parity/reuse count: %s bank=%zu expected=%d got=%lld\n",
                               c.name,i,expected,(long long)p.down_reuse_uses); return 1;
              }
          } }
        if(check) { std::printf("%s expert parity PASS\n",c.name); continue; }
        Sample a,b;
        for(int r=0;r<3;++r) {
            Sample aa,bb;
            if(r%2==0) { aa=measure(inputs,c,threads,old,calls); bb=measure(inputs,c,threads,experiment,calls); }
            else { bb=measure(inputs,c,threads,experiment,calls); aa=measure(inputs,c,threads,old,calls); }
            a.wall+=aa.wall; a.down+=aa.down; a.uses+=aa.uses;
            b.wall+=bb.wall; b.down+=bb.down; b.uses+=bb.uses; b.reused+=bb.reused;
            std::printf("%s round %d: total %.2f -> %.2f ms, down %.2f -> %.2f ms\n",
                        c.name,r+1,aa.wall,bb.wall,aa.down,bb.down);
        }
        std::printf("%s SUMMARY: total %.3f -> %.3f us/use (%+.2f%% throughput), down %.3f -> %.3f us/use (%+.2f%% throughput), reuse %.1f%%; parity PASS\n",
                    c.name,1000*a.wall/a.uses,1000*b.wall/b.uses,100*(a.wall/b.wall-1),
                    1000*a.down/a.uses,1000*b.down/b.uses,100*(a.down/b.down-1),100.0*b.reused/b.uses);
    }
    return 0;
}
