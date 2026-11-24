// Deterministic CPU expert A/B microbenchmarks. No GPU or GGUF required.
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
constexpr int q6_type = (int) Q::Type::Q6_K;
constexpr int ne = 2048, nf = 512;
using Bytes = std::vector<uint8_t>;
struct Weights { Bytes g,u,d; };

Bytes weights(int cols, int rows, std::mt19937& rng, int weight_type=type) {
    const auto* ti = Q::type_info(weight_type);
    Bytes w(Q::row_bytes(weight_type,cols)*(size_t)rows);
    for (auto& v : w) v = (uint8_t)rng();
    for (size_t i=0; i<w.size(); i+=ti->bytes) {
        // Small finite half scales. Randomize magnitudes and signs to exercise correction.
        const uint16_t d = (uint16_t)(0x1400u + rng()%0x600u) | (uint16_t)((rng()&1u)<<15);
        const uint16_t dm = (uint16_t)(0x1000u + rng()%0x600u);
        if (weight_type==q6_type) {
            // Q6_K: ql[128], qh[64], scales[16], fp16 d at byte 208.
            std::memcpy(w.data()+i+ti->bytes-2,&d,2);
        } else {
            std::memcpy(w.data()+i,&d,2);
            std::memcpy(w.data()+i+2,&dm,2);
        }
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

struct Sample { double wall=0, down=0; int64_t uses=0, reused=0; double gate=0; };
struct Case {
    int rows;
    bool mixed;
    const char* name;
    bool sparse = false; // only two of 16 uses share an expert
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
            const int idx=base+(c.sparse ? (row==1 && e==0 ? 0 : 4*row+e) :
                                c.mixed && e>0 ? 1+3*row+(e-1) : e);
            const auto& w=bank[(size_t)idx];
            in.ex.push_back({w.g.data(),w.u.data(),w.d.data(),0.25f,row});
        }
    }
    return in;
}

void run(Workers& pool,const Input& in,int rows,std::vector<float>& out,CpuExpertTiming* timing=nullptr,
         int down_type=type) {
    cpu_experts(pool,in.ex.data(),(int)in.ex.size(),rows,type,type,down_type,ne,nf,in.x.data(),20.f,false,
                out.data(),nullptr,nullptr,nullptr,0.f,0.f,timing);
}

Sample measure(const std::vector<Input>& inputs,const Case& c,int threads,CpuExpertTaskShape shape,int calls,
               int down_type=type) {
    Workers pool(threads,20000,shape);
    std::vector<float> out((size_t)c.rows*ne);
    // Warm each bank; a second worker pool is never alive during measurement.
    for (const auto& in:inputs) run(pool,in,c.rows,out,nullptr,down_type);
    CpuExpertTiming p;
    const auto t=OverlapClock::now();
    for (int i=0;i<calls;++i) run(pool,inputs[(size_t)i%inputs.size()],c.rows,out,&p,down_type);
    const double wall=interval_ms(t,OverlapClock::now());
    return {wall,p.down_reduce_ms,p.expert_uses,p.down_reuse_uses,p.gate_up_activation_ms};
}

int arg(const char* raw,int low,int high) {
    char* end=nullptr;
    const long n=std::strtol(raw,&end,10);
    return end!=raw && *end=='\0' && n>=low && n<=high ? (int)n : -1;
}
// Direct dot parity including valid extreme Q8_K bytes, scale signs,
// non-zero minimum corrections and multiple block counts. Use the same
// AVX2 baseline as production, rather than a generic scalar reference.
bool vnni_dot_check(const Q::DotKernel* base, const Q::DotKernel* opt) {
    const auto* act=Q::act_kernel(Q::Type::Q8_K,Q::Arith::Avx2);
    if (!base || !opt || !act) return false;
    std::mt19937 rng(0xb532u+(unsigned)base->weight);
    for (int n:{256,512,768,1024,2048,2304,4096}) {
        std::vector<float> x((size_t)n);
        const size_t stride=Q::row_bytes((int)Q::Type::Q8_K,n);
        Bytes aq(stride);
        for (int trial=0;trial<256;++trial) {
            const Bytes w=weights(n,1,rng,(int)base->weight);
            for (auto& v:x) v=trial==0 ? 0.f : ((int)(rng()%2049)-1024)/128.f;
            act->fn(x.data(),aq.data(),n);
            if (trial%8==1 || trial%8==2 || trial%8==3) {
                // Q8_K layout: float d, 256 signed bytes, 16 int16 sums.
                // Include -128/+127 (and alternating) without relying on
                // the activation quantizer to produce all edge values.
                for(size_t off=0;off<stride;off+=Q::kBlockQ8_K) {
                    const float d=0.0078125f;
                    std::memcpy(aq.data()+off,&d,4);
                    for(int j=0;j<16;++j) {
                        int16_t sum=0;
                        for(int k=0;k<16;++k) {
                            const int val=trial%8==1 ? -128 : trial%8==2 ? 127 : (k%2 ? -128 : 127);
                            aq[off+4+16*j+k]=(uint8_t)val;
                            sum+=(int16_t)val;
                        }
                        std::memcpy(aq.data()+off+260+2*j,&sum,2);
                    }
                }
            }
            float a=0,b=0;
            base->fn(n,&a,w.data(),aq.data());
            opt->fn(n,&b,w.data(),aq.data());
            if(!std::isfinite(a) || std::memcmp(&a,&b,sizeof(float))!=0) {
                std::fprintf(stderr,"FAIL %s -> %s n=%d trial=%d: %.9g vs %.9g\n",base->name,opt->name,n,trial,a,b);
                return false;
            }
        }
    }
    std::printf("%s -> %s bitwise dot parity PASS\n",base->name,opt->name);
    return true;
}

bool x4_dot_check(const Q::DotKernel* base, const Q::DotX4Kernel* opt) {
    const auto* act=Q::act_kernel(Q::Type::Q8_K,Q::Arith::Avx2);
    if (!base || !opt || !act) return false;
    std::mt19937 rng(0xc945u+(unsigned)base->weight);
    for (int n:{256,512,768,1024,2048,2304,4096}) {
        std::vector<float> x((size_t)n);
        const size_t stride=Q::row_bytes((int)base->weight,n);
        const size_t astride=Q::row_bytes((int)Q::Type::Q8_K,n);
        Bytes aq(astride);
        for (int trial=0;trial<256;++trial) {
            // Exercise packed and padded row strides, with deliberately
            // different weight rows. Pad by 6 to maintain 2-byte alignment.
            const size_t step=stride+(trial%2?6:0);
            Bytes w(4*step);
            for (int r=0;r<4;++r) {
                const Bytes row=weights(n,1,rng,(int)base->weight);
                std::memcpy(w.data()+r*step,row.data(),stride);
            }
            for(auto& v:x) v=trial==0?0.f:((int)(rng()%2049)-1024)/128.f;
            act->fn(x.data(),aq.data(),n);
            if(trial%8==1 || trial%8==2 || trial%8==3) {
                const size_t block=Q::row_bytes((int)Q::Type::Q8_K,256);
                for(size_t off=0;off<astride;off+=block) {
                    const float d=0.0078125f;
                    std::memcpy(aq.data()+off,&d,4);
                    for(int j=0;j<16;++j) {
                        int16_t sum=0;
                        for(int k=0;k<16;++k) {
                            const int val=trial%8==1?-128:trial%8==2?127:(k%2?-128:127);
                            aq[off+4+16*j+k]=(uint8_t)val;
                            sum+=(int16_t)val;
                        }
                        std::memcpy(aq.data()+off+260+2*j,&sum,2);
                    }
                }
            }
            float ref[4], out[4];
            for(int r=0;r<4;++r) base->fn(n,&ref[r],w.data()+r*step,aq.data());
            opt->fn(n,out,w.data(),step,aq.data());
            for(int r=0;r<4;++r) if(!std::isfinite(ref[r]) || std::memcmp(&ref[r],&out[r],sizeof(float))) {
                std::fprintf(stderr,"FAIL %s -> %s n=%d trial=%d row=%d: %.9g vs %.9g\n",
                             base->name,opt->name,n,trial,r,ref[r],out[r]);
                return false;
            }
        }
    }
    std::printf("%s -> %s bitwise x4 parity PASS\n",base->name,opt->name);
    return true;
}

void add_sample(Sample& a, const Sample& b) {
    a.wall+=b.wall; a.gate+=b.gate; a.down+=b.down; a.uses+=b.uses; a.reused+=b.reused;
}

int mixed_bench(bool x4, bool check, int threads, int gate, int down, int calls) {
    const auto* q5=Q::experimental_k_vnni_kernel(Q::Type::Q5_K,Q::default_arith());
    const auto* q6=Q::experimental_k_vnni_kernel(Q::Type::Q6_K,Q::default_arith());
    const auto* b5=Q::dot_kernel(Q::Type::Q5_K,Q::default_arith());
    const auto* b6=Q::dot_kernel(Q::Type::Q6_K,Q::default_arith());
    const auto* x5=Q::experimental_k_x4_kernel(Q::Type::Q5_K,Q::default_arith());
    const auto* x6=Q::experimental_k_x4_kernel(Q::Type::Q6_K,Q::default_arith());
    if ((x4?(!x5 || !x6):(!q5 || !q6)) || !b5 || !b6 || b5->isa!=Q::Isa::Avx2 || b6->isa!=Q::Isa::Avx2) {
        std::printf("%s experiment requires both AVX2 baseline kernels and candidate kernels; %s\n",x4?"AVX2 x4":"AVX-VNNI",check?"SKIP":"unavailable");
        return check?0:2;
    }
    if(x4 ? (!x4_dot_check(b5,x5) || !x4_dot_check(b6,x6))
          : (!vnni_dot_check(b5,q5) || !vnni_dot_check(b6,q6))) return 1;
    CpuExpertTaskShape shapes[4];
    for(int mode=0;mode<4;++mode) {
        shapes[mode].gate_max_rows=gate;
        shapes[mode].down_rows=down;
        if(x4) { shapes[mode].q5_x4=(mode&1)!=0; shapes[mode].q6_x4=(mode&2)!=0; }
        else { shapes[mode].q5_vnni=(mode&1)!=0; shapes[mode].q6_vnni=(mode&2)!=0; }
    }
    const char* names[]={"baseline","q5-only","q6-only","both"};
    if(!shapes[0].valid() || threads<1 || calls<1) return 2;
    std::string err;
    if(!cpu_expert_types_ok(type,q6_type,ne,nf,err)) { std::fprintf(stderr,"%s\n",err.c_str()); return 1; }
    const int banks=check?1:8, per_bank=16;
    std::mt19937 rng(0x619u);
    std::vector<Weights> bank;
    bank.reserve((size_t)banks*per_bank);
    for(int i=0;i<banks*per_bank;++i)
        bank.push_back({weights(ne,nf,rng),weights(ne,nf,rng),weights(nf,ne,rng,q6_type)});
    const Case work[]={{1,false,"single"},{4,false,"sparse-4",true},{4,true,"mixed-4"},
                       {2,false,"shared-2"},{8,false,"shared-8"}};
    std::printf("Q5_K/Q5_K/Q6_K CPU expert bench [%s]: threads=%d gate=%d down=%d calls=%d banks=%d dims=%d/%d\n",
                x4?"AVX2 x4":"AVX-VNNI",threads,gate,down,calls,banks,ne,nf);
    for(const auto& c:work) {
        std::vector<Input> inputs;
        for(int bank_id=0;bank_id<banks;++bank_id) inputs.push_back(input_for(bank,bank_id*per_bank,c,rng));
        std::vector<std::vector<float>> refs(inputs.size(),std::vector<float>((size_t)c.rows*ne));
        { Workers pool(threads,20000,shapes[0]);
          for(size_t i=0;i<inputs.size();++i) run(pool,inputs[i],c.rows,refs[i],nullptr,q6_type); }
        for(int mode=1;mode<4;++mode) {
            Workers pool(threads,20000,shapes[mode]);
            std::vector<float> out((size_t)c.rows*ne);
            for(size_t i=0;i<inputs.size();++i) {
                CpuExpertTiming p;
                run(pool,inputs[i],c.rows,out,&p,q6_type);
                const char* expected5=shapes[mode].q5_vnni?q5->name:b5->name;
                const char* expected6=shapes[mode].q6_vnni?q6->name:b6->name;
                const char* expected_x5=shapes[mode].q5_x4?x5->name:"none";
                const char* expected_x6=shapes[mode].q6_x4?x6->name:"none";
                if(!equal(refs[i],out) || p.quant_mixes.size()!=1 ||
                   std::strcmp(p.quant_mixes[0].gate.x4,expected_x5)!=0 ||
                   std::strcmp(p.quant_mixes[0].up.x4,expected_x5)!=0 ||
                   std::strcmp(p.quant_mixes[0].down.x4,expected_x6)!=0 ||
                   std::strcmp(p.quant_mixes[0].gate.dot,expected5)!=0 ||
                   std::strcmp(p.quant_mixes[0].up.dot,expected5)!=0 ||
                   std::strcmp(p.quant_mixes[0].down.dot,expected6)!=0) {
                    std::fprintf(stderr,"FAIL mixed expert parity/dispatch: %s %s bank=%zu\n",c.name,names[mode],i);
                    return 1;
                }
            }
        }
        std::printf("%s mixed expert parity and dispatch PASS (all modes)\n",c.name);
        if(check) continue;
        // Paired baseline and each candidate, alternating order. Separate
        // baselines reduce drift when a VM or the hybrid CPU changes load.
        for(int mode=1;mode<4;++mode) {
            Sample a,b;
            for(int round=0;round<3;++round) {
                Sample aa,bb;
                if(round%2==0) {
                    aa=measure(inputs,c,threads,shapes[0],calls,q6_type);
                    bb=measure(inputs,c,threads,shapes[mode],calls,q6_type);
                } else {
                    bb=measure(inputs,c,threads,shapes[mode],calls,q6_type);
                    aa=measure(inputs,c,threads,shapes[0],calls,q6_type);
                }
                add_sample(a,aa); add_sample(b,bb);
                std::printf("%s %s round %d: total %.2f -> %.2f ms, gate/up %.2f -> %.2f ms, down %.2f -> %.2f ms\n",
                            c.name,names[mode],round+1,aa.wall,bb.wall,aa.gate,bb.gate,aa.down,bb.down);
            }
            std::printf("%s %s SUMMARY: total %.3f -> %.3f us/use (%+.2f%% throughput), "
                        "gate/up %.3f -> %.3f us/use (%+.2f%%), down %.3f -> %.3f us/use (%+.2f%%); parity PASS\n",
                        c.name,names[mode],1000*a.wall/a.uses,1000*b.wall/b.uses,100*(a.wall/b.wall-1),
                        1000*a.gate/a.uses,1000*b.gate/b.uses,100*(a.gate/b.gate-1),
                        1000*a.down/a.uses,1000*b.down/b.uses,100*(a.down/b.down-1));
        }
    }
    return 0;
}
} // namespace

int main(int argc,char** argv) {
    if (argc>=2 && (std::strcmp(argv[1],"--vnni")==0 || std::strcmp(argv[1],"--vnni-check")==0 ||
                    std::strcmp(argv[1],"--x4")==0 || std::strcmp(argv[1],"--x4-check")==0)) {
        const bool x4=std::strcmp(argv[1],"--x4")==0 || std::strcmp(argv[1],"--x4-check")==0;
        const bool check=std::strcmp(argv[1],"--vnni-check")==0 || std::strcmp(argv[1],"--x4-check")==0;
        if(check) return argc==2?mixed_bench(x4,true,4,8,16,100):2;
        if(argc!=2 && argc!=6) {
            std::fprintf(stderr,"usage: poly_cpu_expert_bench %s [threads gate_rows down_rows calls]\n",x4?"--x4":"--vnni");
            return 2;
        }
        const int threads=argc==6?arg(argv[2],1,64):12;
        const int gate=argc==6?arg(argv[3],4,256):8;
        const int down=argc==6?arg(argv[4],4,256):16;
        const int calls=argc==6?arg(argv[5],8,10000):1000;
        if(threads<1 || calls<1 || !CpuExpertTaskShape::valid_rows(gate) || !CpuExpertTaskShape::valid_rows(down)) return 2;
        return mixed_bench(x4,false,threads,gate,down,calls);
    }
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
