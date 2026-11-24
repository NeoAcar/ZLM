// Batch-one CUDA forward for the ZLM checkpoint.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <fstream>
#include <vector>
namespace zlm {
using B = __nv_bfloat16;
constexpr int H=1024, F=3072, KV=256, V=49152, L=38, D=64;
constexpr size_t layer_size=2*H + size_t(H)*(H+2*KV+H+3*F);
inline void check(cudaError_t result) {
    if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Buffer {
    B* data;
    explicit Buffer(size_t count) { check(cudaMalloc(&data,count*sizeof(B))); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&)=delete;
};
__device__ float warp_sum(float x) {
    for(int offset=16;offset;offset/=2) x+=__shfl_down_sync(0xffffffff,x,offset);
    return x;
}
// Eight warps or fewer/more depending on the projection tile.
constexpr int lanes_per_row=16;
constexpr int warps_per_block=8;
constexpr int rows_per_warp=2;
template<int threads>
__device__ void normalize_input(const B* x,const B* weight,B* normalized){
    __shared__ float scratch[threads/32];
    constexpr int width=H/threads;
    static_assert(width==4||width==8);
    using Vector=std::conditional_t<width==8,uint4,uint2>;
    Vector values=reinterpret_cast<const Vector*>(x)[threadIdx.x];
    Vector weights=reinterpret_cast<const Vector*>(weight)[threadIdx.x];
    auto inputs=reinterpret_cast<const B*>(&values);
    float sum=0.f;
    #pragma unroll
    for(int j=0;j<width;++j)sum=fmaf(float(inputs[j]),float(inputs[j]),sum);
    sum=warp_sum(sum);
    if(threadIdx.x%32==0)scratch[threadIdx.x/32]=sum;
    __syncthreads();
    if(threadIdx.x<32){
        float total=warp_sum(threadIdx.x<threads/32?scratch[threadIdx.x]:0.f);
        if(threadIdx.x==0)scratch[0]=rsqrtf(total/H+1e-6f);
    }
    __syncthreads();
    float scale=scratch[0];
    Vector output;
    #pragma unroll
    for(int j=0;j<width;++j)reinterpret_cast<B*>(&output)[j]=B(float(inputs[j])*scale*float(reinterpret_cast<const B*>(&weights)[j]));
    reinterpret_cast<Vector*>(normalized)[threadIdx.x]=output;
    __syncthreads();
}
// 16-position K pages are stored in the register-fragment order of QK MMA.
__device__ int key_offset(int pos,int kv_col){
    int head=kv_col/D,dim=kv_col%D;
    int lane=(pos%8)*4+(dim%8)/2;
    int tile=((pos/16)*(KV/D)+head)*(D/16)+dim/16;
    return (tile*32+lane)*8+(dim%16/8)*4+(pos%16/8)*2;
}
// Preserve all BF16 bits. A tile outside the local exponent window uses its original fragments.
constexpr int weight_page=32;
struct GateCode {unsigned char base,raw_before;};
struct GatePayload {unsigned low[64],nibbles[32];};
__global__ void encode_gate(uint4* weights,GateCode* codes){
    int lane=threadIdx.x%32,tile=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    uint4 input=weights[tile*32+lane];
    auto* payload=reinterpret_cast<GatePayload*>(weights+tile*32);
    auto bits=reinterpret_cast<const unsigned short*>(&input);
    int lo=127,hi=0;
    #pragma unroll
    for(int i=0;i<8;++i){int e=(bits[i]>>8)&127;lo=min(lo,e);hi=max(hi,e);}
    for(int d=16;d;d/=2){
        lo=min(lo,__shfl_xor_sync(0xffffffff,lo,d));
        hi=max(hi,__shfl_xor_sync(0xffffffff,hi,d));
    }
    bool fits=hi-lo<=7;
    lo=max(0,hi-7);
    if(lane==0)codes[tile].base=fits?lo:255;
    if(!fits)return;
    unsigned nib=0,low[2]={};
    #pragma unroll
    for(int i=0;i<8;++i){
        unsigned val=bits[i];
        unsigned code=(((val>>8)&127)-lo)|((val>>12)&8);
        nib|=code<<(8*(i%4)+4*(i/4));
        low[i/4]|=(val&255)<<(8*(i%4));
    }
    payload->low[lane]=low[0];payload->low[lane+32]=low[1];
    payload->nibbles[lane]=nib;
}
// Pack each page towards its beginning. All fallback tiles remain bit-exact.
__global__ void compact_page(const uint4* input,uint4* output,GateCode* codes){
    int lane=threadIdx.x%32,tile=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    int page=tile/weight_page*weight_page,slot=tile%weight_page;
    unsigned mask=__ballot_sync(0xffffffff,lane<weight_page&&codes[page+lane].base==255);
    int before=__popc(mask&((1u<<slot)-1));
    if(lane==0)codes[tile].raw_before=before;
    int vectors=codes[tile].base==255?32:24;
    if(lane<vectors)output[page*32+slot*24+before*8+lane]=input[tile*32+lane];
}
struct GatePage {unsigned raw_mask,base;};
__global__ void describe_gate_pages(const GateCode* codes,GatePage* pages){
    int tile=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%32;
    unsigned base=codes[tile].base;
    unsigned mask=__ballot_sync(0xffffffff,base==255);
    unsigned hi=base==255?0:base,lo=base;
    for(int d=16;d;d/=2){
        hi=max(hi,__shfl_down_sync(0xffffffff,hi,d));
        lo=min(lo,__shfl_down_sync(0xffffffff,lo,d));
    }
    if(!lane)pages[tile/32]={mask,lo==hi?lo:255u};
}
__device__ uint4 expand_payload(const uint4* start,unsigned base,int lane){
    auto* payload=reinterpret_cast<const GatePayload*>(start);
    unsigned low[2]={__ldcs(&payload->low[lane]),__ldcs(&payload->low[32+lane])};
    unsigned nib=__ldcs(&payload->nibbles[lane]);
    unsigned common=base*0x01010101u;
    uint4 result;
    auto words=reinterpret_cast<unsigned*>(&result);
    #pragma unroll
    for(int i=0;i<2;++i){
        unsigned codes4=nib>>(4*i);
        unsigned high=((codes4&0x08080808u)<<4)|((codes4&0x07070707u)+common);
        words[2*i]=__byte_perm(low[i],high,0x5140);
        words[2*i+1]=__byte_perm(low[i],high,0x7362);
    }
    return result;
}
__device__ uint4 read_gate(const uint4* raw,const GateCode* codes,int tile,int lane){
    GateCode code=codes[tile];
    int slot=tile%weight_page;
    const uint4* start=raw+(tile-slot)*32+slot*24+code.raw_before*8;
    unsigned base=code.base;
    if(base==255)return __ldcs(start+lane);
    return expand_payload(start,base,lane);
}
struct CodedMatrix {const GateCode* codes;const unsigned char* bases;};
constexpr int attention_tile = 64;
enum class Projection {plain, gate, rotary, attended};
template<int cols, Projection kind=Projection::plain,int lanes=lanes_per_row,bool coded=false>
__global__ void matvec(const B* w,const B* x,B* y,const B* residual,const float2* phase=nullptr,B* keys=nullptr,B* values=nullptr,const int* pos=nullptr,const B* norm_weight=nullptr,CodedMatrix compressed={},const float* partial=nullptr,int tiles=0) {
    if constexpr(kind==Projection::attended){
        static_assert(warps_per_block==8);
        __shared__ __align__(16) B merged[H];
        int head=threadIdx.x/16,lane16=threadIdx.x%16;
        int valid=*pos/attention_tile+1;
        const float* first=partial+head*tiles*(D+2);
        float peak=-INFINITY;
        for(int i=lane16;i<valid;i+=16)peak=fmaxf(peak,first[i*(D+2)+D]);
        for(int d=8;d;d/=2)peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,d,16));
        float sum[4]={},den=0;
        for(int i=0;i<valid;++i){
            const float* part=first+i*(D+2);
            float factor=expf(part[D]-peak);
            den=fmaf(part[D+1],factor,den);
            #pragma unroll
            for(int j=0;j<4;++j)sum[j]=fmaf(part[lane16+j*16],factor,sum[j]);
        }
        for(int j=0;j<4;++j)merged[head*D+lane16+j*16]=B(sum[j]/den);
        __syncthreads();
        x=merged;
    }
    if constexpr(kind==Projection::rotary){
        __shared__ __align__(16) B normalized[H];
        normalize_input<32*warps_per_block>(x,norm_weight,normalized);
        x=normalized;
    }
    int row=((blockIdx.x*blockDim.x+threadIdx.x)/lanes)*rows_per_warp;
    int lane=threadIdx.x%lanes;
    float sums[rows_per_warp]={};
    for(int i=lane*8;i<cols;i+=lanes*8) {
        uint4 input=*reinterpret_cast<const uint4*>(x+i);
        #pragma unroll
        for(int r=0;r<rows_per_warp;++r){
            uint4 packed;
            if constexpr(coded){
                int vector=((row+r)*cols+i)/8;
                packed=read_gate(reinterpret_cast<const uint4*>(w),compressed.codes,vector/32,vector%32);
            }else packed=__ldcs(reinterpret_cast<const uint4*>(w+(kind==Projection::gate ? (row/2+r*F) : row+r)*cols+i));
            #pragma unroll
            for(int j=0;j<8;++j)
                sums[r]=fmaf(float(reinterpret_cast<const B*>(&packed)[j]),
                             float(reinterpret_cast<const B*>(&input)[j]),sums[r]);
        }
    }
    B gate_value{};
    B even{};
    #pragma unroll
    for(int r=0;r<rows_per_warp;++r){
        float sum=sums[r];
        for(int d=lanes/2;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d,lanes);
        if(!lane){
            B projected=B(sum);
            if constexpr(kind==Projection::rotary){
                static_assert(rows_per_warp==2);
                if(row>=H+KV)values[(*pos)*KV+row-H-KV+r]=projected;
                else if(r==0)even=projected;
                else{
                    float2 cs=phase[(*pos)*(D/2)+(row%D)/2];
                    B* dest=row<H?y+row:keys+key_offset(*pos,row-H);
                    dest[0]=B(float(even)*cs.x-float(projected)*cs.y);
                    dest[1]=B(float(even)*cs.y+float(projected)*cs.x);
                }
                continue;
            }
            if constexpr(kind==Projection::gate){
                static_assert(rows_per_warp==2);
                if(r==0)gate_value=projected;
                else {
                    float g=float(gate_value);
                    y[row/2]=B(float(B(g/(1.f+expf(-g))))*float(projected));
                }
            }else y[row+r]=residual?B(float(residual[row+r])+float(projected)):projected;
        }
    }
}
inline void linear(const B* w,const B* x,B* y,int rows,int cols,cudaStream_t stream,const B* residual=nullptr) {
    if(cols==H)matvec<H><<<rows/(warps_per_block*(32/lanes_per_row)*rows_per_warp),32*warps_per_block,0,stream>>>(w,x,y,residual);
    else matvec<F,Projection::plain,32><<<rows/(warps_per_block*(32/32)*rows_per_warp),32*warps_per_block,0,stream>>>(w,x,y,residual);
}
__global__ void pack_mlp_fragments(const B* weights,uint4* packed,int cols){
    int index=blockIdx.x*blockDim.x+threadIdx.x,lane=index%32;
    int tile=index/32,group=tile/(cols/16),k=(tile%(cols/16))*16;
    const B* row=weights+(group*8+lane/4)*cols+k+2*(lane%4);
    packed[index]=make_uint4(*reinterpret_cast<const unsigned*>(row),
        *reinterpret_cast<const unsigned*>(row+F*cols),*reinterpret_cast<const unsigned*>(row+8),
        *reinterpret_cast<const unsigned*>(row+F*cols+8));
}
// Each warp applies a 16-row matrix tile to the same activation column.
template<int cols>
__global__ void tensor_gate(const B* weights,const B* input,B* output,const B* norm_weight,const GateCode* codes,const GatePage* pages){
    __shared__ __align__(16) B normalized[H];
    normalize_input<128>(input,norm_weight,normalized);
    input=normalized;
    const int lane=threadIdx.x%32,warp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int row=warp*16+lane/4,kpair=2*(lane%4);
    float d0=0,d1=0,d2=0,d3=0;
    for(int page_step=0;page_step<cols;page_step+=32*16){
        int first=warp*(cols/16)+page_step/16;
        GatePage page=pages[first/32];
        for(int slot=0;slot<32;++slot){
            int k=page_step+slot*16;
            const uint4* raw=reinterpret_cast<const uint4*>(weights);
            uint4 fragment;
            if(page.base==255)fragment=read_gate(raw,codes,first+slot,lane);
            else{
                int before=__popc(page.raw_mask&((1u<<slot)-1));
                const uint4* payload=raw+first*32+slot*24+before*8;
                fragment=(page.raw_mask&(1u<<slot))?__ldcs(payload+lane):expand_payload(payload,page.base,lane);
            }
        unsigned a0=fragment.x,a1=fragment.y,a2=fragment.z,a3=fragment.w;
        unsigned b0=*reinterpret_cast<const unsigned*>(input+k+kpair);
        unsigned b1=*reinterpret_cast<const unsigned*>(input+k+kpair+8);
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    }
    if(lane%4==0){
        float gate=float(B(d0)),up=float(B(d2));
        output[warp*8+lane/4]=B(float(B(gate/(1.f+expf(-gate))))*up);
    }
}
__global__ void norm(const B* x,const B* w,B* y) {
    __shared__ float sums[8];
    int t=threadIdx.x,lane=t%32;
    float sum=0;
    for(int i=t;i<H;i+=256)sum=fmaf(float(x[i]),float(x[i]),sum);
    sum=warp_sum(sum);
    if(lane==0)sums[t/32]=sum;
    __syncthreads();
    if(t<32){
        float total=warp_sum(t<8?sums[t]:0.f);
        if(t==0)sums[0]=rsqrtf(total/H+1e-6f);
    }
    __syncthreads();
    float scale=sums[0];
    for(int i=t;i<H;i+=256)y[i]=B(float(x[i])*scale*float(w[i]));
}
__global__ void rotate_store(const float2* phase,B* qkv,B* keys,B* values,const int* position) {
    int pos=*position;
    int i=2*(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=H+KV)return;
    float2 rotation=phase[pos*(D/2)+(i%D)/2];
    float c=rotation.x,s=rotation.y;
    float a=float(qkv[i]),b=float(qkv[i+1]);
    B* dest=i<H?qkv+i:keys+pos*KV+i-H;
    dest[0]=B(a*c-b*s);dest[1]=B(a*s+b*c);
    if(i>=H){
        int j=i-H;
        values[pos*KV+j]=qkv[H+KV+j];values[pos*KV+j+1]=qkv[H+KV+j+1];
    }
}

constexpr int query_group=2;
// Compute the four GQA queries against sixteen adjacent KV positions per MMA.
__global__ void attention_parts(const B* q,const B* keys,const B* values,float* partial,const int* position){
    static_assert(attention_tile==64);
    __shared__ float scores[4][attention_tile],reductions[4][2];
    int t=threadIdx.x,lane=t%32,warp=t/32,group=blockIdx.y;
    int first=blockIdx.x*attention_tile,count=min(attention_tile,*position-first+1);
    if(count<=0)return;
    int base=first*KV+group*D;
    if(warp<4&&warp*16<count){
        int row=warp*16+lane/4,pair=(lane%4)*2;
        const B* query=q+(group*4+(lane/4)%4)*D;
        float d0=0,d1=0,d2=0,d3=0;
        #pragma unroll
        for(int k=0;k<D;k+=16){
            int tile=((first/16+warp)*(KV/D)+group)*(D/16)+k/16;
            uint4 frag=reinterpret_cast<const uint4*>(keys)[tile*32+lane];
            unsigned a0=frag.x,a1=frag.y,a2=frag.z,a3=frag.w;
            unsigned b0=*reinterpret_cast<const unsigned*>(query+k+pair);
            unsigned b1=*reinterpret_cast<const unsigned*>(query+k+pair+8);
            asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
        if(lane%4<2){
            int col=(lane%4)*2;
            scores[col][row]=d0*.125f;scores[col+1][row]=d1*.125f;
            scores[col][row+8]=d2*.125f;scores[col+1][row+8]=d3*.125f;
        }
    }
    __syncthreads();
    int h=t/64,dim=t%64;
    float score=dim<count?scores[h][dim]:-INFINITY,peak=score;
    for(int d=16;d;d/=2)peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,d));
    if(lane==0)reductions[h][dim/32]=peak;
    __syncthreads();
    peak=fmaxf(reductions[h][0],reductions[h][1]);
    float mass=expf(score-peak);
    scores[h][dim]=mass;
    float den=warp_sum(mass);
    __syncthreads();
    if(lane==0)reductions[h][dim/32]=den;
    __syncthreads();
    float* part=partial+((group*4+h)*gridDim.x+blockIdx.x)*(D+2);
    if(dim==0){part[D]=peak;part[D+1]=reductions[h][0]+reductions[h][1];}
    float out=0;
    for(int i=0;i<count;++i)out=fmaf(scores[h][i],float(values[base+i*KV+dim]),out);
    part[dim]=out;
}
__global__ void combine_attention(const float* partial,B* out,const int* position,int tiles){
    int head=blockIdx.x,dim=threadIdx.x;
    int valid=*position/attention_tile+1;
    const float* first=partial+head*tiles*(D+2);
    extern __shared__ float factors[];
    __shared__ float common_peak,denominator;
    if(dim<32){
        float peak=-INFINITY;
        for(int i=dim;i<valid;i+=32)peak=fmaxf(peak,first[i*(D+2)+D]);
        for(int d=16;d;d/=2)peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,d));
        if(dim==0)common_peak=peak;
    }
    __syncthreads();
    if(dim<32){
        float den=0;
        for(int i=dim;i<valid;i+=32){
            const float* part=first+i*(D+2);
            float factor=expf(part[D]-common_peak);
            factors[i]=factor;
            den=fmaf(part[D+1],factor,den);
        }
        den=warp_sum(den);
        if(dim==0)denominator=den;
    }
    __syncthreads();
    float sum=0;
    for(int i=0;i<valid;++i)sum=fmaf(first[i*(D+2)+dim],factors[i],sum);
    out[head*D+dim]=B(sum/denominator);
}
__global__ void swiglu(B* pair) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    float g=float(pair[i]);
    pair[i]=B(float(B(g/(1.f+expf(-g))))*float(pair[F+i]));
}
__global__ void begin_token(const B* embedding,B* x,int* position,int token,int pos,const GateCode* codes) {
    if(threadIdx.x==0)*position=pos;
    for(int i=threadIdx.x;i<H/8;i+=blockDim.x){
        int index=token*(H/8)+i;
        reinterpret_cast<uint4*>(x)[i]=read_gate(reinterpret_cast<const uint4*>(embedding),codes,index/32,index%32);
    }
}
class Decoder {
    cudaStream_t stream{};
    cudaGraphExec_t body{};
    int* position{};
    static int checked_capacity(int cap,bool yarn){
        if(cap<1||cap>(yarn?16384:4096))
            throw std::invalid_argument("Cache capacity exceeds the active RoPE context");
        return cap;
    }
    int capacity;
    Buffer rotary;
    Buffer weights,keys,values,work;
    static constexpr int gate_tiles=(2*F*H)/256;
    Buffer gate_pages{size_t(L)*gate_tiles/32*sizeof(GatePage)/sizeof(B)};
    Buffer coded_gate{size_t(L)*gate_tiles*sizeof(GateCode)/sizeof(B)};
    static constexpr int down_tiles=(H*F)/256;
    Buffer down_codes{size_t(L)*down_tiles*sizeof(GateCode)/sizeof(B)};
    static constexpr int vocab_tiles=(V*H)/256;
    Buffer vocab_codes{size_t(1)*vocab_tiles*sizeof(GateCode)/sizeof(B)};
    Buffer parts;
public:
    int length=0;
    Decoder(const char* filename,int cap,bool yarn=false):capacity(checked_capacity(cap,yarn)),rotary(size_t(cap)*D*2),weights(size_t(V)*H+H+L*layer_size),
        keys(size_t(L)*KV*((cap+15)/16)*16),values(size_t(L)*KV*cap),work(4*H+2*KV+2*F+V),
        parts(2*(H/D)*(D+2)*((cap+attention_tile-1)/attention_tile)) {
        std::ifstream file(filename,std::ios::binary|std::ios::ate);
        size_t bytes=(size_t(V)*H+H+L*layer_size)*sizeof(B);
        if(file.tellg()!=std::streamoff(bytes))throw std::runtime_error("Invalid packed weights");
        file.seekg(0);
        std::vector<char> chunk(1<<23);
        for(size_t n=0;n<bytes;){
            size_t count=std::min(bytes-n,chunk.size());
            if(!file.read(chunk.data(),count))throw std::runtime_error("Weight read failed");
            check(cudaMemcpy(reinterpret_cast<char*>(weights.data)+n,chunk.data(),count,cudaMemcpyHostToDevice));n+=count;
        }
        std::vector<float2> phase(size_t(cap)*D/2);
        const float scale=yarn?1.f+0.1f*std::log(4.f):1.f;
        const double tau=2.*std::acos(-1.);
        const double low=std::floor(D*std::log(4096./(32.*tau))/(2.*std::log(100000.)));
        const double high=std::ceil(D*std::log(4096./tau)/(2.*std::log(100000.)));
        for(int j=0;j<D/2;++j){
            float frequency=std::pow(100000.f,-2.f*j/D);
            if(yarn){
                float ramp=std::fmin(1.f,std::fmax(0.f,float((j-low)/(high-low))));
                frequency=frequency*(1.f-ramp)+(frequency/4.f)*ramp;
            }
            for(int p=0;p<cap;++p)
                phase[size_t(p)*D/2+j]=make_float2(scale*std::cos(p*frequency),scale*std::sin(p*frequency));
        }
        check(cudaMemcpy(rotary.data,phase.data(),phase.size()*sizeof(float2),cudaMemcpyHostToDevice));
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaMalloc(&position,sizeof(int)));
        check(cudaMemsetAsync(keys.data,0,size_t(L)*KV*((cap+15)/16)*16*sizeof(B),stream));
        auto compact=[&](B* matrix,GateCode* codes,int tiles,Buffer& staging){
            compact_page<<<tiles/4,128,0,stream>>>(reinterpret_cast<uint4*>(matrix),reinterpret_cast<uint4*>(staging.data),codes);
            check(cudaMemcpyAsync(matrix,staging.data,size_t(tiles)*256*sizeof(B),cudaMemcpyDeviceToDevice,stream));
        };
        encode_gate<<<vocab_tiles/4,128,0,stream>>>(reinterpret_cast<uint4*>(weights.data),reinterpret_cast<GateCode*>(vocab_codes.data));
            { Buffer staging(size_t(V)*H); compact(weights.data,reinterpret_cast<GateCode*>(vocab_codes.data),vocab_tiles,staging); check(cudaStreamSynchronize(stream)); }
        Buffer tensor_staging(2*F*H);
        for(int layer=0;layer<L;++layer){
            B* gate=weights.data+size_t(V)*H+H+layer*layer_size+2*H+(H+2*KV)*H+H*H;
            encode_gate<<<down_tiles/4,128,0,stream>>>(reinterpret_cast<uint4*>(gate+2*F*H),reinterpret_cast<GateCode*>(down_codes.data)+size_t(layer)*down_tiles);
            compact(gate+2*F*H,reinterpret_cast<GateCode*>(down_codes.data)+size_t(layer)*down_tiles,down_tiles,tensor_staging);
            pack_mlp_fragments<<<2*F*H/(8*256),256,0,stream>>>(gate,reinterpret_cast<uint4*>(tensor_staging.data),H);
            check(cudaMemcpyAsync(gate,tensor_staging.data,2*F*H*sizeof(B),cudaMemcpyDeviceToDevice,stream));
            encode_gate<<<gate_tiles/4,128,0,stream>>>(reinterpret_cast<uint4*>(gate),reinterpret_cast<GateCode*>(coded_gate.data)+layer*gate_tiles);
            compact(gate,reinterpret_cast<GateCode*>(coded_gate.data)+size_t(layer)*gate_tiles,gate_tiles,tensor_staging);
            describe_gate_pages<<<gate_tiles/128,128,0,stream>>>(reinterpret_cast<const GateCode*>(coded_gate.data)+size_t(layer)*gate_tiles,reinterpret_cast<GatePage*>(gate_pages.data)+size_t(layer)*gate_tiles/32);
        }
        check(cudaStreamSynchronize(stream));
    }
    ~Decoder(){
        if(body)cudaGraphExecDestroy(body);
        cudaFree(position);
        cudaStreamDestroy(stream);
    }
    cudaStream_t cuda_stream() const {return stream;}
    B* forward(int token,bool logits=true) {
        if(length>=capacity||token<0||token>=V)throw std::runtime_error("Invalid decode input");
        B *x=work.data,*xn=x+H,*q=xn+H,*a=q+H+2*KV,*pair=a+H,*output=pair+2*F;
        B *final_norm=weights.data+size_t(V)*H,*w=final_norm+H;
        begin_token<<<1,256,0,stream>>>(weights.data,x,position,token,length,reinterpret_cast<const GateCode*>(vocab_codes.data));
        if(!body){
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int layer=0;layer<L;++layer){
            B *k=keys.data+size_t(layer)*((capacity+15)/16)*16*KV,*v=values.data+size_t(layer)*capacity*KV;
            w+=H;
            matvec<H,Projection::rotary><<<(H+2*KV)/(warps_per_block*(32/lanes_per_row)*rows_per_warp),32*warps_per_block,0,stream>>>(w,x,q,nullptr,reinterpret_cast<const float2*>(rotary.data),k,v,position,w-H);w+=(H+2*KV)*H;
            int tiles=(capacity+attention_tile-1)/attention_tile;
            auto* partial=reinterpret_cast<float*>(parts.data);
            attention_parts<<<dim3(tiles,KV/D),256,0,stream>>>(q,k,v,partial,position);
            matvec<H,Projection::attended><<<H/(warps_per_block*(32/lanes_per_row)*rows_per_warp),32*warps_per_block,0,stream>>>(w,a,x,x,nullptr,nullptr,nullptr,position,nullptr,{},partial,tiles);w+=H*H;
            w+=H;
            tensor_gate<H><<<2*F/64,128,0,stream>>>(w,x,pair,w-H,reinterpret_cast<const GateCode*>(coded_gate.data)+layer*gate_tiles,reinterpret_cast<const GatePage*>(gate_pages.data)+size_t(layer)*gate_tiles/32);w+=2*F*H;
            matvec<F,Projection::plain,32,true><<<H/(warps_per_block*(32/32)*rows_per_warp),32*warps_per_block,0,stream>>>(w,pair,x,x,nullptr,nullptr,nullptr,nullptr,nullptr,{reinterpret_cast<GateCode*>(down_codes.data)+size_t(layer)*down_tiles,nullptr});w+=H*F;
        }
        cudaGraph_t graph;
        check(cudaStreamEndCapture(stream,&graph));
        check(cudaGraphInstantiate(&body,graph,0));
        check(cudaGraphDestroy(graph));
        }
        check(cudaGraphLaunch(body,stream));
        if(logits){norm<<<1,256,0,stream>>>(x,final_norm,xn);matvec<H,Projection::plain,lanes_per_row,true><<<V/(warps_per_block*(32/lanes_per_row)*rows_per_warp),32*warps_per_block,0,stream>>>(weights.data,xn,output,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,{reinterpret_cast<GateCode*>(vocab_codes.data),nullptr});}
        ++length;
        return logits?output:nullptr;
    }
};
}
