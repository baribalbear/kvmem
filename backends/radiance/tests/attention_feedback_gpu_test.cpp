#include "attention_kernel.hpp"
#include "kvmem/page_softmax_cuda.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "rad_device.h"
#include "rad_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <tuple>
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(false)
struct Buffer {
    void* p; int kind;
    Buffer(size_t n,int k):p(rad_dev_alloc(n,k)),kind(k){CHECK(p);}
    ~Buffer(){rad_dev_free(p,kind);}
};
static void exercise(uint32_t dtype,RadStream stream) {
    constexpr unsigned dim=8,qheads=4,kvheads=2,native_block=4,blocks=3,nrows=3;
    const unsigned bytes=dtype==RAD_F32?4:dtype==RAD_F8E4M3?1:2;
    const size_t elements=blocks*kvheads*native_block*2*dim;
    Buffer kh(elements*bytes,RAD_MEM_HOST_PINNED),kd(elements*bytes,RAD_MEM_DEVICE);
    std::vector<float> kv(elements);
    for(size_t i=0;i<elements;++i) {
        const float v=float(int(i%7)-3)*.25f;kv[i]=v;
        if(dtype==RAD_F32)static_cast<float*>(kh.p)[i]=v;
        else if(dtype==RAD_F16)static_cast<uint16_t*>(kh.p)[i]=rad_f32_to_f16(v);
        else if(dtype==RAD_BF16)static_cast<uint16_t*>(kh.p)[i]=rad_f32_to_bf16(v);
        else {
            const unsigned codes[]{0,40,48,52};
            static_cast<uint8_t*>(kh.p)[i]=uint8_t(codes[unsigned(std::abs(v)*4)]|(v<0?128:0));
        }
    }
    Buffer qh(nrows*qheads*dim*4,RAD_MEM_HOST_PINNED),qd(nrows*qheads*dim*4,RAD_MEM_DEVICE);
    auto* q=static_cast<float*>(qh.p);
    for(unsigned i=0;i<nrows*qheads*dim;++i)q[i]=float(int(i%11)-5)*.125f;
    Buffer th(blocks*4,RAD_MEM_HOST_PINNED),td(blocks*4,RAD_MEM_DEVICE);
    const int32_t table[]{2,0,1};std::memcpy(th.p,table,sizeof(table));
    Buffer ph(3*sizeof(AttentionPage),RAD_MEM_HOST_PINNED),pd(3*sizeof(AttentionPage),RAD_MEM_DEVICE);
    const AttentionPage pages[]{{0,4,0},{40,4,4},{100,3,8}};std::memcpy(ph.p,pages,sizeof(pages));
    Buffer rh(12,RAD_MEM_HOST_PINNED),rd(12,RAD_MEM_DEVICE);
    const uint32_t rows[]{0,1,2};std::memcpy(rh.p,rows,sizeof(rows));
    Buffer scratch(nrows*qheads*3*4,RAD_MEM_DEVICE),output(nrows*3*4,RAD_MEM_DEVICE),host(nrows*3*4,RAD_MEM_HOST_PINNED);
    for(auto tuple:std::vector<std::tuple<void*,const void*,size_t>>{{kd.p,kh.p,elements*bytes},{qd.p,qh.p,nrows*qheads*dim*4},
        {td.p,th.p,sizeof(table)},{pd.p,ph.p,sizeof(pages)},{rd.p,rh.p,sizeof(rows)}})
        CHECK(rad_memcpy_async(std::get<0>(tuple),std::get<1>(tuple),std::get<2>(tuple),stream)>=0);
    RadTensor qt{};qt.data=qd.p;qt.dtype=RAD_F32;qt.rank=2;qt.shape[0]=nrows;qt.shape[1]=qheads*dim;rad_tensor_pack(&qt);
    RadTensor kt{};kt.data=kd.p;kt.dtype=dtype;kt.rank=4;kt.shape[0]=blocks;kt.shape[1]=kvheads;kt.shape[2]=native_block;kt.shape[3]=2*dim;rad_tensor_pack(&kt);
    AttentionKernelArgs args{qt,kt,static_cast<int32_t*>(td.p),static_cast<uint32_t*>(rd.p),static_cast<AttentionPage*>(pd.p),
        nrows,3,qheads,dim,99,static_cast<float*>(scratch.p),static_cast<float*>(output.p)};
    CHECK(attention_page_masses(args,stream)==RAD_OK);
    CHECK(rad_memcpy_async(host.p,output.p,nrows*3*4,stream)>=0 && rad_stream_sync(stream)>=0);
    auto* got=static_cast<float*>(host.p);
    for(unsigned row=0;row<nrows;++row) {
        double expected[3]{};
        for(unsigned h=0;h<qheads;++h) {
            double numerator[3]{},denominator=0;
            for(unsigned p=0;p<3;++p)for(unsigned t=0;t<pages[p].count;++t) {
                if(pages[p].start+t>99+row)continue;
                const auto compact=pages[p].compact+t;
                const auto kbase=((table[compact/native_block]*kvheads+h/(qheads/kvheads))*native_block+compact%native_block)*2*dim;
                long double dot=0;
                for(unsigned d=0;d<dim;++d)dot+=static_cast<long double>(q[(row*qheads+h)*dim+d])*kv[kbase+d];
                const double mass=double(std::exp(dot/std::sqrt(static_cast<long double>(dim))));
                numerator[p]+=mass;denominator+=mass;
            }
            for(unsigned p=0;p<3;++p)expected[p]+=numerator[p]/denominator/qheads;
        }
        for(unsigned p=0;p<3;++p)CHECK(std::abs(got[row*3+p]-expected[p])<2e-5);
    }
    CHECK(got[2]==0); // Original position 100 is invisible to Q at 99.
}
int main() {
    CHECK(rad_dev_set(0)>=0);RadStream stream=nullptr;CHECK(rad_stream_create(&stream,0)>=0);
    for(auto dtype:{RAD_F32,RAD_F16,RAD_BF16,RAD_F8E4M3})exercise(dtype,stream);
    rad_stream_destroy(stream);
    // Same shared PageSoftmax kernel used by NInfer, compiled for this HIP device.
    kvmem::RawKvStore store({1,2,0,4});const float a[]{8,0},b[]{0,4},c[]{40,0};
    store.write_layer_mean_sum(0,4,0,a);store.write_layer_mean_sum(4,4,0,b);store.write_layer_mean_sum(8,4,0,c);
    const float query[]{2,0,0,2, 1,2,2,1};const uint32_t count[]{2};
    const kvmem::PageQueryView view{query,count,1,2,2,1,2};
    const auto frame=kvmem::query_score_frame(12,4,2,0,0,{2});kvmem::PageSoftmaxCuda scorer(0);
    for(auto agg:{kvmem::GqaScoreAggregation::Mean,kvmem::GqaScoreAggregation::Max}) {
        const auto expected=kvmem::page_softmax_scores(store,view,frame,agg),actual=scorer.score(store,view,frame,agg);
        CHECK(expected.size()==actual.size());
        for(size_t i=0;i<expected.size();++i)CHECK(std::abs(expected[i]-actual[i])<2e-5);
    }
    std::cout<<"Native sparse causal attention and HIP shared Query reference passed\n";
}
