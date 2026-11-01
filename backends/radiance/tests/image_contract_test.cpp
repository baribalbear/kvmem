#include "image_contract.hpp"
#include "kvmem/query_policy.hpp"
#include "kvmem/graph_recall.hpp"
#include <iostream>
using namespace kvmem::radiance;
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
template<class F> void rejects(F f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}CHECK(caught);}
rad::mm::PromptMedia picture(int start=126,uint64_t hi=11,uint64_t lo=22) {
    rad::mm::PromptMedia m;
    auto p=std::make_shared<rad::mm::Item>();p->kind=rad::mm::Kind::Image;
    p->n_seg=1;p->seg_rows=130;p->rows_w=13;p->rows_h=10;
    p->expand.assign(132,100);p->seg_at={1};p->key_hi=hi;p->key_lo=lo;
    m.items.push_back({p,start});m.rope.runs.push_back({start+1,130,13,start+1,-117,0,0});return m;
}
int main() {
    auto a=picture();const auto id=image_identity(&a,300);
    CHECK(id.size()==1 && id[0].begin==126 && id[0].end==258);
    CHECK(image_prefix_matches(id,&a,500,300));
    // Identical placeholder IDs and shape, but different image content.
    auto b=picture(126,11,23);CHECK(!image_prefix_matches(id,&b,500,300));
    auto c=picture();c.rope.runs[0].base++;CHECK(!image_prefix_matches(id,&c,500,300));
    CHECK(!image_prefix_matches(id,nullptr,500,300));
    CHECK(image_prefix_matches({},nullptr,500,300));
    CHECK(!image_prefix_matches({},&a,500,300));
    auto future=picture(500);CHECK(image_prefix_matches({},&future,700,300));
    CHECK(image_prefix(id,100).empty());rejects([&]{image_prefix(id,200);});
    CHECK(image_required(id,126,128).empty());
    CHECK(image_required(id,128,128)==std::vector<uint32_t>({0}));
    CHECK(image_required(id,256,128)==std::vector<uint32_t>({0,1}));
    CHECK(image_required(id,258,128)==std::vector<uint32_t>({0,1,2}));
    CHECK(kvmem::query_prefix_required(256,128,0,0,image_required(id,256,128))==std::vector<uint32_t>({0,1}));
    image_budget_check(id,128,512,1,1);
    rejects([&]{image_budget_check(id,128,384,1,1);});
    const auto kept=kvmem::pressure_selection({0,1,2,3},{0.,0.,0.,100.},{0,1,2},3);
    CHECK(kept==std::vector<uint32_t>({0,1,2}));
    rejects([&]{kvmem::pressure_selection({0,1,2,3},{0.,0.,0.,100.},{0,1,2},2);});
    auto bad=picture();bad.items[0].item->kind=rad::mm::Kind::Video;
    rejects([&]{image_identity(&bad,300);});
    bad=picture();bad.rope.runs[0].item=1;rejects([&]{image_identity(&bad,300);});
    bad=picture();bad.rope.runs[0].row0=1;rejects([&]{image_identity(&bad,300);});
    rejects([&]{image_identity(&a,250);});
    std::cout<<"Image content/layout identity, prefix boundaries, complete required pages and admission passed\n";
}
