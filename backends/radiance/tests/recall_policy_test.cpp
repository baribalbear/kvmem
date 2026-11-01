#include "recall_policy.hpp"
#include "attention_plan.hpp"
#include <cmath>
#include <iostream>
using namespace kvmem;
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(false)
template<class F> void rejects(F f) { bool failed=false; try{f();}catch(const std::exception&){failed=true;} CHECK(failed); }
int main() {
    auto pages=attention_pages(nullptr,0,10,4);
    CHECK(pages.size()==3 && pages.back().count==2 && pages.back().compact==8);
    MemorySnapshot frozen;frozen.stamp.valid=true;frozen.stamp.frontier=10;
    frozen.resident={0,2};
    frozen.blocks={{{{},0,0,4,1,1},0},{{{},1,4,4,1,1},1},{{{},2,8,2,1,1},0}};
    // Native reservation may now be ahead of 10. The accepted view itself stays
    // frozen; extend its partial tail and later appends without reading native KV.
    pages=attention_pages(&frozen,13,4,4);
    CHECK(pages.size()==4 && pages[1].start==8 && pages[1].count==4);
    CHECK(pages[2].start==12 && pages[2].compact==8 && pages[3].count==1);
    rejects([&]{attention_pages(&frozen,9,1,4);});
    frozen.resident={0};rejects([&]{attention_pages(&frozen,10,1,4);});
    auto global=std::make_shared<GraphStatisticsBudget>(256u<<20);
    RecallPolicy policy({},4,global);
    policy.start_request(true,false);
    policy.accept(0,12,true,{{3,{{0,1}}},{7,{{0,.8},{4,.2}}},{11,{{0,.7},{4,.2},{8,.1}}}}, {0,4,8},{});
    const auto base=policy.checkpoint();
    CHECK(policy.pressure(SelectionReason::PrefillPressure,{0,1,2},{2},2)==std::vector<uint32_t>({0,2}));
    rejects([&]{policy.pressure(SelectionReason::DecodePressure,{1,2},{0},2);});
    rejects([&]{policy.pressure(SelectionReason::Query,{0,1,2},{2},2);});
    policy.accept(12,1,false,{{12,{{0,.1},{4,.8},{8,.1}}}},{0,4,8,12},{false,16});
    CHECK(policy.pressure(SelectionReason::DecodePressure,{0,1,2,3},{3},2)==std::vector<uint32_t>({1,3}));
    // A pending output preview must not commit or decay on repeated Pressure.
    auto first=policy.feedback();
    (void)policy.pressure(SelectionReason::DecodePressure,{0,1,2,3},{3},2);
    CHECK(policy.feedback().output_samples==first.output_samples);
    CHECK(policy.feedback().pending_output==first.pending_output);
    policy.restore(base);
    CHECK(policy.graph().accepted_through==12 && policy.output().eligible_tokens==0);
    CHECK(policy.feedback().output_samples==0);
    CHECK(policy.pressure(SelectionReason::DecodePressure,{1,2},{2},1)==std::vector<uint32_t>({2}));
    // An invalid acceptance must leave the complete old snapshot usable.
    rejects([&]{policy.accept(12,1,false,{{13,{{0,1}}}},{0},{});});
    CHECK(policy.graph().accepted_through==12);
    RawKvStore keys({1,2,0,4});
    const float a[]{8,0},b[]{0,8},c[]{-8,0};
    keys.write_layer_mean_sum(0,4,0,a);keys.write_layer_mean_sum(4,4,0,b);keys.write_layer_mean_sum(8,4,0,c);
    const float q[]{1,0,1,0};const uint32_t rows[]{2};
    const PageQueryView view{q,rows,1,2,1,1,2};
    const auto frame=query_score_frame(12,4,2,0,0,{2});
    CHECK(policy.query(keys,view,frame,{2},2)==std::vector<uint32_t>({0,2}));
    for (auto scorer:{QueryScorer::PageSoftmax,QueryScorer::MeanQCosine}) for(int agg:{0,1}) {
        RecallOptions options;options.scorer=scorer;options.aggregation=agg;options.lrfu=true;options.graph_alpha=0;
        RecallPolicy p(options,4);p.start_request(true,false);p.accept(0,12,true,{}, {},{});
        const auto saved=p.checkpoint();
        CHECK(p.query(keys,view,frame,{2},2)==std::vector<uint32_t>({0,2}));p.commit_query();
        CHECK(p.checkpoint().lrfu.committed);
        p.restore(saved);CHECK(!p.checkpoint().lrfu.committed);
    }
    const auto range=bounded_query(4,1000,1004,512);CHECK(range.begin==492 && range.end==1004);
    CHECK(!query_replay_plan(range,384,3000,2048,128,4,32,512).admitted);
    CHECK(query_replay_plan(range,384,1100,2048,128,4,32,512).admitted);
    std::cout<<"Radiance common recall/Pressure/checkpoint invariants passed\n";
}
