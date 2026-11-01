#include "rank_memory.hpp"
#include <iostream>
using namespace kvmem;using namespace kvmem::radiance;
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
template<class F> void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}CHECK(caught);}
struct Fake final:MemoryBackend {
    MemoryDescriptor desc;MemorySnapshot state;std::vector<uint64_t> device{0,1,2};
    int prepared=0,aborted=0,published=0,polls=0;
    bool busy=false,ready=true,fail_prepare=false,fail_restore=false,fail_publish=false,fail_abort=false;
    explicit Fake(int rank) {
        desc.identity={"radiance","test","model","rank"+std::to_string(rank)};
        desc.layout={"fake-rank"+std::to_string(rank),1,16,32,1,{{3,PayloadPlaneKind::PackedKeyValue,0,32,1,0}}};
        desc.capabilities={16,1024,true,true,true,false,false,true};
        state.stamp={true,{42,1},1,0,64,1};
        for(uint64_t i=0;i<4;++i)state.blocks.push_back({{{42,1},i,i*16,16,1,1},i==3 ? 1u:0u});
        state.resident=device;
    }
    MemoryDescriptor descriptor()const override{return desc;}
    MemorySnapshot snapshot()const override{return state;}
    MemoryStamp stamp()const noexcept override{return state.stamp;}
    void invalidate()noexcept override{state.stamp.valid=false;}
    struct Tx final:MemoryTransfer {
        Fake& f;WorkingSetPlan p;std::vector<uint64_t> old;int phase=0;bool done=false;
        Tx(Fake& f,const WorkingSetPlan& p):f(f),p(p),old(f.device){f.busy=true;}
        ~Tx()override{if(!done)abort();}
        void start_spill()override{CHECK(phase==0);phase=1;}
        TransferPoll poll()override {
            CHECK(phase==1 || phase==3);++f.polls;
            if(!f.ready)return {TransferStatus::Pending,{}};
            if(phase==1)for(auto& b:f.state.blocks)
                for(const auto& s:p.spill)if(s.id==b.block.id)b.host_version=s.content_version;
            ++phase;return {TransferStatus::Complete,{}};
        }
        void start_restore()override {
            CHECK(phase==2);phase=3;f.device.clear();
            for(const auto& b:p.next.blocks)f.device.push_back(b.block.id);
            if(f.fail_restore)throw std::runtime_error("injected rank restore");
        }
        void publish(const ExecutionView& v)override {
            CHECK(phase==4);f.state.stamp=v.stamp;
            if(f.fail_publish)throw std::runtime_error("injected rank publish after mutation");
            f.state.resident=f.device;done=true;f.busy=false;++f.published;
        }
        AbortResult abort()noexcept override {
            if(done)return f.state.stamp.valid ? AbortResult::PreviousViewPreserved:AbortResult::SessionInvalidated;
            ++f.aborted;f.device=old;done=true;f.busy=false;
            if(f.fail_abort){f.invalidate();return AbortResult::SessionInvalidated;}
            return AbortResult::PreviousViewPreserved;
        }
    };
    std::unique_ptr<MemoryTransfer> prepare(const WorkingSetPlan& p)override {
        CHECK(!busy);if(fail_prepare)throw std::runtime_error("injected rank prepare");
        ++prepared;return std::make_unique<Tx>(*this,p);
    }
};
struct Fixture {
    std::shared_ptr<Fake> a=std::make_shared<Fake>(0),b=std::make_shared<Fake>(1);
    std::shared_ptr<RankMemory> memory=std::make_shared<RankMemory>(std::vector<std::shared_ptr<MemoryBackend>>{a,b},BackendIdentity{"radiance","test","model","tp2"});
    MemorySession session{memory};
    void prepare(uint64_t bytes=256){session.prepare(memory->stamp(),{0,3},{0},{32,16,48,bytes});}
    void ready(){CHECK(session.advance()==MemoryPhase::Spilling);CHECK(session.advance()==MemoryPhase::Restoring);CHECK(session.advance()==MemoryPhase::Ready);}
};
int main() {
    {Fixture f;CHECK(f.memory->descriptor().layout.record_bytes==64);f.prepare();
        f.b->ready=false;CHECK(f.session.advance()==MemoryPhase::Spilling);CHECK(f.session.advance()==MemoryPhase::Spilling);
        const auto polls=f.a->polls;CHECK(f.session.advance()==MemoryPhase::Spilling);CHECK(f.a->polls==polls);
        CHECK(f.memory->snapshot().blocks[1].host_version==0); // only rank 0 finished spilling
        f.b->ready=true;CHECK(f.session.advance()==MemoryPhase::Restoring);CHECK(f.session.advance()==MemoryPhase::Ready);
        f.session.publish();CHECK(f.a->published==1 && f.b->published==1);CHECK(f.memory->snapshot().resident==std::vector<uint64_t>({0,3}));}
    {Fixture f;rejects([&]{f.prepare(64);});CHECK(f.a->prepared==0 && f.b->prepared==0);}
    {Fixture f;f.b->fail_prepare=true;rejects([&]{f.prepare();});CHECK(f.a->aborted==1 && f.memory->stamp().valid && !f.a->busy);}
    {Fixture f;f.prepare();f.b->fail_restore=true;CHECK(f.session.advance()==MemoryPhase::Spilling);
        rejects([&]{f.session.advance();});CHECK(f.memory->stamp().valid && f.a->device==std::vector<uint64_t>({0,1,2}) && f.b->device==f.a->device);}
    for(int rank=0;rank<2;++rank){Fixture f;f.prepare();f.ready();(rank ? f.b:f.a)->fail_publish=true;
        rejects([&]{f.session.publish();});CHECK(!f.a->stamp().valid && !f.b->stamp().valid);}
    {Fixture f;f.prepare();f.a->fail_abort=true;CHECK(f.session.abort()==AbortResult::SessionInvalidated);CHECK(!f.b->stamp().valid);}
    {Fixture f;f.b->state.stamp.frontier++;CHECK(!f.memory->stamp().valid);rejects([&]{f.memory->snapshot();});}
    std::cout<<"rank memory: coordinated publication, partial spill/restore, quota, prepare/publish/abort fault isolation PASS\n";
}
