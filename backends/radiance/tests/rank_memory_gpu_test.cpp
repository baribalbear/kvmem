// Two physical devices, independent byte-pattern oracle for sharded KV/state.
#include "rank_native_memory.hpp"
#include "rank_prefix.hpp"
#include "native_pages.hpp"
#include "mem/pools.h"
#include "rad_internal.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <thread>
using namespace kvmem;using namespace kvmem::radiance;
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
struct Rank {
    int device;rad::Pools pools;rad::KVManager kv;RadStream stream=nullptr;
    std::vector<rad::KVGroupInfo> infos{3};
    std::unique_ptr<NativePages> pages;
    std::vector<GroupPool> pool;
    std::vector<std::vector<uint8_t>> expected;
    explicit Rank(int device):device(device) {
        CHECK(rad_dev_set(device)==RAD_OK);
        rad::Config cfg;cfg.vram_weights_mib=cfg.vram_kv_mib=cfg.gpu_headroom_mib=1;
        cfg.host_pool_mib=0;cfg.max_ctx=256;cfg.max_tok=32;cfg.max_seqs=2;cfg.n_spec=0;
        CHECK(pools.configure(cfg,0,&rad::rad_device_allocator(),&rad::rad_device_allocator())==RAD_OK);
        auto& full=infos[1];full.name="full";full.layers={3};full.block_size=16;
        full.decl.kind=RAD_KV_FULL;full.decl.dtype=RAD_BF16;full.decl.n_head_kv=1;full.decl.head_dim=8;
        auto& linear=infos[2];linear.name="linear";linear.layers={1};
        linear.decl.kind=RAD_KV_LINEAR;linear.decl.dtype=RAD_F32;linear.decl.n_head_kv=1;
        linear.decl.state_dim[0]=linear.decl.state_dim[1]=8;
        CHECK(kv.configure(infos,cfg,pools.kv())==RAD_OK);
        infos[1].bytes_per_block=kv.plan(1)->bytes_per_block;
        CHECK(rad_stream_create(&stream,0)==RAD_OK);
        pages=std::make_unique<NativePages>(kv.plans(),infos,32,"rank-test");
        pool={{1,kv.plan(1)->base,uint64_t(kv.plan(1)->pool_bytes)}};
    }
    ~Rank(){(void)rad_dev_set(device);if(stream){(void)rad_stream_sync(stream);rad_stream_destroy(stream);}}
    std::vector<CopySpan> spans(uint64_t seq,size_t block) const {
        const auto& table=kv.block_table(seq,1);CHECK(2*block+2<=table.size());
        return pages->spans(kv.plans(),{{1,uint64_t(kv.live_blocks(1)),{table.begin()+2*block,table.begin()+2*block+2}}});
    }
    void fill() {
        for(size_t block=0;block<4;++block) {
            std::vector<uint8_t> bytes(pages->layout().record_bytes);
            for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(device*71+block*31+i*17);
            PageTransfer copy(device,bytes.size(),stream);copy.restore(spans(17,block),pool,stream,bytes);CHECK(copy.drain());
            expected.push_back(std::move(bytes));
        }
        CHECK(rad_dev_set(device)==RAD_OK);
        CHECK(rad_memset_async(kv.state_ptr(2,0,kv.state_slot(17,2)),device+29,kv.plan(2)->layer_bytes_per_state(),stream)==RAD_OK);
        CHECK(rad_stream_sync(stream)==RAD_OK);
    }
    void verify(uint64_t seq,const std::vector<uint64_t>& selected) {
        for(size_t i=0;i<selected.size();++i) {
            PageTransfer copy(device,pages->layout().record_bytes,stream);copy.spill(spans(seq,i),pool,stream);CHECK(copy.drain());
            CHECK(copy.payload()==expected.at(selected[i]));
        }
        const auto* plan=kv.plan(2);const auto offset=static_cast<char*>(kv.state_ptr(2,0,kv.state_slot(seq,2)))-static_cast<char*>(plan->base);
        PageTransfer copy(device,plan->layer_bytes_per_state(),stream);
        copy.spill({{2,uint64_t(offset),0,uint64_t(plan->layer_bytes_per_state())}},{{2,plan->base,uint64_t(plan->pool_bytes)}},stream);
        CHECK(copy.drain());CHECK(copy.payload()==std::vector<uint8_t>(size_t(plan->layer_bytes_per_state()),device+29));
    }
};
int main() {
    try {
        if(rad_dev_count()<2){std::cout<<"SKIP: two visible GPUs are required\n";return 77;}
        auto a=std::make_shared<Rank>(0),b=std::make_shared<Rank>(1);
        CHECK(a->kv.add_sequence(17)==RAD_OK && a->kv.ensure(17,128)==RAD_OK);
        CHECK(b->kv.mirror_adapter_metadata(a->kv)==RAD_OK);
        a->fill();b->fill();
        const auto record=a->pages->layout().record_bytes;
        const auto directory=std::filesystem::temp_directory_path()/("kvmem-rank-test-"+std::to_string(std::random_device{}()));
        std::filesystem::create_directory(directory);
        auto disk=std::make_shared<SpillFile>(directory,64*record);
        auto storage=std::make_shared<HostKvStorage>(2*record,disk);
        auto io=std::make_shared<StorageIoQueue>(8,64*record,2);
        MemoryStamp stamp{true,{17,1},1,0,128,1};std::vector<BlockDescriptor> catalog;
        for(uint64_t i=0;i<4;++i)catalog.push_back({stamp.session,i,i*32,32,1,1});
        std::vector<std::shared_ptr<NativeMemory>> children;
        for(const auto& rank:{a,b})children.push_back(std::make_shared<NativeMemory>(
            MemoryBinding{rank,&rank->kv,rank->stream,rank->device,1},rank->infos,
            BackendIdentity{"radiance","test","byte-oracle","rank"+std::to_string(rank->device)},stamp,catalog,32,256,32*record,storage,io));
        auto memory=std::make_shared<RankNativeMemory>(children,BackendIdentity{"radiance","test","byte-oracle","tp2"});
        MemorySession session(memory);
        auto ready=[&] {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while(session.phase()!=MemoryPhase::Ready){CHECK(std::chrono::steady_clock::now()<deadline);session.advance();std::this_thread::sleep_for(std::chrono::microseconds(100));}
        };
        auto verify=[&](uint64_t seq,const std::vector<uint64_t>& selected) {
            CHECK(a->kv.block_table(seq,1)==b->kv.block_table(seq,1));CHECK(a->kv.state_slot(seq,2)==b->kv.state_slot(seq,2));
            a->verify(seq,selected);b->verify(seq,selected);
        };
        for(const std::vector<uint64_t>& selected:{std::vector<uint64_t>{0,3},{1,3},{2,3},{0,1,2,3}}) {
            session.prepare(memory->stamp(),selected,{3},{selected.size()*32,32,selected.size()*32+32,32*record});ready();session.publish();verify(17,selected);
        }
        session.prepare(memory->stamp(),{1,3},{3},{64,32,96,32*record});ready();
        CHECK(session.abort()==AbortResult::PreviousViewPreserved);verify(17,{0,1,2,3});
        memory->retain_view();
        RankPrefix prefix;std::vector<int32_t> tokens(128,5),prompt(129,5);
        for(const auto& rank:{a,b})prefix.add_rank(rank->device,rank->stream,rank->kv,
            std::make_unique<PrefixLease>(rank,rank->kv,17,tokens,rank->device,rank->stream));
        prefix.demote(prefix.page_bytes());prefix.park(storage,io);
        auto fork=prefix.fork();auto payload=fork->load_cpu();auto branch=memory->fork_retained({99,2});
        a->kv.free_sequence(17);CHECK(a->kv.add_sequence(99)==RAD_OK && a->kv.ensure(99,1)==RAD_OK);
        CHECK(b->kv.mirror_adapter_metadata(a->kv)==RAD_OK);
        fork->restore(99,prompt,&payload);branch->rebind_sequence(99,true);verify(99,{0,1,2,3});
        CHECK(branch->snapshot().stamp.frontier==128 && storage->charged_bytes()<=2*record && storage->disk_bytes()<=64*record);
        CHECK(storage->io_stats().read_bytes>0 && storage->io_stats().written_bytes>0);
        std::cout<<"TP2 byte oracle PASS: distinct-rank KV/state, shared physical indices, SSD selection/restore, abort and parked prefix fork\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
