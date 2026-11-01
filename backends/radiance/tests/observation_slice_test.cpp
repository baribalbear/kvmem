#include "observation_slice.hpp"
#include <iostream>
#include <stdexcept>
using namespace kvmem::radiance;
#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);}while(false)
int main() {
    try {
        float values[96]={};int32_t positions[12]={100,101,102,103,900,901,902,903,904,905,906,907};
        int32_t table[8]={11,12,13,14,31,32,33,34};int32_t state[4]={4,5,8,9};
        RadTensor tensor{};tensor.data=values;tensor.dtype=RAD_F32;tensor.rank=2;
        tensor.shape[0]=12;tensor.shape[1]=8;tensor.stride[0]=8;tensor.stride[1]=1;
        const auto sliced=observation_rows(tensor,4,8);
        CHECK(sliced.data==values+32 && sliced.shape[0]==8 && sliced.shape[1]==8);
        CHECK(tensor.data==values && tensor.shape[0]==12);
        RadKVGroupBatch group{};group.group=1;group.block_table=table;group.block_table_pitch=4;
        group.state_index=state;group.state_index_pitch=2;
        RadBatch batch{};batch.n_tok=12;batch.n_seq=2;batch.n_seq_decode=2;batch.n_tok_decode=12;
        batch.positions=positions;batch.n_kv_groups=1;batch.kv=&group;
        ObservationBatch one(batch,1,4,8);
        CHECK(one.batch.n_seq==1 && one.batch.n_tok==8 && one.batch.positions[0]==900);
        CHECK(one.batch.kv[0].block_table[0]==31 && one.batch.kv[0].state_index[0]==8);
        CHECK(one.batch.cu_seqlens==nullptr && one.batch.n_out==0);
        CHECK(batch.kv[0].block_table[0]==11 && batch.n_seq==2);
        int refused=0;
        try{(void)observation_rows(tensor,8,5);}catch(const std::invalid_argument&){++refused;}
        try{ObservationBatch bad(batch,2,0,1);}catch(const std::invalid_argument&){++refused;}
        try{ObservationBatch bad(batch,1,11,2);}catch(const std::invalid_argument&){++refused;}
        group.block_table_pitch=0;
        try{ObservationBatch bad(batch,1,4,8);}catch(const std::invalid_argument&){++refused;}
        CHECK(refused==4);
        std::cout<<"observation slicing ownership and bounds passed\n";
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
