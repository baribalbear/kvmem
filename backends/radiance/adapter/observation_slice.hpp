#pragma once
#include "rad_runtime.h"
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kvmem::radiance {
inline RadTensor observation_rows(RadTensor value, int64_t begin, int64_t count) {
    const int bytes=value.dtype==RAD_F32 ? 4 : (value.dtype==RAD_F16 || value.dtype==RAD_BF16 ? 2 : 0);
    if (!value.data || !bytes || value.rank<2 || begin<0 || count<=0 || begin>value.shape[0] ||
        count>value.shape[0]-begin || value.stride[0]<=0 ||
        value.stride[0]>INT64_MAX/bytes || begin>INT64_MAX/(value.stride[0]*bytes))
        throw std::invalid_argument("invalid sequence observation rows");
    value.data=static_cast<char*>(value.data)+begin*value.stride[0]*bytes;
    value.shape[0]=count;
    return value;
}
struct ObservationBatch {
    RadBatch batch{};
    std::vector<RadKVGroupBatch> groups;
    ObservationBatch(const ObservationBatch&)=delete;
    ObservationBatch& operator=(const ObservationBatch&)=delete;
    ObservationBatch(const RadBatch& source, int64_t sequence, int64_t begin, int64_t count) : batch(source) {
        if (sequence<0 || sequence>=source.n_seq || begin<0 || count<=0 || begin>source.n_tok ||
            count>source.n_tok-begin || !source.positions || source.n_kv_groups<0 ||
            (source.n_kv_groups && !source.kv))
            throw std::invalid_argument("invalid sequence observation batch");
        if (source.n_kv_groups) groups.assign(source.kv,source.kv+source.n_kv_groups);
        for (auto& g : groups) {
            if (g.block_table) {
                if (g.block_table_pitch<=0 || sequence>INT64_MAX/g.block_table_pitch)
                    throw std::invalid_argument("invalid sequence table pitch");
                g.block_table+=sequence*g.block_table_pitch;
            }
            if (g.slot_mapping) g.slot_mapping+=begin;
            if (g.seqused) g.seqused+=sequence;
            if (g.state_index) g.state_index+=sequence*(g.state_index_pitch>0 ? g.state_index_pitch : 1);
        }
        batch.kv=groups.data(); batch.n_seq=1; batch.n_tok=count;
        batch.positions+=begin;
        if (batch.token_ids) batch.token_ids+=begin;
        if (batch.seq_ids) batch.seq_ids+=sequence;
        if (batch.q_lens) batch.q_lens+=sequence;
        if (batch.ctx_lens) batch.ctx_lens+=sequence;
        if (batch.num_accepted) batch.num_accepted+=sequence;
        // This is an observation view, never a model execution batch. Global
        // sampler/gather and cu offsets do not describe its rebased token rows.
        batch.cu_seqlens=nullptr; batch.out_ids=nullptr; batch.n_out=0;
        batch.n_seq_decode=sequence<source.n_seq_decode ? 1 : 0;
        batch.n_tok_decode=batch.n_seq_decode ? count : 0;
        batch.max_q_len=int32_t(count);
    }
};
} // namespace kvmem::radiance
