#pragma once
#include "kvmem/memory_contract.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kvmem::radiance {
struct AttentionPage { uint64_t start; uint32_t count, compact; };

// Use the view frozen at the last accepted/published boundary. By begin(), the
// scheduler has already reserved the new native batch: querying NativeMemory
// there would violate its accepted-frontier contract. Appends since the frozen
// view are contiguous, even when older resident pages are sparse.
inline std::vector<AttentionPage> attention_pages(const MemorySnapshot* frozen,
                                                uint32_t begin, uint32_t rows, uint32_t block) {
    if (!rows || !block || uint64_t(begin)+rows>UINT32_MAX)
        throw std::invalid_argument("invalid attention append geometry");
    std::vector<AttentionPage> pages;
    uint64_t appended=0, compact=0;
    if (frozen) {
        appended=frozen->stamp.frontier;
        if (!frozen->stamp.valid || appended>begin)
            throw std::invalid_argument("invalid frozen attention frontier");
        for (const auto& item:frozen->blocks) {
            if (std::find(frozen->resident.begin(),frozen->resident.end(),item.block.id)==frozen->resident.end())continue;
            const auto& b=item.block;
            if (!b.valid_tokens || b.valid_tokens>block || b.original_start%block ||
                b.original_start>appended || b.valid_tokens>appended-b.original_start || compact+b.valid_tokens>UINT32_MAX ||
                (!pages.empty() && pages.back().start+pages.back().count>b.original_start))
                throw std::invalid_argument("invalid frozen attention page");
            pages.push_back({b.original_start,b.valid_tokens,uint32_t(compact)});
            compact+=b.valid_tokens;
        }
        if (appended%block && (pages.empty() || pages.back().start+pages.back().count!=appended))
            throw std::invalid_argument("partial accepted tail must remain resident");
    }
    for (uint64_t pos=appended;pos<uint64_t(begin)+rows;) {
        const auto count=uint32_t(std::min<uint64_t>(block-pos%block,uint64_t(begin)+rows-pos));
        const auto start=pos/block*block;
        if (!pages.empty() && pages.back().start==start) pages.back().count+=count;
        else pages.push_back({start,count,uint32_t(compact)});
        compact+=count;pos+=count;
    }
    return pages;
}
}
