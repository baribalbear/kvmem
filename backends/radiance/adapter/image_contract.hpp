#pragma once
#include "mm/processor.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kvmem::radiance {
// Own only immutable content/layout identity. Pixel and encoder buffers remain
// with the current native request; a retained KV entry never borrows them.
struct ImageIdentity {
    uint64_t key_hi=0,key_lo=0;
    int32_t begin=0,end=0,run_begin=0,rows=0,width=0,base=0,shift=0,row0=0;
    bool operator==(const ImageIdentity&) const = default;
};
using ImageIdentities=std::vector<ImageIdentity>;

inline ImageIdentities image_identity(const rad::mm::PromptMedia* media,uint64_t prompt) {
    ImageIdentities out;
    if(!media)return out;
    if(media->items.size()!=media->rope.runs.size())
        throw std::invalid_argument("Radiance KVMem supports still images with one complete native segment each");
    int64_t prior=0;
    for(size_t i=0;i<media->items.size();++i) {
        const auto& placed=media->items[i];
        if(!placed.item || placed.item->kind!=rad::mm::Kind::Image)
            throw std::invalid_argument("Radiance KVMem currently accepts images, not video or audio");
        const auto& item=*placed.item;const auto& r=media->rope.runs[i];
        const int64_t end=int64_t(placed.tok)+int64_t(item.expand.size());
        if(placed.tok<prior || item.expand.empty() || end>INT32_MAX || uint64_t(end)>prompt ||
           item.n_seg!=1 || item.seg_rows<=0 || r.item!=int32_t(i) || r.n!=item.seg_rows ||
           r.w<=0 || r.n%r.w || r.row0!=0 || r.tok<placed.tok || int64_t(r.tok)+r.n>end ||
           item.seg_at.size()!=1 || int64_t(placed.tok)+item.seg_at[0]!=r.tok)
            throw std::invalid_argument("invalid native image placement or rotary layout");
        out.push_back({item.key_hi,item.key_lo,placed.tok,int32_t(end),r.tok,r.n,r.w,r.base,r.shift,r.row0});
        prior=end;
    }
    return out;
}
inline ImageIdentities image_prefix(const ImageIdentities& all,uint64_t frontier) {
    ImageIdentities out;
    for(const auto& image:all) {
        if(uint64_t(image.begin)>=frontier)break;
        if(uint64_t(image.end)>frontier)
            throw std::invalid_argument("retained prefix cuts through an image");
        out.push_back(image);
    }
    return out;
}
inline bool image_prefix_matches(const ImageIdentities& saved,const rad::mm::PromptMedia* media,
                                 uint64_t prompt,uint64_t frontier) {
    try{return saved==image_prefix(image_identity(media,prompt),frontier);}
    catch(const std::invalid_argument&){return false;}
}
// Keep every accepted page intersecting an image, including its delimiter
// tokens and a partial page. Future rows are never inserted into a selection.
inline std::vector<uint32_t> image_required(const ImageIdentities& images,uint64_t frontier,uint32_t block) {
    if(!block)throw std::invalid_argument("zero image selection block");
    std::vector<uint32_t> out;
    for(const auto& image:images) {
        const auto end=std::min<uint64_t>(image.end,frontier);
        if(end<=uint64_t(image.begin))break;
        for(uint64_t page=uint64_t(image.begin)/block;page<=(end-1)/block;++page)
            if(out.empty() || out.back()!=page)out.push_back(uint32_t(page));
    }
    return out;
}
inline void image_budget_check(const ImageIdentities& images,uint32_t block,uint32_t budget,
                               uint32_t sink,uint32_t recent) {
    if(images.empty() || !budget)return;
    auto required=image_required(images,UINT32_MAX,block);
    for(uint32_t i=0;i<sink;++i)required.push_back(i);
    std::sort(required.begin(),required.end());
    required.erase(std::unique(required.begin(),required.end()),required.end());
    // A later decode can move the recent band beyond all images. Admit only
    // when that complete set and the accepted tail still fit the select budget.
    if(uint64_t(required.size())+std::max(recent,1u)>budget/block)
        throw std::invalid_argument("image pages plus sink/recent bands exceed the KVMem budget; reduce image size/count or increase Budget");
}
} // namespace kvmem::radiance
