#pragma once
#include "page_transfer.hpp"
#include "mem/kv.h"
#include <cstring>
#include <stdexcept>

namespace kvmem::radiance {
// Optional producer export. It folds accepted factors on an exclusively owned
// two-slot clone, on its own HIP stream; the call fences that stream before return.
using GdnMaterialize = int (*)(void*, int, int, int);
struct AcceptedState {
    const std::vector<rad::KVGroupInfo>* groups = nullptr;
    GdnMaterialize gdn = nullptr;
    uint32_t rows = 0; // zero = prefill, whose recurrent state is already committed
};

inline void canonical_conv_window(uint8_t* bytes, size_t size, uint64_t slots,
                                  uint64_t width, uint64_t element, uint32_t accepted) {
    if (!bytes || !accepted || width < 2 || slots < width - 1 || !element ||
        slots > SIZE_MAX / element || size % (slots * element) ||
        uint64_t(accepted - 1) > slots - (width - 1))
        throw std::runtime_error("invalid accepted convolution window");
    const size_t channel = size_t(slots * element), history = size_t((width - 1) * element);
    for (size_t offset = 0; offset < size; offset += channel) {
        std::memmove(bytes + offset, bytes + offset + size_t(accepted - 1) * element, history);
        std::memset(bytes + offset + history, 0, channel - history);
    }
}

inline void materialize_state(rad::KVManager& kv, uint64_t sequence,
                              const rad::KVGroupPlan& plan, int layer,
                              int device, RadStream stream, const AcceptedState& accepted,
                              uint8_t* destination, size_t size) {
    if (!accepted.rows) return;
    if (!accepted.groups || plan.index < 0 || size_t(plan.index) >= accepted.groups->size())
        throw std::runtime_error("missing accepted-state geometry");
    const auto& d = accepted.groups->at(size_t(plan.index)).decl;
    const auto element = d.dtype == RAD_F32 ? 4 :
                         (d.dtype == RAD_F16 || d.dtype == RAD_BF16 ? 2 : 0);
    if (plan.kind == RAD_KV_CONV) {
        canonical_conv_window(destination, size, uint64_t(plan.conv_slots),
                              uint64_t(d.conv_width), uint64_t(element), accepted.rows);
        return;
    }
    if (plan.kind != RAD_KV_LINEAR || plan.state_copies == 1) return;
    if (!accepted.gdn || plan.state_copies != 2 || d.state_dim[0] != 128 || d.state_dim[1] != 128 ||
        (d.dtype != RAD_F32 && d.dtype != RAD_F16) || d.n_head_kv <= 0 || d.n_head_kv > 65535 ||
        uint64_t(d.n_head_kv) * 128 * 128 * uint64_t(element) != size || size > size_t(INT64_MAX / 2))
        throw std::runtime_error("unsupported accepted recurrent-state representation");
    if (rad_dev_set(device) < 0) throw std::runtime_error("state snapshot device selection failed");
    struct Clone {
        void* data;
        RadStream stream;
        ~Clone() { if (data) { (void)rad_stream_sync(stream); rad_dev_free(data, RAD_MEM_DEVICE); } }
    } clone{rad_dev_alloc(int64_t(size * 2), RAD_MEM_DEVICE), stream};
    if (!clone.data) throw std::runtime_error("accepted-state clone allocation failed");
    const auto slot = kv.state_slot(sequence, plan.index);
    for (int copy = 0; copy < 2; ++copy) {
        const void* src = kv.state_ptr(plan.index, layer, int32_t(slot + copy * plan.n_seq_states()));
        if (!src || rad_memcpy_async(static_cast<char*>(clone.data) + copy * size, src, size, stream) < 0)
            throw std::runtime_error("accepted-state clone copy failed");
    }
    if (rad_stream_sync(stream) < 0 ||
        accepted.gdn(clone.data, int(d.n_head_kv), d.dtype == RAD_F16, int(accepted.rows)) != 0)
        throw std::runtime_error("accepted-state materialization failed");
    PageTransfer transfer(device, size, stream);
    transfer.spill({{0, 0, 0, size}}, {{0, clone.data, size * 2}}, stream);
    if (!transfer.drain()) throw std::runtime_error("accepted-state readback failed");
    const auto canonical = transfer.payload();
    std::memcpy(destination, canonical.data(), size);
}
} // namespace kvmem::radiance
