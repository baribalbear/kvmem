#pragma once
#include "attention_kernel.hpp"
#include "statistics_capture.hpp"
#include "kvmem/graph_recall.hpp"
#include <memory>
namespace kvmem::radiance {
class AttentionCapture {
public:
    AttentionCapture(int device, std::vector<StatisticsLayer>, uint32_t max_rows, uint32_t max_pages);
    ~AttentionCapture();
    void begin(uint32_t original_begin, const std::vector<uint32_t>& rows,
               const std::vector<AttentionPage>& pages, RadStream);
    int observe(int layer, rad_kvgroup, int64_t head_dim, const RadTensor& q,
                const RadTensor& cache, const RadBatch&, RadStream);
    // Final accepted sample rows come from the accepted-token parser, not from
    // proposal count or emitted tokens that the target has not consumed yet.
    std::vector<AcceptedAttentionRow> collect(uint32_t accepted, const std::vector<uint32_t>& samples);
    void discard() noexcept;
private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace kvmem::radiance
