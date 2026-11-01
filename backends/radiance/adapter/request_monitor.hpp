#pragma once
#include "kvmem/monitor.hpp"
#include "rad_core.h"

namespace kvmem::radiance {
class RequestMonitor {
    kvmem::monitor requests_;
public:
    void update(const rad::Request& r,int64_t age_us,int64_t first_us) noexcept {
        try {
            const std::string id=std::to_string(r.id);
            requests_.begin("generation",id);
            const bool terminal=r.state==rad::ReqState::Finished || r.state==rad::ReqState::Failed || r.state==rad::ReqState::Cancelled;
            const char* status=r.state==rad::ReqState::Waiting || r.state==rad::ReqState::Preempted ? "queued" :
                (r.output.empty() ? "reading" : "generating");
            nlohmann::json data={{"status",status},{"prompt_tokens",r.prompt.size()},{"cached_tokens",r.n_cached},
                {"prompt_processed_tokens",std::min<size_t>(r.prompt.size(),std::max(0,r.n_computed))},
                {"generated_tokens",r.output.size()},{"limit",r.max_tokens},{"has_media",bool(r.media)},
                {"started_at_ms",kvmem::monitor::now_ms()-age_us/1000},{"finish_reason",r.finish_reason},
                {"ttft_ms",nullptr},{"decode_tps",nullptr}};
            if(first_us>=0) {
                data["ttft_ms"]=double(first_us)/1000;
                if(age_us>first_us && r.output.size()>1)
                    data["decode_tps"]=double(r.output.size()-1)*1e6/double(age_us-first_us);
            }
            // TTFT includes admission/queue time. Do not label it prefill time.
            // All counts are native committed tokens; no request text is retained.
            requests_.update(id,data);
            if(terminal)requests_.finish(id,r.state==rad::ReqState::Finished ? "completed" :
                r.state==rad::ReqState::Cancelled ? "disconnected" : "error");
        } catch(...) { RAD_ERR("KVMem request monitor metadata allocation failed"); }
    }
    nlohmann::json snapshot() const { return requests_.snapshot(); }
};
} // namespace kvmem::radiance
