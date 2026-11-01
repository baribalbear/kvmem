#include "request_monitor.hpp"
#include <cstdlib>
#include <iostream>
static void check(bool ok){if(!ok)std::abort();}
int main() {
    kvmem::radiance::RequestMonitor monitor;
    rad::Request a,b;a.id=1;b.id=2;a.prompt={12345,54321};b.prompt={99999};
    a.max_tokens=b.max_tokens=10;
    monitor.update(a,0,-1);monitor.update(b,0,-1);
    check(monitor.snapshot()["active"].size()==2);
    a.state=rad::ReqState::Running;a.n_computed=2;a.output={70000,80000};
    monitor.update(a,3000000,1000000);
    const auto row=monitor.snapshot()["active"][0];
    check(row["generated_tokens"]==2 && row["ttft_ms"]==1000 && row["decode_tps"]==.5);
    check(row["prefill_tps"].is_null());
    const auto data=monitor.snapshot().dump();
    check(data.find("54321")==std::string::npos && data.find("80000")==std::string::npos);
    a.state=rad::ReqState::Cancelled;a.finish_reason="cancelled";monitor.update(a,4000000,1000000);
    check(monitor.snapshot()["active"].size()==1 && monitor.snapshot()["recent"][0]["status"]=="disconnected");
    b.state=rad::ReqState::Finished;b.finish_reason="stop";b.output={5};monitor.update(b,4000000,2000000);
    check(monitor.snapshot()["active"].empty() && monitor.snapshot()["recent"].size()==2);
    std::cout << "request metadata isolation, privacy and native timing semantics passed\n";
}
