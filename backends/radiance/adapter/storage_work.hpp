#pragma once
#include "kvmem/memory_contract.hpp"
#include "kvmem/spill_io.hpp"
#include <chrono>
#include <functional>

namespace kvmem::radiance {
// CPU/file errors are isolated from fatal device copy errors even though the
// portable storage contract uses TransferFailed for both domains.
class StorageFailure final : public MemoryError {
public:
    explicit StorageFailure(const std::string& message) : MemoryError(MemoryErrorCode::TransferFailed,message) {}
};
template<class F> void storage_operation(F&& operation) {
    try { operation(); }
    catch(const MemoryError& error) {
        if(error.code()==MemoryErrorCode::TransferFailed)throw StorageFailure(error.what());
        throw;
    }
}
// Used only at already-drained cache/checkpoint boundaries. The selection
// transaction uses nonblocking tickets instead. No GPU operation in action.
inline void complete_storage(const std::shared_ptr<StorageIoQueue>& io,uint64_t bytes,
                             std::function<void()> action) {
    if(!io) { storage_operation(action);return; }
    StorageTicket ticket;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    while(!ticket) {
        const auto epoch=io->completion_epoch();
        try { ticket=io->submit(bytes,[action]{storage_operation(action);}); }
        catch(const MemoryError& error) {
            if(error.code()!=MemoryErrorCode::Pending)throw;
            if(std::chrono::steady_clock::now()>=deadline)throw StorageFailure("storage queue admission timed out");
            io->wait_for_completion(epoch,std::chrono::milliseconds(20));
        }
    }
    ticket.wait(); // preserves captures until all started I/O is drained
}
} // namespace kvmem::radiance
