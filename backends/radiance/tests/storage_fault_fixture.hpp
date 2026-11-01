#pragma once
#include <filesystem>
#include <stdexcept>
#ifdef __linux__
#include <unistd.h>
// Test-only fault injection. SpillFile immediately unlinks its private file.
// Match only this fixture's unique directory and exact spill-file prefix in
// THIS process, then truncate the still-open descriptor. No external file/FD.
inline void truncate_fixture_spill(const std::filesystem::path& directory) {
    const auto prefix=(std::filesystem::canonical(directory)/"kvmem-spill-").string();
    size_t count=0;
    for(const auto& entry:std::filesystem::directory_iterator("/proc/self/fd")) {
        std::error_code error;
        const auto target=std::filesystem::read_symlink(entry.path(),error).string();
        if(error || target.compare(0,prefix.size(),prefix)!=0)continue;
        if(target.size()<10 || target.substr(target.size()-10)!=" (deleted)")
            throw std::runtime_error("unexpected spill fault fixture descriptor");
        const auto fd=std::stoi(entry.path().filename().string());
        if(::ftruncate(fd,0)!=0)throw std::runtime_error("could not truncate fixture spill descriptor");
        ++count;
    }
    if(count!=1)throw std::runtime_error("fault injection must identify exactly one fixture spill file");
}
#endif
