// Windows CPU Sets for an opt-in CPU-expert scheduling experiment.
// No process-wide affinity changes: GPU helpers, staging and prefetch keep their normal scheduling.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::poly::cpu_affinity {

struct Entry {
    unsigned long id;
    unsigned char efficiency;
    bool parked;
    bool allocated;
    bool ours;
};

// Pick every available logical CPU in the highest efficiency class.  If the CPU has only one
// class, it has no P/E distinction to request, so do not pretend that P-core mode is active.
inline std::vector<unsigned long> fastest_sets(const std::vector<Entry>& entries) {
    int lowest = 256, highest = -1;
    for (const Entry& e : entries) {
        if (e.parked || (e.allocated && !e.ours)) continue;
        lowest = std::min(lowest, (int) e.efficiency);
        highest = std::max(highest, (int) e.efficiency);
    }
    if (highest <= lowest) return {};
    std::vector<unsigned long> ids;
    for (const Entry& e : entries)
        if (!e.parked && (!e.allocated || e.ours) && e.efficiency == highest) ids.push_back(e.id);
    return ids;
}

#ifdef _WIN32
// GetSystemCpuSetInformation uses byte-sized, variable-length records.  Return an empty list
// when Windows cannot distinguish P and E cores or the query fails.
inline std::vector<unsigned long> windows_pcore_sets() {
    ULONG bytes = 0;
    if (GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) return {};
    std::vector<std::max_align_t> storage((bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
    ULONG received = 0;
    if (!GetSystemCpuSetInformation(reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(storage.data()), bytes,
                                    &received, GetCurrentProcess(), 0)) return {};
    std::vector<Entry> entries;
    const auto* p = reinterpret_cast<const unsigned char*>(storage.data());
    for (size_t at = 0; at + sizeof(DWORD) * 2 <= received;) {
        const auto* c = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(p + at);
        if (c->Size < sizeof(DWORD) * 2 || c->Size > received - at) return {};
        if (c->Type == CpuSetInformation && c->Size >= sizeof(SYSTEM_CPU_SET_INFORMATION))
            entries.push_back({c->CpuSet.Id, c->CpuSet.EfficiencyClass, c->CpuSet.Parked != 0,
                               c->CpuSet.Allocated != 0, c->CpuSet.AllocatedToTargetProcess != 0});
        at += c->Size;
    }
    return fastest_sets(entries);
}

inline bool select_current(const std::vector<unsigned long>& ids) {
    return ids.empty() || SetThreadSelectedCpuSets(GetCurrentThread(), ids.data(), (ULONG) ids.size()) != 0;
}

// The caller of Workers::run also computes expert tasks.  Restrict that thread ONLY while
// cpu_experts() runs, then restore its previous explicit selection (including no selection).
class CallerScope {
public:
    explicit CallerScope(const std::vector<unsigned long>& ids) {
        if (ids.empty()) return;
        ULONG count = 0;
        if (!GetThreadSelectedCpuSets(GetCurrentThread(), nullptr, 0, &count) &&
            GetLastError() != ERROR_INSUFFICIENT_BUFFER) return;
        old_.resize(count);
        if (count && !GetThreadSelectedCpuSets(GetCurrentThread(), old_.data(), count, &count)) return;
        applied_ = select_current(ids);
    }
    ~CallerScope() {
        if (applied_) (void) SetThreadSelectedCpuSets(GetCurrentThread(),
                                                       old_.empty() ? nullptr : old_.data(), (ULONG) old_.size());
    }
    CallerScope(const CallerScope&) = delete;
    CallerScope& operator=(const CallerScope&) = delete;
    bool applied() const { return applied_; }
private:
    std::vector<unsigned long> old_;
    bool applied_ = false;
};
#else
inline std::vector<unsigned long> windows_pcore_sets() { return {}; }
inline bool select_current(const std::vector<unsigned long>&) { return false; }
class CallerScope {
public:
    explicit CallerScope(const std::vector<unsigned long>&) {}
    bool applied() const { return false; }
};
#endif

} // namespace strata::poly::cpu_affinity
