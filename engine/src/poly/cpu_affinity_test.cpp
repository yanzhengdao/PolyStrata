// Host-only selection tests: Windows CPU Sets are exercised on the actual hardware.
#include "cpu_affinity.hpp"
#include <cstdio>
#include <vector>

using strata::poly::cpu_affinity::Entry;
using strata::poly::cpu_affinity::fastest_sets;

int main() {
    const auto pick = [](const std::vector<Entry>& e) { return fastest_sets(e); };
    if (!pick({{1, 0, false, false, false}, {2, 0, false, false, false}}).empty()) return 1;
    if (pick({{1, 0, false, false, false}, {2, 1, false, false, false},
              {3, 1, false, false, false}}) != std::vector<unsigned long>{2, 3}) return 2;
    if (pick({{1, 0, false, false, false}, {2, 2, true, false, false},
              {3, 1, false, false, false}}) != std::vector<unsigned long>{3}) return 3;
    if (pick({{1, 0, false, false, false}, {2, 2, false, true, false},
              {3, 1, false, false, false}}) != std::vector<unsigned long>{3}) return 4;
    if (pick({{1, 0, false, false, false}, {2, 2, false, true, true}}) !=
        std::vector<unsigned long>{2}) return 5;
    std::printf("cpu_affinity_test: five checks passed\n");
    return 0;
}
