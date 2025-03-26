//
// Created by gllekk on 11.02.25.
//

#ifndef TIMEMEASUREMENT_H
#define TIMEMEASUREMENT_H

#include <chrono>
#include <atomic>

template<class D>
inline long long to_ms(const D& d)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
}

inline std::chrono::high_resolution_clock::time_point get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}
#endif //TIMEMEASUREMENT_H
