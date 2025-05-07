//
// Created by gllekk on 11.02.25.
//

#ifndef TIMEMEASUREMENT_H
#define TIMEMEASUREMENT_H

#include <chrono>
#include <atomic>

template<class D>
inline double to_ms(const D& d)
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(d).count();
}

template<class D>
inline double to_ms_f(const D& d)
{
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(d).count();
}

inline std::chrono::high_resolution_clock::time_point get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}
#endif //TIMEMEASUREMENT_H
