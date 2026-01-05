#pragma once

#include <chrono>

class StopWatch
{
private:
    std::chrono::time_point<std::chrono::steady_clock> lastStartTime;

public:
    StopWatch()
    {
        lastStartTime = std::chrono::steady_clock::now();
    }

    ~StopWatch()
    {
    }

    void start()
    {
        lastStartTime = std::chrono::steady_clock::now();
    }

    double elapsedTime()
    {
        auto now = std::chrono::steady_clock::now();
        auto duration = now - lastStartTime;
        return std::chrono::duration<double>(duration).count();
    }
};
