/**
Copyright 2020 Anonymous and Anonymous

Contributed by Anonymous

This file is part of PDoptFlow.

PDoptFlow is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PDoptFlow is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PDoptFlow.  If not, see <https://www.gnu.org/licenses/>.

**/

#pragma once

#include <chrono>

class Stopwatch {
private:
    std::chrono::high_resolution_clock::time_point t1, t2;

public:
    explicit Stopwatch(bool run = false) {
        if (run) {
            start();
        }
    }

    void start() { t2 = t1 = std::chrono::high_resolution_clock::now(); }

    void stop() { t2 = std::chrono::high_resolution_clock::now(); }

    double ms() const { return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0; }
};
