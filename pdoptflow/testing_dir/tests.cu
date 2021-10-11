/**
Copyright 2020, 2021 Tamal Dey and Simon Zhang

Contributed by Simon Zhang

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

#include <iostream>
#include <IO/read_asciidiagram.h>
#include "wasserstein.h"
#include <string>
#include <profiling/stopwatch.h>
#include <cmath>
#include <assert.h>

int main(int argc, char **argv) {
    std::ifstream infile(argv[1]);
    //std::ifstream infile("../../../datasets/tests/wasserstein_tests_QA_exact.txt");

    std::string f1, f2;
    std::string str_ans;
    double ans;
    int iter = 0;
    std::vector<PDoptFlow::W1::Point> diagramA;
    std::vector<PDoptFlow::W1::Point> diagramB;
    while (infile >> f1 >> f2 >> str_ans) {
        double computed = 0.0;
        int s = 12;
        do {
//if(iter<712)break;
            ans = std::stod(str_ans);
//            int decPrecision = 999999999;
//            if (!PDoptFlow::hera::readDiagramPointSet(f1, diagramA, decPrecision)) {
//                std::exit(1);
//            }
//            if (!PDoptFlow::hera::readDiagramPointSet(f2, diagramB, decPrecision)) {
//                std::exit(1);
//            }
//            computed = PDoptFlow::wasserstein_dist(diagramA, diagramB, s);
            computed = PDoptFlow::wasserstein_dist_fromfile(f1, f2, s, true, 250000,1);
            std::cout << "tested: " << f1 << " " << f2 << " with s: " << s << std::endl;
            printf("answer: %lf, computed (with lowest multiple of 12): %lf\n\n", ans, computed);
            s += 12;
            if (ans == computed) break;
        } while (std::fabs(ans - computed) > 0.1 * ans && s <= 168); //{
        iter++;
        assert(ans == computed || std::fabs(ans - computed) <= 0.1 * ans);
        //if(std::fabs(ans-computed)> 0.01*std::fabs(ans)){
            //std::cout << "FAILED: " << f1 << " " << f2 << " ans:" << ans << "!=computed:" << computed << std::endl;
            //exit(1);
        //}
    }
    std::cout << "ALL 0.1-RELATIVE ERROR TESTS SUCEEDED" << std::endl;
}