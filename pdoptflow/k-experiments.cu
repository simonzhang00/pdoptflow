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

#include <IO/read_asciidiagram.h>
#include <wasserstein.h>

//k-experiments did not show up in the paper, but are meant to demonstrate that the matching induced by the flow found by PDoptFlow
//surprisingly converges to a constant fraction of a "solution matching." Knowing this fraction
//allows us to lower the theoretical bound on the relative error
int main(int argc, char **argv) {

    for(int i=1; i<argc; i++){
        if(strcmp(argv[i], "--help") == 0 ){
            std::cout << "Usage: \n" << argv[0] << " file1 file2 < matchingfile " << std::endl;
            std::cout << "notice that the matchingfile must be read from std in"<<std::endl;
            std::cout << "compute k-experiment between persistence diagrams in file1 and file2.\n";
            exit(1);
        }
    }

    Stopwatch sw;
    sw.start();

    Stopwatch loading_sw;
    loading_sw.start();
    std::vector<PDoptFlow::W1::Point> diagramA;
    std::vector<PDoptFlow::W1::Point> diagramB;
    int decPrecision{0};
    if (!PDoptFlow::hera::readDiagramPointSet(argv[1], diagramA, decPrecision)) {
        exit(1);
    }
    if (!PDoptFlow::hera::readDiagramPointSet(argv[2], diagramB, decPrecision)) {
        exit(1);
    }

    loading_sw.stop();
    std::cout << "time to load PDs: " << loading_sw.ms() / 1000.0 << "s" << std::endl;
#define k_experiment
#ifdef k_experiment
    std::vector < std::pair < std::pair < int, int >, int >> diagonal_pairs_flow;
    std::vector < std::pair < std::pair < int, int >, int >> nondiagonal_pairs_flow;

//WARNING: run the k-experiments at the risk of your computer crashing due to a very dense min-cost flow transportation graph
//IT IS HIGHLY RECOMMENDED TO USE A SUPER COMPUTER WITH LARGE MEMORY FOR THE k-experiments on ANY of the provided datasetss
//be very careful about running _fullbipartite_, not all datasets will be able to handle the k-experiment due to size of bipartite graph
//std::vector<PDoptFlow::W1::Point> indextopoint;
//PDoptFlow::wasserstein_fullbipartite_generate_stats(argv[1], argv[2],
//                                                     diagonal_pairs_flow,nondiagonal_pairs_flow, indextopoint);

    std::vector <std::pair<PDoptFlow::W1::Point, PDoptFlow::W1::Point>> matching;
    std::string dummy;
    double ans = PDoptFlow::load_matching(dummy, matching);
    std::cout << "matching ans: " << ans << std::endl;
    for (double s = 1; s * s <= 64; s++) {
        std::cout << "RUNNING k experiment for s= " << s * s << std::endl;
        PDoptFlow::wasserstein_check_matching(argv[1], argv[2], s * s, matching);
//PDoptFlow::wasserstein_check_stats(argv[1], argv[2], s,
//                                         diagonal_pairs_flow,nondiagonal_pairs_flow, indextopoint);
#endif
    }
}
