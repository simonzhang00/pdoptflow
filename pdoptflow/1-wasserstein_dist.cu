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
#include <exception>

void printHelpAndExit(char* cmd){
    std::cout << "Usage: \n" << cmd << " file1 file2 --s <WSPD parameter, [default: 5]> " << std::endl;
    std::cout << "compute 1-Wasserstein distance persistence diagrams in file1 and file2.\n";
    exit(1);
}
int main(int argc, char **argv) {
    int additive= 250000;
    int multiplicative= 1;
    double s= 5;
    bool default_s= true;
    try {
        if (argc >= 3) {
            for (int i = 1; i < argc - 2; i++) {
                if (strcmp(argv[i], "--help") == 0) {
                    printHelpAndExit(argv[0]);
                } else if (strcmp(argv[i], "--s") == 0) {
                    s = atof(argv[i + 1]);
                    default_s= false;
                } else if (strcmp(argv[i], "--additive") == 0) {
                    additive = atoi(argv[i + 1]);
                } else if (strcmp(argv[i], "--multiplicative") == 0) {
                    multiplicative = atoi(argv[i + 1]);
                }
            }
        }else{
            printHelpAndExit(argv[0]);
        }
    }catch(std::exception& e){
        printHelpAndExit(argv[0]);
    }

    if(default_s){
        std::cout<<"no s specified, defaulting to s=5.\n";
    }else{
        std::cout<<"s: "<<s<<std::endl;
    }

/*
    if(argc>=7 && strcmp(argv[5], "--additive")==0){
        additive= atoi(argv[6]);
    }
    if(argc>=9 && strcmp(argv[7], "--multiplicative")==0){
        multiplicative= atoi(argv[8]);
    }
*/
    Stopwatch sw;
    sw.start();

    Stopwatch loading_sw;
    loading_sw.start();
    std::vector<PDoptFlow::W1::Point> diagramA;
    std::vector<PDoptFlow::W1::Point> diagramB;
    int decPrecision{0};
    if (!PDoptFlow::hera::readDiagramPointSet(argv[argc-2], diagramA, decPrecision)) {
        exit(1);
    }
    if (!PDoptFlow::hera::readDiagramPointSet(argv[argc-1], diagramB, decPrecision)) {
        exit(1);
    }

    loading_sw.stop();
    std::cout << "time to load PDs: " << loading_sw.ms() / 1000.0 << "s" << std::endl;
    double w1dist;
//    if (argc >= 5 && !std::string(argv[3]).compare("--s")) {
//        double s = atof(argv[4]);
//    }
        w1dist = PDoptFlow::wasserstein_dist(diagramA, diagramB, s, true, additive,multiplicative);//1.00248756219);//1+0.01);

//for(int i=1; i<40; i++){
//    std::cout<<"s " << i<<std::endl;
//    w1dist = PDoptFlow::wasserstein_dist(diagramA, diagramB, i, true, additive, multiplicative);//1.00248756219);//1+0.01);
//    std::cout << std::setprecision(15) << w1dist << std::endl << std::endl << std::endl;
//}

    sw.stop();
    std::cout << "time: " << std::setprecision(15) << sw.ms() / 1000.0 << "s" << std::endl;
    std::cout << std::setprecision(15) << w1dist << std::endl << std::endl << std::endl;
    return 0;
}
