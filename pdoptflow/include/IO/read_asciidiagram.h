/**
modified from Hera (https://bitbucket.org/grey_narn/hera/)

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

#pragma once

#include <config.h>
#include <fstream>
#include <algorithm>
#include <sstream>

namespace PDoptFlow {
    namespace hera {


// cannot choose stod, stof or stold based on RealType,
// lazy solution: partial specialization
        template<class RealType = double>
        inline RealType parse_real_from_str(const std::string &s);

        template<>
        inline double parse_real_from_str<double>(const std::string &s) {
            return std::stod(s);
        }


        template<>
        inline long double parse_real_from_str<long double>(const std::string &s) {
            return std::stold(s);
        }


        template<>
        inline float parse_real_from_str<float>(const std::string &s) {
            return std::stof(s);
        }


        template<class RealType>
        inline RealType parse_real_from_str(const std::string &s) {
            static_assert(sizeof(RealType) != sizeof(RealType),
                          "Must be specialized for each type you want to use, see above");
        }

// fill in result with points from file fname
// return false if file can't be opened
// or error occurred while reading
// decPrecision is the maximal decimal precision in the input,
// it is zero if all coordinates in the input are integers

        template<class RealType_ = double, class ContType_ = std::vector <PDoptFlow::W1::Point>>
        inline bool readDiagramPointSet(const char *fname, ContType_ &result, int &decPrecision) {
            using RealType = RealType_;

            size_t lineNumber{0};
            result.clear();
            std::ifstream f(fname);
            if (!f.good()) {
#ifndef FOR_R_TDA
                std::cerr << "Cannot open file " << fname << std::endl;
#endif
                return false;
            }
            std::locale loc;
            std::string line;
            while (std::getline(f, line)) {
                lineNumber++;
                // process comments: remove everything after hash
                auto hashPos = line.find_first_of("#", 0);
                if (std::string::npos != hashPos) {
                    line = std::string(line.begin(), line.begin() + hashPos);
                }
                if (line.empty()) {
                    continue;
                }
                // trim whitespaces
                auto whiteSpaceFront = std::find_if_not(line.begin(), line.end(), isspace);
                auto whiteSpaceBack = std::find_if_not(line.rbegin(), line.rend(), isspace).base();
                if (whiteSpaceBack <= whiteSpaceFront) {
                    // line consists of spaces only - move to the next line
                    continue;
                }
                line = std::string(whiteSpaceFront, whiteSpaceBack);

                // transform line to lower case
                // to parse Infinity
                for (auto &c : line) {
                    c = std::tolower(c, loc);
                }

                bool fracPart = false;
                int currDecPrecision = 0;
                for (auto c : line) {
                    if (c == '.') {
                        fracPart = true;
                    } else if (fracPart) {
                        if (isdigit(c)) {
                            currDecPrecision++;
                        } else {
                            fracPart = false;
                            if (currDecPrecision > decPrecision)
                                decPrecision = currDecPrecision;
                            currDecPrecision = 0;
                        }
                    }
                }

                RealType x, y;
                std::string str_x, str_y;
                std::istringstream iss(line);
                try {
                    iss >> str_x >> str_y;

                    x = parse_real_from_str<RealType>(str_x);
                    y = parse_real_from_str<RealType>(str_y);

                    if (x != y) {
                        PDoptFlow::W1::Point p;
                        //result.push_back(std::make_pair(x, y));
                        //result.push_back({x,y});
                        p.x = x;
                        p.y = y;
//                    p.name= std::to_string(x)+","+std::to_string(y);
                        result.push_back(p);
                        //std::cout<<x <<" "<<y<<std::endl;
                        //std::cout << typeid(x).name() << std::endl;
                    } else {
#ifndef FOR_R_TDA
                        std::cerr << "Warning: point with 0 persistence ignored in " << fname << ":" << lineNumber
                                  << "\n";
#endif
                    }
                }
                catch (const std::invalid_argument &e) {
#ifndef FOR_R_TDA
                    std::cerr << "Error in file " << fname << ", line number " << lineNumber << ": cannot parse \""
                              << line << "\"" << std::endl;
#endif
                    return false;
                }
                catch (const std::out_of_range &) {
#ifndef FOR_R_TDA
                    std::cerr << "Error while reading file " << fname << ", line number " << lineNumber
                              << ": value too large in \"" << line << "\"" << std::endl;
#endif
                    return false;
                }
            }
            f.close();
            return true;
        }

        // wrappers
        template<class RealType_ = double, class ContType_ = std::vector <std::pair<RealType_, RealType_>>>
        inline bool readDiagramPointSet(const std::string &fname, ContType_ &result, int &decPrecision) {
            return readDiagramPointSet<RealType_, ContType_>(fname.c_str(), result, decPrecision);
        }

        // these two functions are now just wrappers for the previous ones,
        // in case someone needs them; decPrecision is ignored
        template<class RealType_ = double, class ContType_ = std::vector <std::pair<RealType_, RealType_>>>
        inline bool readDiagramPointSet(const char *fname, ContType_ &result) {
            int decPrecision;
            return readDiagramPointSet<RealType_, ContType_>(fname, result, decPrecision);
        }

        template<class RealType_ = double, class ContType_ = std::vector <std::pair<RealType_, RealType_>>>
        inline bool readDiagramPointSet(const std::string &fname, ContType_ &result) {
            int decPrecision;
            return readDiagramPointSet<RealType_, ContType_>(fname.c_str(), result, decPrecision);
        }

    } // end namespace hera
}