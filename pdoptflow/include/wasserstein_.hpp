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
#pragma once

#include <config.h>
#include <profiling/stopwatch.h>
#include <iostream>
#include <spanner/wspd.hh>
#include <lemon/static_graph.h>
#include <lemon/smart_graph.h>
#include <lemon/network_simplex.h>
#include <zconf.h>
#include <numeric>
#include <iomanip>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <map>
#include <set>
#include <bits/stdc++.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/extrema.h>
#include <kdtree/kdtree.h>
//#define DEBUG

namespace PDoptFlow {
    namespace W1 {
        template<typename T>
        struct absolute_value {
            __host__ __device__

            T operator()(const T &x) const {
                return x < T(0) ? -x : x;
            }
        };

        struct is_not_infinity_point {
            __host__ __device__

            bool operator()(Point p) {
                return !((p.x == std::numeric_limits<real_t>::infinity()) ||
                         (p.x == -std::numeric_limits<real_t>::infinity()) ||
                         (p.y == std::numeric_limits<real_t>::infinity()) ||
                         (p.y == -std::numeric_limits<real_t>::infinity()));
            }
        };

        struct eq_points {
            bool operator()(Point a, Point b) {
                return a == b;
            }
        };


        struct compare_Point {
            bool operator()(const Point &p1, const Point &p2) const {
                return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y);
            }
        };

        struct nondiag_compare_for_PDonenearestneighbor{
            Point* d_basepoint;
            Point* h_basepoint;

            __host__ __device__
            bool operator()(Point lhs, Point rhs)
            {
                //return lhs.dist_linfty(*h_basepoint)<rhs.dist_linfty(*h_basepoint);
                return lhs.dist_l2(*h_basepoint)<rhs.dist_l2(*h_basepoint);
            }
        };

        struct diag_compare_for_PDonenearestneighbor{
            __host__ __device__
            bool operator()(Point lhs,Point rhs)
            {
                //return std::abs(lhs.y-lhs.x)/2.0 < std::abs(rhs.y-rhs.x)/2.0;
                return std::abs(lhs.y-lhs.x)*sqrt2div2 < std::abs(rhs.y-rhs.x)*sqrt2div2;
            }
        };

        __global__ void map_WCD_xterms(real_t* d_terms, Point* d_PD,int* d_mass, int length, int mass_to_subtract) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (int i = tid; i < length; i += stride) {
                d_terms[i]= (d_mass[i]-mass_to_subtract)*d_PD[i].x;
            }
        }
        __global__ void map_WCD_yterms(real_t* d_terms, Point* d_PD,int* d_mass, int length, int mass_to_subtract) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (int i = tid; i < length; i += stride) {
                d_terms[i]= (d_mass[i]-mass_to_subtract)*d_PD[i].y;
            }
        }
        __global__ void copy_reverse_edges(int_pair *d_edges, int num_edges) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (int i = tid; i < num_edges; i += stride) {
                d_edges[i + num_edges].first = d_edges[i].second;
                d_edges[i + num_edges].second = d_edges[i].first;
            }
        }

        __global__ void transform_to_count_kernel(int *d_diagonalEdges, int *d_counts, int length) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (int i = tid; i < length; i += stride) {
                if (d_diagonalEdges[i] == 0) {
                    d_counts[i] = 2;
                } else {
                    d_counts[i] = d_diagonalEdges[i] < 0 ? -d_diagonalEdges[i] : d_diagonalEdges[i];
                }
            }
        }

        __global__ void gather_diagonal_edges(int_pair *d_edges, int *d_diagonalEdges, int *d_prefixsum_pointindices,
                                              int num_nondiagonal_edges, int num_diagonal_edges,
                                              int num_nondiagonal_nodes, int Abar, int Bbar) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            if (tid == 0) {
                d_edges[num_nondiagonal_edges + num_diagonal_edges - 1].first = Bbar;
                d_edges[num_nondiagonal_edges + num_diagonal_edges - 1].second = Abar;
            }
            for (int i = tid; i < num_nondiagonal_nodes; i += stride) {
                int j = d_prefixsum_pointindices[i];
                if (d_diagonalEdges[i] == -1) {
                    d_edges[num_nondiagonal_edges + j].first = Bbar;
                    d_edges[num_nondiagonal_edges + j].second = i;
                } else if (d_diagonalEdges[i] == 1) {
                    d_edges[num_nondiagonal_edges + j].first = i;
                    d_edges[num_nondiagonal_edges + j].second = Abar;
                } else if (d_diagonalEdges[i] == 0) {
                    d_edges[num_nondiagonal_edges + j].first = Bbar;
                    d_edges[num_nondiagonal_edges + j].second = i;
                    d_edges[num_nondiagonal_edges + j + 1].first = i;
                    d_edges[num_nondiagonal_edges + j + 1].second = Abar;
                }
            }
        }

        struct pair_hash {
            template<class T1, class T2>
            std::size_t operator()(const std::pair<T1, T2> &pair) const {
                return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
            }
        };

        inline real_t get_one_dimensional_cost(std::vector<real_t> set_A, std::vector<real_t> set_B) {
            if (set_A.size() != set_B.size()) {
                return std::numeric_limits<real_t>::infinity();
            }
            std::sort(set_A.begin(), set_A.end());
            std::sort(set_B.begin(), set_B.end());
            real_t result = 0.0;
            for (size_t i = 0; i < set_A.size(); ++i) {
                result += std::fabs(set_A[i] - set_B[i]);
            }
            return result;
        }

        real_t compute_infinity_cost(std::vector<Point> infinity_diagramA, std::vector<Point> infinity_diagramB) {
            std::vector<real_t> x_plus_A, x_minus_A, y_plus_A, y_minus_A;
            std::vector<real_t> x_plus_B, x_minus_B, y_plus_B, y_minus_B;

            for (auto a : infinity_diagramA) {
                real_t x = a.x;
                real_t y = a.y;
                if (x == std::numeric_limits<real_t>::infinity()) {
                    y_plus_A.push_back(y);
                } else if (x == -std::numeric_limits<real_t>::infinity()) {
                    y_minus_A.push_back(y);
                } else if (y == std::numeric_limits<real_t>::infinity()) {
                    x_plus_A.push_back(x);
                } else if (y == -std::numeric_limits<real_t>::infinity()) {
                    x_minus_A.push_back(x);
                }
            }
            // the same for B
            for (auto b : infinity_diagramB) {
                real_t x = b.x;
                real_t y = b.y;
                if (x == std::numeric_limits<real_t>::infinity()) {
                    y_plus_B.push_back(y);
                } else if (x == -std::numeric_limits<real_t>::infinity()) {
                    y_minus_B.push_back(y);
                } else if (y == std::numeric_limits<real_t>::infinity()) {
                    x_plus_B.push_back(x);
                } else if (y == -std::numeric_limits<real_t>::infinity()) {
                    x_minus_B.push_back(x);
                }
            }

            real_t infinity_cost = get_one_dimensional_cost(x_plus_A, x_plus_B);
            infinity_cost += get_one_dimensional_cost(x_minus_A, x_minus_B);
            infinity_cost += get_one_dimensional_cost(y_plus_A, y_plus_B);
            infinity_cost += get_one_dimensional_cost(y_minus_A, y_minus_B);
            return infinity_cost;
        }


        void zerocondense_points_staticgraph(std::vector<Point> &diagramA, std::vector<Point> &diagramB,
                                             std::vector<Point> &points, //std::vector<lemon::ListDigraph::Node>& nodes,
                                             std::vector<int> &massA, std::vector<int> &massB,
                                             std::vector<int> &mass,
                                             std::vector<int> &diagonalEdges, // holds count of each encountered number
                                             real_t &infinity_cost
        ) {
            auto infinity_startA = std::partition(diagramA.begin(), diagramA.end(), is_not_infinity_point());
            auto infinity_startB = std::partition(diagramB.begin(), diagramB.end(), is_not_infinity_point());
            std::vector<Point> infinity_diagramA(infinity_startA, diagramA.end());
            std::vector<Point> infinity_diagramB(infinity_startB, diagramB.end());
            diagramA.resize(infinity_startA - diagramA.begin());
            diagramB.resize(infinity_startB - diagramB.begin());

            infinity_cost = compute_infinity_cost(infinity_diagramA, infinity_diagramB);

            std::map<Point, std::pair<int, int>, compare_Point> supply;  // holds count of each encountered number in .first holds -1,0,1 indicator telling what diagonal edge incident to this node for .second

            massA.reserve(diagramA.size() + 1);
            massB.reserve(diagramB.size() + 1);

            compare_Point cmp;
            std::sort(diagramA.begin(), diagramA.end(), cmp);
            std::sort(diagramB.begin(), diagramB.end(), cmp);
            //https://stackoverflow.com/questions/39676779/counting-duplicates-in-c
            //the above post has an error at counter>=1 condition

            int counter;
            int j = 0;
            for (int i = 0; i < diagramA.size(); i += counter) {
                supply[diagramA[i]].first++;
                supply[diagramA[i]].second = 1;//atleast one edge from A to Abar
                for (counter = 1; i + counter < diagramA.size() && diagramA[i + counter] == diagramA[i];) {
                    counter++;       // count consecutives dups
                    supply[diagramA[i]].first++;
                }
                if (counter >= 1) {     // if more than one, process the dups.

                    massA.push_back(counter);
                    j++;
                }
            }
            auto uAend = std::unique(diagramA.begin(), diagramA.end());//this is optional

            int numAptns = uAend - diagramA.begin();
            diagramA.resize(numAptns);

            assert(j == uAend - diagramA.begin());

            j = 0;
            for (int i = 0; i < diagramB.size(); i += counter) {
                supply[diagramB[i]].first--;
                if (supply[diagramB[i]].second == 1) {
                    supply[diagramB[i]].second = 0;//both
                } else {
                    supply[diagramB[i]].second = -1;//only edge from Bbar to B
                }
                for (counter = 1; i + counter < diagramB.size() && diagramB[i + counter] == diagramB[i];) {
                    counter++;       // count consecutives dups
                    supply[diagramB[i]].first--;
                }
                if (counter >= 1) {     // if more than one, process the dups.

                    massB.push_back(counter);
                    j++;
                }
            }


            auto uBend = std::unique(diagramB.begin(), diagramB.end());//this is optional
            int numBptns = uBend - diagramB.begin();
            diagramB.resize(numBptns);

            assert(j == uBend - diagramB.begin());
            int massAsum = std::accumulate(massA.begin(), massA.end(), 0);
            int massBsum = std::accumulate(massB.begin(), massB.end(), 0);

            massB.push_back(massAsum);
            massA.push_back(massBsum);

#ifdef COUNTING
            printf("number of A masses (including diagonal node): %ld number of B masses (including diagonal node): %ld\n",
                   massA.size(), massB.size());
#endif

            points.reserve(numAptns + numBptns); // preallocate memory
            points.insert(points.end(), diagramA.begin(), uAend);
            points.insert(points.end(), diagramB.begin(), uBend);
            std::sort(points.begin(), points.end(), cmp);
            auto uptns_end = std::unique(points.begin(), points.end());
            points.erase(uptns_end, points.end());

#pragma omp parallel for schedule(static, 1)
            for (int i = 0; i < uptns_end - points.begin(); i++) {
                points[i].vertice = i;
            }
            j = 0;
            for (auto &e:supply) {
                mass.push_back(e.second.first);
                diagonalEdges.push_back(e.second.second);
                j++;
            }

            assert(j == uptns_end - points.begin());
            mass.push_back(massBsum);
            mass.push_back(-1 * massAsum);

            assert(mass.size() - 2 == points.size());

#ifdef DEBUG
            for(int i=0; i<mass.size(); i++){
                std::cout<<"mass for node "<<i<<" is: "<<mass[i]<<std::endl;
            }
#endif
        }


        void zerocondense_points(std::vector<Point> &diagramA, std::vector<Point> &diagramB,
                                 std::vector<Point> &points,
                                 std::vector<int> &massA, std::vector<int> &massB,
                                 std::vector<int> &mass, lemon::SmartDigraph &graph,
                                 std::vector<int> &diagonalEdges,// holds count of each encountered number
                                 real_t &infinity_cost
        ) {
            auto infinity_startA = std::partition(diagramA.begin(), diagramA.end(), is_not_infinity_point());
            auto infinity_startB = std::partition(diagramB.begin(), diagramB.end(), is_not_infinity_point());
            std::vector<Point> infinity_diagramA(infinity_startA, diagramA.end());
            std::vector<Point> infinity_diagramB(infinity_startB, diagramB.end());
            diagramA.resize(infinity_startA - diagramA.begin());
            diagramB.resize(infinity_startB - diagramB.begin());

            infinity_cost = compute_infinity_cost(infinity_diagramA, infinity_diagramB);

            std::map<Point, std::pair<int, int>, compare_Point> supply;  // holds count of each encountered number in .first holds -1,0,1 indicator telling what diagonal edge incident to this node for .second
            massA.reserve(diagramA.size() + 1);
            massB.reserve(diagramB.size() + 1);

            compare_Point cmp;
            std::sort(diagramA.begin(), diagramA.end(), cmp);
            std::sort(diagramB.begin(), diagramB.end(), cmp);
            //https://stackoverflow.com/questions/39676779/counting-duplicates-in-c
            //except it has an error at counter>=1 condition
            int counter;
            int j = 0;
            for (int i = 0; i < diagramA.size(); i += counter) {
                supply[diagramA[i]].first++;
                supply[diagramA[i]].second = 1;//atleast one edge from A to Abar
                for (counter = 1; i + counter < diagramA.size() && diagramA[i + counter] == diagramA[i];) {
                    counter++;       // count consecutives dups
                    supply[diagramA[i]].first++;
                }
                if (counter >= 1) {     // if more than one, process the dups.

                    massA.push_back(counter);
                    //Aindex.push_back(j);
                    j++;
                }
            }
            auto uAend = std::unique(diagramA.begin(), diagramA.end());//this is optional
            int numAptns = uAend - diagramA.begin();
            diagramA.resize(numAptns);

            assert(j == uAend - diagramA.begin());

            j = 0;
            for (int i = 0; i < diagramB.size(); i += counter) {
                supply[diagramB[i]].first--;
                if (supply[diagramB[i]].second == 1) {
                    supply[diagramB[i]].second = 0;//both
                } else {
                    supply[diagramB[i]].second = -1;//only edge from Bbar to B
                }
                for (counter = 1; i + counter < diagramB.size() && diagramB[i + counter] == diagramB[i];) {
                    counter++;       // count consecutives dups
                    supply[diagramB[i]].first--;
                }
                if (counter >= 1) {     // if more than one, process the dups.
                    massB.push_back(counter);
                    j++;
                }
            }


            auto uBend = std::unique(diagramB.begin(), diagramB.end());//this is optional
            int numBptns = uBend - diagramB.begin();
            diagramB.resize(numBptns);

            assert(j == uBend - diagramB.begin());

            int massAsum = std::accumulate(massA.begin(), massA.end(), 0);
            int massBsum = std::accumulate(massB.begin(), massB.end(), 0);

            massB.push_back(massAsum);
            massA.push_back(massBsum);
#ifdef COUNTING
            std::cout<<"number of A masses: "<<massA.size() <<" number of B masses: "<<massB.size()<<std::endl;
#endif
            points.reserve(numAptns + numBptns); // preallocate memory
            points.insert(points.end(), diagramA.begin(), uAend);
            points.insert(points.end(), diagramB.begin(), uBend);
            std::sort(points.begin(), points.end(), cmp);
            auto uptns_end = std::unique(points.begin(), points.end());
            points.erase(uptns_end, points.end());

            for (int i = 0; i < uptns_end - points.begin(); i++) {
                points[i].vertice = i;
                graph.addNode();
            }
//Bbar
            graph.addNode();
//Abar
            graph.addNode();
            j = 0;
            for (auto &e:supply) {
                mass.push_back(e.second.first);
                diagonalEdges.push_back(e.second.second);
                j++;
            }

            assert(j == uptns_end - points.begin());
            mass.push_back(massBsum);
            mass.push_back(-1 * massAsum);

            assert(mass.size() - 2 == points.size());

#ifdef DEBUG
            for(int i=0; i<mass.size(); i++){
                std::cout<<"mass for node "<<i<<" is: "<<mass[i]<<std::endl;
            }
#endif
        }

        class union_find {
            std::vector<int> parent;
            std::vector<uint8_t> rank;
            std::vector<float> birth;

        public:
            union_find(int n) : parent(n), rank(n, 0), birth(n, 0) {
                for (int i = 0; i < n; ++i)
                    parent[i] = i;
            }

            void set_birth(int i, int val) { birth[i] = val; }

            float get_birth(int i) { return birth[i]; }

            int find(int x) {
                int y = x, z = parent[y];
                while (z != y) {
                    y = z;
                    z = parent[y];
                }
                y = parent[x];
                while (z != y) {
                    parent[x] = z;
                    x = y;
                    y = parent[x];
                }
                return z;
            }

            void link(int x, int y) {
                x = find(x);
                y = find(y);
                if (x == y)
                    return;
                if (rank[x] > rank[y]) {
                    parent[y] = x;
                    birth[x] = std::min(birth[x], birth[y]);  // Elder rule
                } else {
                    parent[x] = y;
                    birth[y] = std::min(birth[x], birth[y]);  // Elder rule
                    if (rank[x] == rank[y])
                        ++rank[y];
                }
            }
        };

        void zero_condensation(std::vector<PDoptFlow::W1::Point> &diagramA, std::vector<PDoptFlow::W1::Point> &diagramB,
                               std::vector<int> &massA, std::vector<int> &massB, std::map<Point, std::pair<int, int>, compare_Point>& supply, real_t &infinity_cost) {
            auto infinity_startA = std::partition(diagramA.begin(), diagramA.end(), is_not_infinity_point());
            auto infinity_startB = std::partition(diagramB.begin(), diagramB.end(), is_not_infinity_point());
            std::vector<Point> infinity_diagramA(infinity_startA, diagramA.end());
            std::vector<Point> infinity_diagramB(infinity_startB, diagramB.end());
            diagramA.resize(infinity_startA - diagramA.begin());
            diagramB.resize(infinity_startB - diagramB.begin());

            infinity_cost = compute_infinity_cost(infinity_diagramA, infinity_diagramB);

            //zero condensation, collect masses
            //std::map<Point, std::pair<int, int>, compare_Point> supply;  // holds count of each encountered number in .first holds -1,0,1 indicator telling what diagonal edge incident to this node for .second
            massA.reserve(diagramA.size() + 1);
            massB.reserve(diagramB.size() + 1);

            compare_Point cmp;
            std::sort(diagramA.begin(), diagramA.end(), cmp);
            std::sort(diagramB.begin(), diagramB.end(), cmp);
            //https://stackoverflow.com/questions/39676779/counting-duplicates-in-c
            //except it has an error at counter>=1 condition
            int counter;
            int j = 0;
            for (int i = 0; i < diagramA.size(); i += counter) {
                supply[diagramA[i]].first++;
                supply[diagramA[i]].second = 1;//atleast one edge from A to Abar
                for (counter = 1; i + counter < diagramA.size() && diagramA[i + counter] == diagramA[i];) {
                    counter++;       // count consecutives dups
                    supply[diagramA[i]].first++;
                }
                if (counter >= 1) {     // if more than one, process the dups.

                    massA.push_back(counter);
                    //Aindex.push_back(j);
                    j++;
                }
            }
            auto uAend = std::unique(diagramA.begin(), diagramA.end());//this is optional
            int numAptns = uAend - diagramA.begin();
            diagramA.resize(numAptns);

            assert(j == uAend - diagramA.begin());

            j = 0;
            for (int i = 0; i < diagramB.size(); i += counter) {
                supply[diagramB[i]].first--;
                if (supply[diagramB[i]].second == 1) {
                    supply[diagramB[i]].second = 0;//both
                } else {
                    supply[diagramB[i]].second = -1;//only edge from Bbar to B
                }
                for (counter = 1; i + counter < diagramB.size() && diagramB[i + counter] == diagramB[i];) {
                    counter++;       // count consecutives dups
                    supply[diagramB[i]].first--;
                }
                if (counter >= 1) {     // if more than one, process the dups.
                    massB.push_back(counter);
                    j++;
                }
            }

            auto uBend = std::unique(diagramB.begin(), diagramB.end());//this is optional
            int numBptns = uBend - diagramB.begin();
            diagramB.resize(numBptns);

            assert(j == uBend - diagramB.begin());

            int massAsum = std::accumulate(massA.begin(), massA.end(), 0);
            int massBsum = std::accumulate(massB.begin(), massB.end(), 0);

            massB.push_back(massAsum);
            massA.push_back(massBsum);

        }

        real_t relaxed_WMD_kdtree(std::vector<PDoptFlow::W1::Point> diagramA, std::vector<PDoptFlow::W1::Point> diagramB,
                                  std::vector<int> massA, std::vector<int> massB) {
            int num_nondiagA = diagramA.size();
            int num_nondiagB = diagramB.size();

            struct diag_compare_for_PDonenearestneighbor diag_cmp;

            real_t res1 = 0.0;
            Point *d_otherDiagram;
            int num_nondiag_buffer_length = std::max(num_nondiagA, num_nondiagB);
            CUDACHECK(cudaMalloc((void **) &d_otherDiagram, num_nondiag_buffer_length * sizeof(Point)));

            CUDACHECK(cudaMemcpy(d_otherDiagram, &diagramB[0], num_nondiagB * sizeof(Point),
                                 cudaMemcpyHostToDevice));
            kdt::KDTree<Point> kdtree(diagramB);
            std::vector<real_t> res1_array(num_nondiagA+1,0.0);
#pragma omp parallel for schedule(static,1)
            for (int i = 0; i < num_nondiagA; i++) {
                real_t diag_dist=0.0;
                real_t mindist= 0.0;
                real_t* _mindist= &mindist;
                if (i >= 0) {
                    //mindist = (diagramA[i].get_linfty_dist_to_diag());
//                    mindist = (diagramA[i].get_l2_dist_to_diag());
                    diag_dist= fabs(diagramA[i].y-diagramA[i].x)*sqrt2div2;
                    kdtree.nnSearch(diagramA[i], _mindist);
                    //res1 += massA[i] * (*_mindist);
                    //std::cout<<"pointA: "<<diagramA[i].x<<" "<<diagramA[i].y<<" massA[i]"<<massA[i]<<" min dist: "<<(mindist)<<" _mindist: "<<*_mindist<<std::endl;

                    mindist= min(diag_dist, *_mindist);
                    res1_array[i]= massA[i]* mindist;
                } else {
                    auto minimizer_pointer = thrust::min_element(thrust::device, d_otherDiagram,
                                                                 d_otherDiagram + num_nondiagB, diag_cmp);
                    int minimizer_index = minimizer_pointer - d_otherDiagram;
                    assert(minimizer_index < num_nondiagB);
                    //mindist = diagramB[minimizer_index].get_linfty_dist_to_diag();
                    mindist = diagramB[minimizer_index].get_l2_dist_to_diag();
                    //res1 += massA.back() * mindist;
                    res1_array.back()= massA.back()*mindist;
                }
            }

#pragma omp parallel for reduction(+:res1)
            for (int i=0; i<res1_array.size(); i++)
            {
                res1+= res1_array[i];
            }

            real_t res2 = 0.0;
            kdtree.clear();
            kdtree.build(diagramA);
            CUDACHECK(cudaMemcpy(d_otherDiagram, &diagramA[0], num_nondiagA * sizeof(Point),
                                 cudaMemcpyHostToDevice));
            std::vector<real_t> res2_array(num_nondiagB+1,0.0);
#pragma omp parallel for schedule(static,1)
            for (int i = 0; i < num_nondiagB; i++) {
                real_t diag_dist= 0.0;
                real_t mindist = 0.0;
                real_t* _mindist= &mindist;
                if (i >= 0) {
                    //mindist = (diagramB[i].get_linfty_dist_to_diag());
//                    mindist = (diagramB[i].get_l2_dist_to_diag());
                    diag_dist= fabs(diagramB[i].y-diagramB[i].x)*sqrt2div2;
                    kdtree.nnSearch(diagramB[i], _mindist);
                    mindist= min(diag_dist, *_mindist);
                    //res2 += massB[i] * (*_mindist);
                    //std::cout<<"pointB: "<<diagramB[i].x<<" "<<diagramB[i].y<<" massB[i]"<<massB[i]<<" min dist: "<<mindist<<" _mindist: "<<*_mindist<<std::endl;
                    res2_array[i]= massB[i]* (mindist);
                }else {

                    auto minimizer_pointer= thrust::min_element(thrust::device, d_otherDiagram, d_otherDiagram+ num_nondiagB, diag_cmp);
                    int minimizer_index= minimizer_pointer-d_otherDiagram;
                    assert(minimizer_index<num_nondiagB);
                    //mindist= diagramA[minimizer_index].get_linfty_dist_to_diag();
                    mindist= diagramA[minimizer_index].get_l2_dist_to_diag();
                    //res2 += massB.back() * mindist;

                    res2_array.back()= massB.back() *mindist;
                }
            }
#pragma omp parallel for reduction(+:res2)
            for (int i=0; i<res2_array.size(); i++)
            {
                res2+= res2_array[i];
            }
#ifdef VERBOSE
            std::cout<<"res1: "<<res1<<" "<<"res2: "<<res2<<std::endl;
#endif
            cudaFree(d_otherDiagram);
            return std::max(res1, res2);
        }

        real_t relaxed_WMD_gpu(std::vector<PDoptFlow::W1::Point> diagramA, std::vector<PDoptFlow::W1::Point> diagramB,
                               std::vector<int> massA, std::vector<int> massB) {
            real_t res1 = 0.0;
            assert(massA.size() == diagramA.size() + 1);
            assert(massB.size() == diagramB.size() + 1);

            Stopwatch sw;
            sw.start();
            int num_nondiagA = diagramA.size();
            int num_nondiagB = diagramB.size();

            // Declare, allocate, and initialize device-accessible pointers for input and output
            Point* d_otherDiagram;
            int num_nondiag_buffer_length = std::max(num_nondiagA, num_nondiagB);
            CUDACHECK(cudaMalloc((void **) &d_otherDiagram, num_nondiag_buffer_length * sizeof(Point)));

            Point* d_basepoint;
            Point* h_basepoint;

            cudaHostAlloc((void **)&h_basepoint, sizeof(Point), cudaHostAllocPortable | cudaHostAllocMapped);
            cudaHostGetDevicePointer(&d_basepoint, h_basepoint,0);

            struct nondiag_compare_for_PDonenearestneighbor nondiag_cmp;
            nondiag_cmp.d_basepoint= d_basepoint;
            nondiag_cmp.h_basepoint= h_basepoint;

            struct diag_compare_for_PDonenearestneighbor diag_cmp;

            Point* h_minimizer_point;

            CUDACHECK(cudaMemcpy(d_otherDiagram, &diagramB[0], num_nondiagB * sizeof(Point),
                                 cudaMemcpyHostToDevice));
            for (int i = -1; i < num_nondiagA; i++) {
                real_t mindist;
                if (i >= 0) {
                    h_basepoint->x= diagramA[i].x;
                    h_basepoint->y= diagramA[i].y;
                    auto minimizer_pointer= thrust::min_element(thrust::device, d_otherDiagram, d_otherDiagram+num_nondiagB, nondiag_cmp);
                    int minimizer_index= minimizer_pointer-d_otherDiagram;
                    //mindist= diagramB[minimizer_index].dist_linfty(*h_basepoint);
                    mindist= diagramB[minimizer_index].dist_l2(*h_basepoint);
                    res1 += (massA[i]) * mindist;
                }else{
                    auto minimizer_pointer= thrust::min_element(thrust::device, d_otherDiagram, d_otherDiagram+ num_nondiagB, diag_cmp);
                    int minimizer_index= minimizer_pointer-d_otherDiagram;
                    assert(minimizer_index<num_nondiagB);
                    //mindist= diagramB[minimizer_index].get_linfty_dist_to_diag();
                    mindist= diagramB[minimizer_index].get_l2_dist_to_diag();
                    res1 += massA.back() * mindist;
                }
            }
            double res2= 0.0;


            CUDACHECK(cudaMemcpy(d_otherDiagram, &diagramA[0], num_nondiagA * sizeof(Point),
                                 cudaMemcpyHostToDevice));
            for (int i = -1; i < num_nondiagB; i++) {
                real_t mindist;
                if (i >= 0) {
                    h_basepoint->x= diagramB[i].x;
                    h_basepoint->y= diagramB[i].y;

                    auto minimizer_pointer= thrust::min_element(thrust::device, d_otherDiagram, d_otherDiagram+num_nondiagA, nondiag_cmp);
                    int minimizer_index= minimizer_pointer-d_otherDiagram;
                    //mindist= diagramA[minimizer_index].dist_linfty(*h_basepoint);
                    mindist= diagramA[minimizer_index].dist_l2(*h_basepoint);

                    res2 += (massB[i]) * mindist;
                }else{
                    auto minimizer_pointer= thrust::min_element(thrust::device, d_otherDiagram, d_otherDiagram+ num_nondiagA, diag_cmp);
                    int minimizer_index= minimizer_pointer-d_otherDiagram;
                    assert(minimizer_index<num_nondiagA);
                    //mindist= diagramA[minimizer_index].get_linfty_dist_to_diag();
                    mindist= diagramA[minimizer_index].get_l2_dist_to_diag();

                    res2 += massB.back() * mindist;
                }
            }
#ifdef VERBOSE
            std::cout<<"WMD; L_A: "<<res1<<", L_B: "<<res2<<std::endl;
#endif
            cudaFree(d_otherDiagram);
            return std::max(res1,res2);
        }

        real_t relaxed_WMD_cpu(std::vector<PDoptFlow::W1::Point> diagramA, std::vector<PDoptFlow::W1::Point> diagramB,
                               std::vector<int> massA, std::vector<int> massB) {
            real_t res1 = 0.0;
            assert(massA.size()==diagramA.size()+1);
            assert(massB.size()==diagramB.size()+1);

            Stopwatch sw;
            sw.start();
            int num_nondiagA= diagramA.size();
            int num_nondiagB= diagramB.size();


            //nearest neighbor search: O(n^2) brute force on CPU!!
            for (int i = 0; i < num_nondiagA; i++) {
                real_t mindist= std::numeric_limits<float>::infinity();
                //float mindist = diagramA[i].distance(diagramB[0]);
                if(i>-1){
//                    mindist = fabs(diagramA[i].y - diagramA[i].x) / 2.0;
                    mindist = fabs(diagramA[i].y - diagramA[i].x) *sqrt2div2;
                }
                int index = -1;
                real_t dist = mindist;
                for (int j = 0; j < num_nondiagB; j++) {
                    if (i == -1) {
                        dist= fabs(diagramB[j].x-diagramB[j].y)*sqrt2div2;
                    } else {
                        //dist = diagramB[j].dist_linfty(diagramA[i]);
                        dist = diagramB[j].dist_l2(diagramA[i]);
                    }
                    if (dist < mindist) {
                        mindist = dist;
                        index = j;
                        //std::cout<<massA[i]<<" "<<mindist<<std::endl;
                    }
                }
                if (i == -1) {
                    res1 += massA.back() * mindist;
                } else {
                    res1 += (massA[i]) * mindist;
                }
                //std::cout<<"point: "<<diagramA[i].x<<" "<<diagramA[i].y<<" "<<": "<<massA[i]<<" "<<mindist<<std::endl;
                //std::cout<<"res1: "<<res1<<std::endl;
            }
            real_t res2 = 0.0;
            //nearest neighbor search: O(n^2) brute force on CPU!!
            for (int i = 0; i < num_nondiagB; i++) {
                real_t mindist= std::numeric_limits<real_t>::infinity();
                //real_t mindist = diagramA[i].distance(diagramB[0]);
                if(i>-1){
//                    mindist = fabs(diagramB[i].y - diagramB[i].x) / 2.0;
                    mindist = fabs(diagramB[i].y - diagramB[i].x)*sqrt2div2;
                }
                int index = -1;
                real_t dist = mindist;
                for (int j = 0; j < num_nondiagA; j++) {
                    if (i == -1) {
                        dist= fabs(diagramA[j].x-diagramA[j].y)*sqrt2div2;
                    } else {
                        //dist = diagramA[j].dist_linfty(diagramB[i]);
                        dist = diagramA[j].dist_l2(diagramB[i]);
                    }
                    if (dist < mindist) {
                        mindist = dist;
                        index = j;
                    }
                }
                if (i == -1) {
                    res2 += massB.back() * mindist;
                } else {
                    res2 += (massB[i]) * mindist;
                }
                //std::cout<<"point: "<<diagramB[i].x<<" "<<diagramB[i].y<<" "<<": "<<massB[i]<<" "<<mindist<<std::endl;
                //std::cout<<"res1: "<<res2<<std::endl;
            }
            sw.stop();
#ifdef PROFILING
            std::cout<<"relaxed WMD time: "<<sw.ms()/1000.0<<"s"<<std::endl;
            std::cout << "res1: " << res1 << " res2: " << res2 << std::endl;
#endif
            return max(res1, res2);
        }
        real_t relaxed_WMD_kdtree_on_points(std::vector<PDoptFlow::W1::Point> diagramA, std::vector<PDoptFlow::W1::Point> diagramB){
            int num_nondiagA = diagramA.size();
            int num_nondiagB = diagramB.size();
            real_t res1 = 0.0;
            std::vector<Point> diagramA_augmented(num_nondiagB+num_nondiagA);
#pragma omp parallel for schedule(static,1)
            for(int i=0; i<num_nondiagB+num_nondiagA; i++) {
                if (i < num_nondiagB) {
                    diagramA_augmented[i] = {(diagramB[i].x + diagramB[i].y) / 2.0,
                                             (diagramB[i].x + diagramB[i].y) / 2.0};
                }else{
                    diagramA_augmented[i]= diagramA[i-num_nondiagB];
                }
            }
//            diagramA_augmented.insert(diagramA_augmented.end(), diagramA.begin(), diagramA.end());
//            std::cout<<"inserted diagramA_augmented"<<std::endl;
            std::vector<Point> diagramB_augmented(num_nondiagB+num_nondiagA);
#pragma omp parallel for schedule(static,1)
            for(int i=0; i<num_nondiagA+num_nondiagB; i++) {
                if (i < num_nondiagA) {
                    diagramB_augmented[i] = {(diagramA[i].x + diagramA[i].y) / 2.0,
                                             (diagramA[i].x + diagramA[i].y) / 2.0};
                }else{
                    diagramB_augmented[i]= diagramB[i-num_nondiagA];
                }
            }
//            diagramB_augmented.insert(diagramB_augmented.end(), diagramB.begin(), diagramB.end());
            std::cout<<"inserted diagramB_augmented"<<std::endl;
            kdt::KDTree<Point> kdtree(diagramB_augmented);
            std::vector<real_t> res1_array(num_nondiagA+num_nondiagB,0.0);
#pragma omp parallel for schedule(static,1)
            for (int i = 0; i < num_nondiagA+num_nondiagB; i++) {
                real_t diag_dist=0.0;
                real_t mindist= 0.0;
                real_t* _mindist= &mindist;
                if (i >= 0) {
                    //mindist = (diagramA[i].get_linfty_dist_to_diag());
//                    mindist = (diagramA[i].get_l2_dist_to_diag());
                    //diag_dist= fabs(diagramA_augmented[i].y-diagramA_augmented[i].x)*sqrt2div2;
                    kdtree.nnSearch(diagramA_augmented[i], _mindist);
                    //res1 += massA[i] * (*_mindist);
//                    std::cout<<"pointA: "<<diagramA_augmented[i].x<<" "<<diagramA_augmented[i].y<<" min dist: "<<(mindist)<<" _mindist: "<<*_mindist<<std::endl;

                    //mindist= min(diag_dist, *_mindist);
                    res1_array[i]= mindist;
                }
            }

#pragma omp parallel for reduction(+:res1)
            for (int i=0; i<res1_array.size(); i++)
            {
                res1+= res1_array[i];
            }

            real_t res2 = 0.0;
            kdtree.clear();
            kdtree.build(diagramA_augmented);
            std::vector<real_t> res2_array(num_nondiagB+num_nondiagA,0.0);
#pragma omp parallel for schedule(static,1)
            for (int i = 0; i < num_nondiagB+num_nondiagA; i++) {
                real_t diag_dist= 0.0;
                real_t mindist = 0.0;
                real_t* _mindist= &mindist;
                if (i >= 0) {
                    //mindist = (diagramB[i].get_linfty_dist_to_diag());
//                    mindist = (diagramB[i].get_l2_dist_to_diag());
                    //diag_dist= fabs(diagramB_augmented[i].y-diagramB_augmented[i].x)*sqrt2div2;
                    kdtree.nnSearch(diagramB_augmented[i], _mindist);
                    //mindist= min(diag_dist, *_mindist);
                    //res2 += massB[i] * (*_mindist);
//                    std::cout<<"pointB: "<<diagramB_augmented[i].x<<" "<<diagramB_augmented[i].y<<" min dist: "<<mindist<<" _mindist: "<<*_mindist<<std::endl;
                    res2_array[i]= (mindist);
                }
            }
#pragma omp parallel for reduction(+:res2)
            for (int i=0; i<res2_array.size(); i++)
            {
                res2+= res2_array[i];
            }
#ifdef VERBOSE
            std::cout<<"res1: "<<res1<<" "<<"res2: "<<res2<<std::endl;
#endif
//            std::cout<<"res1: "<<res1<<" "<<"res2: "<<res2<<std::endl;

            return std::max(res1, res2);
        }
        struct add_Point:public thrust::binary_function<PDoptFlow::W1::Point, PDoptFlow::W1::Point, PDoptFlow::W1::Point>
        {

            __host__ __device__
            PDoptFlow::W1::Point operator()(PDoptFlow::W1::Point p, PDoptFlow::W1::Point q) { PDoptFlow::W1::Point(p.x+q.x,p.y+q.y); }
        };

        struct unary_: public thrust::unary_function<int, int> {

            __host__ __device__ int operator()(const int x) const
            {
                return x+2;
            }
        };
        struct Point_median: public thrust::unary_function<PDoptFlow::W1::Point, PDoptFlow::W1::Point> {

            __host__ __device__ PDoptFlow::W1::Point operator()(const PDoptFlow::W1::Point &p) const
            {
                return PDoptFlow::W1::Point((p.x+p.y)/2.0, (p.x+p.y)/2.0);
            }
        };
        real_t centroid_distance_gpu(std::vector<PDoptFlow::W1::Point> diagramA, std::vector<PDoptFlow::W1::Point> diagramB){
            PDoptFlow::W1::Point* d_diagramA;
            PDoptFlow::W1::Point* d_diagramB;

            CUDACHECK(cudaMalloc((void**) &d_diagramA, sizeof(Point)* diagramA.size()));
            CUDACHECK(cudaMalloc((void**) &d_diagramB, sizeof(Point)* diagramB.size()));
            cudaMemcpy(d_diagramA, &diagramA[0], sizeof(Point)*diagramA.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_diagramB, &diagramB[0], sizeof(Point)*diagramB.size(), cudaMemcpyHostToDevice);

            thrust::plus<Point> plus;
//            struct add_Point add;
            struct Point_median unary;
            auto pA= thrust::reduce(thrust::device, d_diagramA, d_diagramA+diagramA.size(), Point(0.0,0.0), plus);
            CUDACHECK(cudaDeviceSynchronize());
            //            struct unary_ u;
//            thrust::negate<int> uu;
//            int* d_int;
//            cudaMalloc((void**) d_int, sizeof(int)*3);
//            thrust::transform(thrust::host,d_int, d_int+3,d_int,u);
            Point pB= thrust::reduce(thrust::device, d_diagramB, d_diagramB+diagramB.size(), Point(0,0), plus);
            CUDACHECK(cudaDeviceSynchronize());
//            thrust::transform(thrust::device, d_diagramA, d_diagramA+diagramA.size(), d_diagramA, unary);
//            CUDACHECK(cudaDeviceSynchronize());
//            thrust::transform(thrust::device, d_diagramB, d_diagramB+diagramB.size(), d_diagramB, unary);
//            CUDACHECK(cudaDeviceSynchronize());
//            auto pA2= thrust::reduce(thrust::device, d_diagramA, d_diagramA+diagramA.size(), Point(0,0), add);
//            CUDACHECK(cudaDeviceSynchronize());
//            auto pB2= thrust::reduce(thrust::device, d_diagramB, d_diagramB+diagramA.size(), Point(0,0), add);
//            CUDACHECK(cudaDeviceSynchronize());
            Point pA2= thrust::transform_reduce(thrust::device,
                                                  d_diagramB, d_diagramB + diagramB.size(),
                                                  unary,
                                                  Point(0,0),
                                                  plus);

            Point pB2= thrust::transform_reduce(thrust::device,
                                                d_diagramA, d_diagramA + diagramA.size(),
                                                unary,
                                                Point(0,0),
                                                plus);
            cudaFree(d_diagramA);
            cudaFree(d_diagramB);
            return sqrt(((pA.x+pA2.x)-(pB.x+pB2.x))*((pA.x+pA2.x)-(pB.x+pB2.x)) + ((pA.y+pA2.y)-(pB.y+pB2.y))*((pA.y+pA2.y)-(pB.y+pB2.y)));

        }
        real_t centroid_distance_cpu(std::vector<PDoptFlow::W1::Point> diagramA, std::vector<PDoptFlow::W1::Point> diagramB){
            Point pA= {0.0,0.0};
//            real_t xA= 0.0;
//            real_t yA= 0.0;
            for(auto p : diagramA){
                pA.x+= p.x;
                pA.y+= p.y;
            }
            for(auto p : diagramB){
                pA.x+= (p.x+p.y)/2.0;
                pA.y+= (p.x+p.y)/2.0;
            }
            Point pB= {0.0,0.0};
//            real_t xB= 0.0;
//            real_t yB= 0.0;
            for(auto p : diagramB){
                pB.x+= p.x;
                pB.y+= p.y;
            }
            for(auto p : diagramA){
                pB.x+= (p.x+p.y)/2.0;
                pB.y+= (p.x+p.y)/2.0;
            }
            return sqrt((pB.y-pA.y)*(pB.y-pA.y)+(pB.x-pA.x)*(pB.x-pA.x));
        }

        real_t ICT(std::vector<PDoptFlow::W1::Point> diagramA, std::vector<PDoptFlow::W1::Point> diagramB,
                   std::vector<int>& massA, std::vector<int>& massB){
            Stopwatch sw;
            sw.start();
            real_t t=0;
            for(int i=0; i<massA.size(); i++){
                std::vector<int> idx(massB.size());
                iota(idx.begin(), idx.end(), 0);
                PDoptFlow::W1::Point p= diagramA[i];
                int nA= massA.size()-1;
                int nB= massB.size()-1;
                sort(idx.begin(), idx.end(),
                     [nA, nB, &p, &diagramB](size_t i1, size_t i2) {
                         if(i1==nA) {
                             if (i2 == nB) {
                                 return (p.y-p.x)/2.0<(diagramB[i2].y-diagramB[i2].x)/2.0;
                             }else{
                                 //return (p.y-p.x)/2.0<p.dist_linfty(diagramB[i2]);
                                 return (p.y-p.x)/2.0<p.dist_l2(diagramB[i2]);
                             }
                         }else{
                             if(i2==nB){
                                 //return p.dist_linfty(diagramB[i1])<(diagramB[i2].y-diagramB[i2].x)/2.0;
                                 return p.dist_l2(diagramB[i1])<std::abs(diagramB[i2].y-diagramB[i2].x)/2.0;
                             }else{
                                 //return p.dist_linfty(diagramB[i1]) < p.dist_linfty(diagramB[i2]);
                                 return p.dist_l2(diagramB[i1]) < p.dist_l2(diagramB[i2]);
                             }
                         }
                     });
                int l=0;
                while(massA[i]>0) {
                    int r = std::min(massA[i], massB[idx[l]]);

                    massA[i] -= r;
                    if(i==nA) {
                        if(idx[l]==nB){
                            t+= 0;
                        }else{
                            t+= r* (diagramB[idx[l]].y-diagramB[idx[l]].x)/2.0;
                        }
                    }else {
                        if (idx[l] == nB) {
                            t += r * (p.y - p.x) / 2.0;
                        } else {
                            //t += r * p.dist_linfty(diagramB[idx[l]]);
                            t += r * p.dist_l2(diagramB[idx[l]]);
                        }
                    }
                    l++;
                }
            }
            sw.stop();
#ifdef PROFILING
            std::cout<<"ICT time: "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
            return t;
        }
        void delta_condense_points_staticgraph(std::vector<PDoptFlow::W1::Point>& diagramA, std::vector<PDoptFlow::W1::Point>& diagramB,
                                               std::vector<int>massA, std::vector<int> massB,
                                               real_t epsilon_rel_error, real_t radii_lower_bound, std::vector<Point> &points,
                                               std::vector<int> &mass,int totalAMass, int totalBMass, std::vector<int> &diagonalEdges,  std::map<Point, std::pair<int, int>, compare_Point> supply_zerocondensed) {
            Stopwatch sw;
            sw.start();

            std::vector<PDoptFlow::W1::Point> diagramA_deltacondensed(diagramA.size());
            std::vector<PDoptFlow::W1::Point> diagramB_deltacondensed(diagramB.size());

//            real_t delta = epsilon_rel_error * radii_lower_bound * log2((double)(totalAMass+totalBMass)) / (totalAMass + totalBMass);
//            real_t delta = epsilon_rel_error * radii_lower_bound * log2((double) (diagramA.size()+diagramB.size())) / (diagramA.size()+diagramB.size());
            real_t delta = 0.99*epsilon_rel_error * radii_lower_bound /(sqrt2div2*(totalAMass+totalBMass));

#ifdef VERBOSE
            std::cout<<"delta: "<<delta<<std::endl;
#endif
            #ifdef COUNTING
            std::cout << "zero condensed diagramA.size()" << diagramA.size() << std::endl;
            std::cout << "zero condensed diagramB.size()" << diagramB.size() << std::endl;
#endif
            //delta= 0.5;
            real_t shiftx = 0.0;
            real_t shifty = 0.0;
            int pers0A= 0;
            int pers0A_mass= 0;
            srand(0);

            //shiftx= -delta + static_cast <real_t> (rand()) /( static_cast <real_t> (RAND_MAX/(2*delta)));
            //shifty= -delta + static_cast <real_t> (rand()) /( static_cast <real_t> (RAND_MAX/(2*delta)));
#pragma omp parallel for schedule(static,1)
            for (int i = 0; i < diagramA.size(); i++) {
                diagramA_deltacondensed[i].x = delta * ( round((diagramA[i].x + shiftx) / delta));// +static_cast <float> (rand()) / static_cast <float> (RAND_MAX)*delta;
                diagramA_deltacondensed[i].y = delta * ( round((diagramA[i].y + shifty) / delta));

                if (fabs(diagramA_deltacondensed[i].x - diagramA_deltacondensed[i].y)< EPS) {//get rid of 0-persistence points
                    diagramA_deltacondensed[i].x = std::numeric_limits<real_t>::max();
                    diagramA_deltacondensed[i].y = std::numeric_limits<real_t>::max();

                }else{
//                    std::cout << std::setprecision(15)<<diagramA_deltacondensed[i].x << " " << std::setprecision(15)<< diagramA_deltacondensed[i].y << std::endl;
                }
            }
            int pers0B= 0;
            int pers0B_mass= 0;
#ifdef VERBOSE
            std::cout << "-------------------------------------" << std::endl;
#endif
            #pragma omp parallel for schedule(static,1)
            for (int i = 0; i < diagramB.size(); i++) {
                diagramB_deltacondensed[i].x = delta * ( round((diagramB[i].x + shiftx) / delta));
                diagramB_deltacondensed[i].y = delta * ( round((diagramB[i].y + shifty) / delta));

                if (fabs(diagramB_deltacondensed[i].x - diagramB_deltacondensed[i].y)<EPS) {//get rid of 0-persistence points
                    diagramB_deltacondensed[i].x = std::numeric_limits<real_t>::max();
                    diagramB_deltacondensed[i].y = std::numeric_limits<real_t>::max();

                }else{
//                    std::cout << std::setprecision(15)<<diagramB_deltacondensed[i].x << " " << std::setprecision(15)<< diagramB_deltacondensed[i].y << std::endl;
                }
            }

            compare_Point cmp;

            /*experiment in paper: how much does graph shrink when doing grid snapping?*/

            points.reserve(diagramA_deltacondensed.size() + diagramB_deltacondensed.size()); // preallocate memory
            points.insert(points.end(), diagramA_deltacondensed.begin(), diagramA_deltacondensed.end());
            points.insert(points.end(), diagramB_deltacondensed.begin(), diagramB_deltacondensed.end());
            std::sort(points.begin(), points.end(), cmp);
            auto uptns_end = std::unique(points.begin(), points.end());
            if ((points.back().x == std::numeric_limits<real_t>::max() &&
                points.back().y == std::numeric_limits<real_t>::max())) {
                    //points.erase(uptns_end, points.end());
                    points.resize(uptns_end - points.begin()-1);
                    uptns_end--;
                    //std::cout<<"deleting infinity"<<std::endl;
            } else {
                //points.erase(uptns_end, points.end());
                points.resize(uptns_end - points.begin());
            }
            //std::cout<<"points.back() "<<std::setprecision(15)<<points.back()<<std::endl;
            assert(!(points.back().x == std::numeric_limits<real_t>::max() &&
                   points.back().y == std::numeric_limits<real_t>::max()));
//            for(int i=0; i<points.size(); i++){
//                std::cout<<"old supply "<<points[i].x<<" "<<points[i].y<<std::endl;
//                std::cout<<supply_zerocondensed[points[i]].first<<std::endl;
//            }
#pragma omp parallel for schedule(static, 1)
            for (int i = 0; i < uptns_end - points.begin(); i++) {
                points[i].vertice = i;
                points[i].x+= static_cast <double> (rand()) / static_cast <double> (RAND_MAX)*(delta/0.99)*0.01/2.0;
                //points[i].y-= static_cast <double> (rand()) / static_cast <double> (RAND_MAX)*(delta/0.99)*0.01;
            }

            //////////////////////////////
#ifdef VERBOSE
            std::cout<<"starting mass collection "<<std::endl;
#endif
            std::map<Point, std::pair<int, int>, compare_Point> supply;

            int counter=1 ;
            int j = 0;
            for (int i = 0; i < diagramA_deltacondensed.size(); i += counter) {
                if(!(diagramA_deltacondensed[i].x==std::numeric_limits<real_t>::max() && diagramA_deltacondensed[i].y==std::numeric_limits<real_t>::max())){
                    supply[diagramA_deltacondensed[i]].first+= massA[i];//supply_zerocondensed[diagramA[i]].first;
                    supply[diagramA_deltacondensed[i]].second = 1;//atleast one edge from A to Abar
                    counter=1;
                    //std::cout<<diagramA_deltacondensed[i].x<<" "<<diagramA_deltacondensed[i].y<<" "<<massA[i]<<std::endl;
                    assert(counter>0);

                }else{
                    pers0A_mass+= massA[i];
                }
            }

            counter= 1;
            j=0;
            for (int i = 0; i < diagramB_deltacondensed.size(); i += counter) {
                if(!(diagramB_deltacondensed[i].x==std::numeric_limits<real_t>::max() && diagramB_deltacondensed[i].y==std::numeric_limits<real_t>::max())) {
                    supply[diagramB_deltacondensed[i]].first -= massB[i];//+= supply_zerocondensed[diagramB[i]].first;
                    if (supply[diagramB_deltacondensed[i]].second == 1){
                        supply[diagramB_deltacondensed[i]].second = 0;//both
                    } else {
                        supply[diagramB_deltacondensed[i]].second = -1;//only edge from Bbar to B
                    }
//                std::cout<<diagramB_deltacondensed[i].x<<" "<<diagramB_deltacondensed[i].y<<" "<<massB[i]<<std::endl;

                    counter = 1;//assumes we are operating on a zero condensed diagram
                    assert(counter > 0);
                    if (counter >= 1) {     // if more than one, process the dups.
                        j++;
                    }
                }else{
                    pers0B_mass+= massB[i];
                }
            }

            j = 0;
            //std::cout<<"totalAMass"<<totalAMass<<std::endl;
            //std::cout<<"totalBMass"<<totalBMass<<std::endl;

            int doublediag= 0;
            int Bdiag= 0;
            int Adiag= 0;

            for (auto &e:supply) {
                mass.push_back(e.second.first);
                diagonalEdges.push_back(e.second.second);
                if(e.second.second==0) {
                    doublediag++;
//                    std::cout<<"doubled point: "<<e.first.x<< " "<< e.first.y<<std::endl;
                }else if(e.second.second==-1){
                    Bdiag++;
                }else if(e.second.second==1){
                    Adiag++;
                }
                //std::cout<<std::setprecision(15)<<points[j].x<<" "<<points[j].y<<" ; "<< e.first.x<< " "<<e.first.y<<std::endl;

                j++;
            }
#ifdef COUNTING
            std::cout<<"Adiag: "<<Adiag<<",Bdiag: "<<Bdiag<<", doubldiag: "<<doublediag<<std::endl;
#endif
            assert(supply.size() == points.size());

            mass.push_back(totalBMass-pers0B_mass);
            mass.push_back(-1 * (totalAMass-pers0A_mass));
#ifdef COUNTING
            std::cout<<"delta condense to "<<mass.size()<<" number of nodes for min cost flow"<<std::endl;
#endif
//            std::cout<<"totalAMass"<<totalAMass<<std::endl;
//            std::cout<<"totalBMass"<<totalBMass<<std::endl;
            assert(mass.size() - 2 == points.size());

        }
    }
}