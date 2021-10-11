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
#include <IO/read_asciidiagram.h>
#include <wasserstein_.hpp>

#include <lemon/dijkstra.h>

namespace PDoptFlow {
    namespace W1 {

        real_t compute_RWMD(std::vector<Point> &diagramA, std::vector<Point> &diagramB) {
            std::map<Point, std::pair<int, int>, compare_Point> supply;
            std::vector<int> massA;
            std::vector<int> massB;
            double infinity_cost= 0.0;
            zero_condensation(diagramA, diagramB, massA, massB, supply, infinity_cost);

//            real_t lower_bound = relaxed_WMD_kdtree_on_points(diagramA, diagramB);

//            real_t lower_bound = relaxed_WMD_cpu(diagramA, diagramB, massA, massB);
            real_t lower_bound2 = relaxed_WMD_kdtree(diagramA, diagramB, massA, massB);
//            std::cout<<"competing relaxed WMD: "<<lower_bound2<<std::endl;
//            return lower_bound/2.0;
            return lower_bound2;
        }

        real_t compute_centroid_dist(std::vector<Point> &diagramA, std::vector<Point> &diagramB){
            //std::map<Point, std::pair<int, int>, compare_Point> supply;
            //zero_condensation(diagramA, diagramB, massA, massB, supply, infinity_cost);
//            real_t lower_bound2 = centroid_distance_cpu(diagramA, diagramB);
            real_t lower_bound = centroid_distance_gpu(diagramA, diagramB);
//            if(lower_bound2!=lower_bound){
//                std::cout<<"gpu: "<<lower_bound<<"\n";
//                std::cout<<"cpu: "<<lower_bound2<<"\n";
//                exit(110);
//            }
            //assert(lower_bound2==lower_bound);
            return lower_bound/2.0;
        }

        real_t compute_wasserstein_dist(std::vector<Point> &diagramA, std::vector<Point> &diagramB, real_t s, bool delta_condense, int ADDITIVE_COEFF, int MULTIPLICATIVE_COEFF) {
#ifdef MIN_VERBOSE
            std::cout<<"Setting up sparsified transshipment network with s=" <<s<<"\n";
#endif
            Stopwatch sw;
            sw.start();

#ifdef COUNTING
            std::cout << "original diagramA.size()" << diagramA.size() << std::endl;
            std::cout << "original diagramB.size()" << diagramB.size() << std::endl;
#endif
            std::vector<Point> points;
            std::vector<int> massA;
            std::vector<int> massB;

            std::vector<int> mass;
            std::vector<int> diagonalEdges;
            // build the nodes and mass vectors for the spanner graph

            real_t infinity_cost = 0.0;
            real_t cost = 0.0;
            if (delta_condense && s < 164) {
                std::map<Point, std::pair<int, int>, compare_Point> supply;
                zero_condensation(diagramA, diagramB, massA, massB, supply, infinity_cost);
                //check for equality of diagrams
                if (diagramA.size() == diagramB.size()) {
                    eq_points eq;
                    bool equal = std::equal(diagramA.begin(), diagramA.end(), diagramB.begin(), eq);
                    if (equal) {
                        return 0.0 + infinity_cost;
                    }
                }

                if (diagramA.size() == 0) {
                    for (Point p : diagramB) {
                        cost += (p.y - p.x) / 2.0;
                    }
                    return infinity_cost + cost;
                }
                if (diagramB.size() == 0) {
                    for (Point p : diagramA) {
                        cost += (p.y - p.x) / 2.0;
                    }
                    return infinity_cost + cost;
                }
                //FIND LOWER BOUND then apply delta condensation:
                //three ways of finding lower bound:
                //1. http://proceedings.mlr.press/v37/kusnerb15.pdf (relaxed WMD)
                //2. https://web.eecs.umich.edu/~pettie/matching/Agarwal-Varadarajan-bichromatic-euclidean-mwpm-approx.pdf (MST based solution won't work for unbalanced optimal transport)
                //3. https://arxiv.org/pdf/1812.02091. (e.g. ICT: slow)
                //note that 2. requires the L2 metric!!
                //take the max over all lower bounds

                //1:
#ifdef PROFILING
                std::cout << "RELAXED WMD being computed\n";
#endif
                real_t lower_bound = relaxed_WMD_kdtree(diagramA, diagramB, massA, massB);

//                lower_bound = relaxed_WMD_gpu(diagramA, diagramB, massA, massB);
//                std::cout << (lower_bound) << std::endl;
                sw.stop();
#ifdef PROFILING
                std::cout << "time for relaxed WMD: " << sw.ms() / 1000.0 << "s" << std::endl;
#endif
#ifdef VERBOSE
                std::cout << "lower_bound: " << lower_bound << std::endl;
#endif
                sw.start();
                //lower_bound = relaxed_WMD_cpu(diagramA, diagramB, massA, massB);

                real_t epsilon = 0.9999;//2*1.414-1;//0.9999;//2*1.414-1;//log2(diagramA.size()+diagramB.size())*log2(diagramA.size()+diagramB.size());
                if (s > 12) {
                    epsilon = std::min((real_t) epsilon, (real_t) (8.0 / (s - 4.0)));
                }
#ifdef DEBUG
                std::cout << "epsilon: " << epsilon << std::endl;
#endif
                //replace zerocondense_points with delta condensation where delta=epsilon*lower_bound
                delta_condense_points_staticgraph(diagramA, diagramB, massA, massB,
                                                  epsilon, lower_bound, points, mass, massB.back(), massA.back(),
                                                  diagonalEdges, supply);
                sw.stop();
#ifdef PROFILING
                std::cout << "delta condense points time: " << sw.ms() / 1000.0 << std::endl;
#endif
            } else {

                zerocondense_points_staticgraph(diagramA, diagramB, points, //nodes,
                                                massA, massB, mass, diagonalEdges, infinity_cost);

                sw.stop();
#ifdef PROFILING
                std::cout << "zero condense points time: " << sw.ms() / 1000.0 << std::endl;
#endif
            }

#ifdef DEBUG
            for (int i = 0; i < points.size(); i++) {
                printf("%lf %lf, vertice: %ld\n", points[i].x, points[i].y, points[i].vertice);
            }
#endif
            if (points.size() == 0) {
                return 0.0 + infinity_cost;
            }
            if (diagramA.size() == diagramB.size()) {
                eq_points eq;
                bool equal = std::equal(diagramA.begin(), diagramA.end(), diagramB.begin(), eq);
                if (equal) {
                    return 0.0 + infinity_cost;
                }
            }

            if (diagramA.size() == 0) {
                for (Point p : diagramB) {
                    cost += (p.y - p.x) / 2.0;
                }
                return infinity_cost + cost;
            }
            if (diagramB.size() == 0) {
                for (Point p : diagramA) {
                    cost += (p.y - p.x) / 2.0;
                }
                return infinity_cost + cost;
            }
#ifdef DEBUG
            for (int i = 0; i < points.size(); i++) {
                std::cerr << i << "," << points[i].x << "," << points[i].y << std::endl;
            }
#endif
            compare_Point cmp;
            auto tree = SplitTree(points, Rectangle(points));
            sw.stop();
#ifdef PROFILING
            std::cout << "time to form Split Tree: " << std::setprecision(15) << sw.ms() / 1000.0 << "s" << std::endl;
#endif
            sw.start();
            std::vector<int_pair> arcs;
            lemon::StaticDigraph static_g;
            //should prevent reserving too much memory if s is large
            arcs.reserve(points.size() * (int) (s * s));//should be num_points/eps^2 where eps is approx factor
            wspd_parallel_write_construction(tree, s, arcs);

            sw.stop();
#ifdef PROFILING
            std::cout << "time to form wspd: " << std::setprecision(15) << sw.ms() / 1000.0 << "s" << std::endl;
#endif

            //setup the nondiagonal subgraph arcs
            sw.start();

//why not just randomly put in arcs??!!? (this will not give anything accurate whatsoever)

            Stopwatch innersw;
            innersw.start();
            int num_arcs = arcs.size();

            int_pair *d_arcs;
            int *d_offsets;
            CUDACHECK(
                    cudaMalloc((void **) &d_arcs, sizeof(int_pair) * (arcs.size() * 2 + 2 * (diagonalEdges.size())+1)));
            //perhaps we can build wspd on GPU?: would need a morton encoding and a change in traversal order
            cudaMemcpy(d_arcs, &(arcs[0]), sizeof(int_pair) * arcs.size(), cudaMemcpyHostToDevice);

            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            int grid_size;
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size, copy_reverse_edges, 256, 0));
            grid_size *= deviceProp.multiProcessorCount;
            copy_reverse_edges<<<grid_size, 256>>>(d_arcs, arcs.size());
            CUDACHECK(cudaDeviceSynchronize());
            num_arcs = arcs.size() * 2;//we know this from the copy_reverse_edges kernel

            int num_nondiagonal_arcs = num_arcs;

            innersw.stop();
#ifdef PROFILING
            std::cout << "(part of time to set up min-cost flow graph) form biarcs from edges construction time: "
                      << innersw.ms() / 1000.0 << "s" << std::endl;
#endif
            //build the diagonal subgraph arcs by adding in the arcs from Bbar to B and A to Abar and the 0 cost edge from A to B

            innersw.start();
            //last two nodes are Abar and Bbar
            int num_nodes = mass.size();
            int Bbar = num_nodes - 2;
            int Abar = num_nodes - 1;

            int num_nondiagonal_nodes_w_diagonal_edges = diagonalEdges.size();
            int *d_diagonalArcs;//indexed by node number (not involving the last two diagonal node indices)s
            int *d_prefixsum_pointindices;
            CUDACHECK(cudaMalloc((void **) &d_diagonalArcs, sizeof(int) * (num_nondiagonal_nodes_w_diagonal_edges)));
            CUDACHECK(cudaMalloc((void **) &d_prefixsum_pointindices,
                                 sizeof(int) * (num_nondiagonal_nodes_w_diagonal_edges + 1)));

            cudaMemcpy(d_diagonalArcs, &(diagonalEdges[0]), sizeof(int) * diagonalEdges.size(),
                       cudaMemcpyHostToDevice);
            //there is a +1 to account for the arc from Bbar to Abar
            cudaMemset(d_prefixsum_pointindices, 0, sizeof(int) * (diagonalEdges.size() + 1));
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size, transform_to_count_kernel, 256, 0));
            grid_size *= deviceProp.multiProcessorCount;
            transform_to_count_kernel<<<grid_size, 256>>>(d_diagonalArcs, d_prefixsum_pointindices,
                                                          diagonalEdges.size());
            CUDACHECK(cudaDeviceSynchronize());
            int num_diagonal_arcs = 1 + thrust::reduce(thrust::device, d_prefixsum_pointindices,
                                                        d_prefixsum_pointindices + diagonalEdges.size());
#ifdef COUNTING
            std::cout << "num_diagonal_arcs: " << num_diagonal_arcs << std::endl;
            std::cout<<"total num arcs: "<<num_arcs+num_diagonal_arcs<<std::endl;
#endif
            thrust::plus<int> binary_plus;
            thrust::exclusive_scan(thrust::device, d_prefixsum_pointindices,
                                   d_prefixsum_pointindices + diagonalEdges.size(), d_prefixsum_pointindices);

            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size, gather_diagonal_edges, 256, 0));
            grid_size *= deviceProp.multiProcessorCount;
            gather_diagonal_edges<<<grid_size, 256>>>(d_arcs, d_diagonalArcs, d_prefixsum_pointindices,
                                                      num_nondiagonal_arcs, num_diagonal_arcs, diagonalEdges.size(),
                                                      Abar, Bbar);
            CUDACHECK(cudaDeviceSynchronize());

            arcs.resize(num_nondiagonal_arcs + num_diagonal_arcs);
            cudaMemcpy(&(arcs[0]), d_arcs, sizeof(int_pair) * (num_nondiagonal_arcs + num_diagonal_arcs),
                       cudaMemcpyDeviceToHost);
            innersw.stop();
#ifdef PROFILING
            std::cout << "(part of time to set up min-cost flow graph) gather diagonal arcs: " << innersw.ms() / 1000.0
                      << "s" << std::endl;
#endif
            innersw.start();
            assert(diagonalEdges.size() == num_nodes - 2);

            assert(points.size() == diagonalEdges.size());

            num_arcs = num_diagonal_arcs + num_nondiagonal_arcs;
            compare_int_pair cmp_int_pair;

            thrust::sort(thrust::device, d_arcs, d_arcs + num_nondiagonal_arcs + num_diagonal_arcs,
                             cmp_int_pair);

            int_pair* d_common_points;
            CUDACHECK(cudaMalloc((void**) &d_offsets, sizeof(int)*num_nodes));
            CUDACHECK(cudaMalloc((void**) &d_common_points, sizeof(int_pair)*num_nodes));

            cudaMemset(d_offsets, 0, sizeof(int)*num_nodes);

            //count (u,v) pairs with common u index
            equality_int_pair_first eq_int_pair_first;
            auto offsets_end= thrust::reduce_by_key(thrust::device, d_arcs, d_arcs+ num_arcs,  thrust::make_constant_iterator(1), d_common_points, d_offsets, eq_int_pair_first);
            assert(offsets_end.second-d_offsets+1<= num_nodes);

            //prefix sum
            thrust::exclusive_scan(thrust::device, d_offsets, d_offsets+num_nodes, d_offsets,0);

            arcs.resize(num_nondiagonal_arcs + num_diagonal_arcs);
            cudaMemcpy(&(arcs[0]), d_arcs, sizeof(int_pair) * (num_nondiagonal_arcs + num_diagonal_arcs),
                       cudaMemcpyDeviceToHost);
            std::vector<int> h_offsets(num_nodes);
            cudaMemcpy(&(h_offsets[0]), d_offsets, sizeof(int) *num_nodes, cudaMemcpyDeviceToHost);

            h_offsets.push_back(num_arcs);

            /*for(int i=0; i<h_offsets.size(); i++){
                std::cout<<h_offsets[i]<<std::endl;
            }*/

            static_g.clear();
            //static_g.build(num_nodes, edges.begin(), edges.end());
            static_g.build_parallel(num_nodes, arcs, h_offsets );
            innersw.stop();
#ifdef PROFILING
            std::cout << "(part of time to set up min-cost flow graph) sort then build graph " << innersw.ms() / 1000.0 << "s"<<std::endl;
#endif
#ifdef DEBUG
            for (int i = 0; i < static_g.arcNum(); i++) {
                printf("all edge enumerate: %d %d\n", static_g.source(static_g.arcFromId(i)),
                       static_g.target(static_g.arcFromId(i)));
            }
#endif
//old way of doing things... results in ~1.2x slowdown
//            lemon::StaticDigraph::ArcMap<real_t> costs(static_g);
//            lemon::StaticDigraph::ArcMap<int> capacities(static_g);
//            lemon::StaticDigraph::NodeMap<int> supplies(static_g);
            using NS = lemon::NetworkSimplex<lemon::StaticDigraph, int, real_t>;
            std::vector<int> _reindex;//this is used for arc mixing reindexing from lemon
            NS ns(_reindex,static_g);

            ns.s= s;

            assert(num_arcs == static_g.arcNum());
            innersw.start();
#pragma omp parallel for schedule(static, 1)
            for (int i = 0; i < static_g.arcNum(); i++) {
                auto k = static_g.arcFromId(i);
                auto uidx = static_g.id(static_g.source(k));
                auto vidx = static_g.id(static_g.target(k));
                if (vidx == Abar && uidx == Bbar) {
                    ns._cost[_reindex[i]] = 0.0;
                } else if (vidx == Abar) {
                    //ns._cost[_reindex[i]] = std::abs(points[uidx].y - points[uidx].x) / 2.0;
                    ns._cost[_reindex[i]] = points[uidx].get_l2_dist_to_diag();
                } else if (uidx == Bbar) {
                    //ns._cost[_reindex[i]] = std::abs(points[vidx].y - points[vidx].x) / 2.0;
                    //ns._cost[_reindex[i]] = std::abs(points[vidx].y - points[vidx].x) / 2.0;
                    ns._cost[_reindex[i]] = points[vidx].get_l2_dist_to_diag();
                } else {
//                    ns._cost[_reindex[i]] = points[uidx].dist_linfty(
//                            points[vidx]);
                    ns._cost[_reindex[i]] = points[uidx].dist_l2(
                            points[vidx]);
                }
            }
            innersw.stop();
#ifdef PROFILING
            std::cout<<"(part of time to set up min-cost flow graph) time to set up costs and capacities "<<innersw.ms()/1000.0<<"s"<<std::endl;
#endif
#ifdef DEBUG
            /*for (lemon::StaticDigraph::ArcIt e(static_g); e != lemon::INVALID; ++e) {
                printf("edge: %d , src: %d, target: %d, cost: %f, cap: %d\n", static_g.id(e), static_g.source(e),
                       static_g.target(e), costs[e], capacities[e]);
            }
            for (lemon::StaticDigraph::NodeIt i(static_g); i != lemon::INVALID; ++i) {
                printf("node: %d supplies: %d\n", static_g.id(i), supplies[i]);
            }*/
#endif
            innersw.start();
#pragma omp parallel for schedule(static,1)
            for(int i=0; i<mass.size(); i++) {
                ns._supply[num_nodes-1-i]= mass[i];//reverse as part of node order in lemon
            }

#pragma omp parallel for schedule(static,1)
            for(int a=0; a<num_arcs;a++) {
                ns._upper[a] = massA.back()+massB.back();//std::numeric_limits<int>::max();
            }
            //old way of populating the static graph
            //ns.costMap(costs).upperMap(capacities).supplyMap(supplies);
            //ns.costMap(costs).supplyMap(supplies);//the upper are automatically set to INF inside costMap()
            //ns.costMap(costs).supplyMap(supplies);//the upper are automatically set to INF inside costMap()

            innersw.stop();
#ifdef PROFILING
            std::cout<<"(part of time to set up min-cost flow graph) set the costs, capacities and supplies: "<<innersw.ms()/1000.0<<"s"<<std::endl;
#endif
            sw.stop();
#ifdef PROFILING
            std::cout << "time to set up min-cost flow graph " << sw.ms() / 1000.0 << std::endl;
#endif

#ifdef COUNTING
            assert(countNodes(static_g)==num_nodes);
            std::cout << "We have a directed graph with " << countNodes(static_g) << " nodes "
                      << "and " << countArcs(static_g) << " arcs" << std::endl;
#endif
#ifdef MIN_VERBOSE
            std::cout << "We have a directed graph with " << countNodes(static_g) << " nodes "
                      << "and " << countArcs(static_g) << " arcs" << std::endl;
#endif
#ifdef MIN_VERBOSE
            std::cout<<"Starting network simplex algorithm\n";
#endif
            auto start = std::chrono::high_resolution_clock::now();
            NS::ProblemType status = ns.run(ADDITIVE_COEFF,MULTIPLICATIVE_COEFF);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

#ifdef PROFILING
            std::cout << "TIME TO DO MINCOST FLOW "<<std::setprecision(15)<< (duration.count()) / 1000.0 / 1000.0<<"s"<<std::endl;
#endif
            switch (status) {
                case NS::INFEASIBLE:
                    std::cerr << "insufficient flow" << std::endl;
                    break;
                case NS::OPTIMAL:
                    cost = ns.totalCost();
                    //std::cout << std::setprecision(15) << "noninfinity cost computed by lemon=" << cost << std::endl;
                    break;
                case NS::UNBOUNDED:
                    std::cerr << "infinite flow" << std::endl;
                    break;
                default:
                    break;
            }

            cudaFree(d_arcs);
            cudaFree(d_prefixsum_pointindices);
            cudaFree(d_offsets);
            cudaFree(d_diagonalArcs);
            cudaFree(d_common_points);
            return infinity_cost + cost;
        }

        real_t launch_check_matching(std::string f1, std::string f2, real_t s, std::vector<std::pair<Point,Point>> matching){

            Stopwatch sw;
            sw.start();

            lemon::SmartDigraph mygraph;
            std::vector<Point> diagramA;
            std::vector<Point> diagramB;
            int decPrecision = 0;
            if (!hera::readDiagramPointSet(f1, diagramA, decPrecision)) {
                exit(1);
            }
            if (!hera::readDiagramPointSet(f2, diagramB, decPrecision)) {
                exit(1);
            }


            std::vector<Point> points;
            std::vector<int> massA;
            std::vector<int> massB;
            std::vector<int> Aindex;
            std::vector<int> Bindex;

            std::vector<int> mass;
            std::vector<int> diagonalEdges;
            sw.stop();

            std::cout << "time to load PDs: " << sw.ms() / 1000.0 <<
                      std::endl;
// build the nodes and mass vectors for the spanner graph
            sw.start();

            real_t infinity_cost = 0.0;
            zerocondense_points(diagramA, diagramB, points, //nodes,
                                massA, massB, mass, mygraph, diagonalEdges, infinity_cost
            );

            //try the delta-condensation algorithm see: https://web.eecs.umich.edu/~pettie/matching/Agarwal-Varadarajan-bichromatic-euclidean-mwpm-approx.pdf lemma 2.2 alpha give us a lower bound on the optimal matching value

//check for equality of diagrams
            std::cout << "zero condense points time: " << sw.ms()/ 1000.0 <<
                      std::endl;

#ifdef DEBUG
            for(int i=0; i<points.size(); i++){
                printf("%lf %lf, vertice: %ld\n", points[i].x, points[i].y, points[i].vertice);
            }
#endif

            auto tree = SplitTree(points, Rectangle(points));
            sw.stop();

            std::cout << "time to form Split Tree: " << sw.ms()/ 1000.0 << "s" <<
                      std::endl;

            sw.start();

            std::vector<int_pair> edges;
            assert(points.size()> 0);
            edges.reserve(points.size() * (int) (s * s));//should be num_points/eps^2 where eps is approx factor
            wspd_parallel_write_construction(tree, s, edges);

            sw.stop();

#ifdef PROFILING
            std::cout << "time to form wspd: " << sw.ms() / 1000.0 << std::endl;
#endif

//setup the nondiagonal subgraph edges

            sw.start();

            for (auto pair: edges) {
                auto left = pair.first;
                auto right = pair.second;
                mygraph.addArc(mygraph.nodeFromId(left), mygraph.nodeFromId(right));
                mygraph.addArc(mygraph.nodeFromId(right), mygraph.nodeFromId(left));
            }

//try the full bipartite graph and check the answer, also try the complete graph on nodes (without nodes.size()-2 and nodes.size()-1)
/*
    for(int i=0; i<(nodes.size()-2); i++){
        for(int j=0; j<nodes.size()-2; j++) {
            if(i!=j){
                mygraph.addArc(nodes[i], nodes[j]);
            }
        }
    }
*/
            lemon::SmartDigraph::ArcMap<real_t> costs(mygraph);
            lemon::SmartDigraph::ArcMap<int> capacities(mygraph);
            lemon::SmartDigraph::NodeMap<int> supplies(mygraph);
            int j = 0;

            for (lemon::SmartDigraph::NodeIt i(mygraph);i !=lemon::INVALID;
                 ++i) {
                int k = mygraph.id(i);
                supplies[i] = mass[k];
                j++;
            }
            assert(j== mygraph.maxNodeId()+ 1);

            j = 0;
            for (lemon::SmartDigraph::ArcIt i(mygraph);
                 i !=lemon::INVALID;
                 ++i) {
                capacities[i] =std::numeric_limits<int>::max();

                lemon::SmartDigraphBase::Node u = mygraph.source(i);
                lemon::SmartDigraphBase::Node v = mygraph.target(i);
#ifdef DEBUG
                printf("edge: %d %d\n", mygraph.id(u),mygraph.id(v));
#endif
                costs[i] = points[mygraph.id(u)].
                        dist_l2(
                        //dist_linfty(
                        points[mygraph.id(v)]);// this needs to be the actual distance between
//costs[i]= points[mygraph.id(u)].dist_linfty(points[mygraph.id(v)]);
//for q=2: (doesn't converge)
///costs[i]*= costs[i];
                j++;
            }

//build the diagonal subgraph edges by adding in the arcs from Bbar to B and A to Abar and the 0 cost edge from A to B

//last two nodes are Abar and Bbar
            int Bbar = mygraph.maxNodeId() - 1;//nodes.size()-2;
            int Abar = mygraph.maxNodeId();//nodes.size()-1;

            assert(mygraph.maxNodeId()+ 1 == mass.size());
            assert(diagonalEdges.size()== mass.size()- 2);
            for (int i = 0;i < diagonalEdges.size();i++) {
                if (diagonalEdges[i] == -1) {
#ifdef DEBUG
                    printf("node: %d has incoming edge\n", i);
#endif
                    auto e = mygraph.addArc(mygraph.nodeFromId(Bbar), mygraph.nodeFromId(i));
                    capacities[e] = std::numeric_limits<int>::max();

//                    costs[e] = std::abs(points[i].y - points[i].x) / 2.0;
                    costs[e] = points[i].get_l2_dist_to_diag();//std::abs(points[i].y - points[i].x) / 2.0;
//for q=2: (doesn't converge)
//costs[e]*= costs[e];
                } else if (diagonalEdges[i] == 1) {
#ifdef DEBUG
                    printf("node: %d has outgoing edge\n", i);
#endif

                    auto e = mygraph.addArc(mygraph.nodeFromId(i), mygraph.nodeFromId(Abar));
                    capacities[e] =std::numeric_limits<int>::max();

//                    costs[e] =std::abs(points[i].y - points[i].x) / 2.0;
                    costs[e] =points[i].get_l2_dist_to_diag();//std::abs(points[i].y - points[i].x) / 2.0;
//for q=2: (doesn't converge)
//costs[e]*= costs[e];
                } else {
#ifdef DEBUG
                    printf("node: %d has incoming and outgoing edge\n", i);
#endif
                    auto e = mygraph.addArc(mygraph.nodeFromId(Bbar), mygraph.nodeFromId(i));
                    capacities[e] =std::numeric_limits<int>::max();

//                    costs[e] = std::abs(points[i].y - points[i].x) / 2.0;
                    costs[e] =points[i].get_l2_dist_to_diag(); //std::abs(points[i].y - points[i].x) / 2.0;
//for q=2: (doesn't converge)
//costs[e]*= costs[e];
                    auto f = mygraph.addArc(mygraph.nodeFromId(i), mygraph.nodeFromId(Abar));

                    capacities[f] = std::numeric_limits<int>::max();

//                    costs[f] = std::abs(points[i].y - points[i].x) / 2.0;
                    costs[f] = points[i].get_l2_dist_to_diag();//std::abs(points[i].y - points[i].x) / 2.0;

//for q=2: (doesn't converge)
//costs[f]*= costs[f];
                }
            }
            auto e = mygraph.addArc(mygraph.nodeFromId(Bbar), mygraph.nodeFromId(Abar));

            capacities[e] = std::numeric_limits<int>::max();

            costs[e] = 0;
#ifdef DEBUG
            for (lemon::SmartDigraph::ArcIt e(mygraph); e!=lemon::INVALID; ++e) {
                printf("edge: %d , src: %d, target: %d, cost: %f, cap: %d\n", mygraph.id(e), mygraph.source(e), mygraph.target(e), costs[e], capacities[e]);
            }
            for (lemon::SmartDigraph::NodeIt i(mygraph); i!=lemon::INVALID; ++i) {
                printf("node: %d supplies: %d\n", mygraph.id(i), supplies[i]);
            }
#endif
            using NS = lemon::NetworkSimplex<lemon::SmartDigraph, int, real_t>;
            std::vector<int> reindex;
            NS ns(reindex, mygraph);
            ns.costMap(costs).upperMap(capacities).supplyMap(supplies);//.stSupply(nodes[1], nodes[2], 10);
            lemon::SmartDigraph::ArcMap<int> flows(mygraph);
            sw.stop();

#ifdef PROFILING
            std::cout << "time to set up min-cost flow graph " << sw.ms() / 1000.0 << "s"<<std::endl;
#endif
            printf("check, Abar: %d\n", Abar);
            printf("check, Bbar: %d\n", Bbar);
            auto start = std::chrono::high_resolution_clock::now();
            NS::ProblemType status = ns.run(250000,70);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

            real_t cost = 0;
            real_t original_cost= 0.0;
            int diagonal_flows = 0;
            int total_flows = 0;
            int diagonal_edges_wflow = 0;
            int correct_num_diag_matchings = 0;
            int correct_num_nondiag_matchings = 0;
            int total_supply = 0;
            real_t correct_cost = 0.0;
            switch (status) {
                case NS::INFEASIBLE:
                    std::cerr << "insufficient flow" <<
                              std::endl;
                    break;
                case NS::OPTIMAL:
                    ns.flowMap(flows);

                    for (lemon::SmartDigraph::NodeIt i(mygraph);i !=lemon::INVALID;++i) {
                        if (supplies[i] > 0) {
                            total_supply += supplies[i];
                        }
                    }
                    for (lemon::SmartDigraph::ArcIt i(mygraph);i !=lemon::INVALID;++i) {

                        cost += costs[i] * ns.flow(i);

                        //std::cout<<"flow arc: "<<points[mygraph.id(mygraph.source(i))].x<<" "<<points[mygraph.id(mygraph.source(i))].y<<" "<<points[mygraph.id(mygraph.target(i))].x<<" "<<points[mygraph.id(mygraph.target(i))].y<<std::endl;
                        //
                        if (ns.flow(i) > 0 &&
                            (mygraph.id(mygraph.target(i)) == Abar) &&
                            !(mygraph.id(mygraph.source(i)) == Bbar && mygraph.id(mygraph.target(i)) == Abar)) {
                            diagonal_flows += ns.flow(i);
                            diagonal_edges_wflow++;
                            int count = 0;
                            for (auto pq : matching) {
                                if ((pq.second.x == pq.second.y
                                     && eps_close(points[mygraph.id(mygraph.source(i))].x, pq.first.x)
                                     && eps_close(points[mygraph.id(mygraph.source(i))].y, pq.first.y))
                                        ) {
                                    count++;
                                }
                            }
                            correct_num_diag_matchings += std::min(ns.flow(i), count);
                            correct_cost += costs[i] * std::min(ns.flow(i), count);
                            original_cost += costs[i] * count;
                        }
                        else if(ns.flow(i)>0 && (mygraph.id(mygraph.source(i)) == Bbar) &&
                           !(mygraph.id(mygraph.source(i)) == Bbar && mygraph.id(mygraph.target(i)) == Abar)) {
                            diagonal_flows += ns.flow(i);
                            diagonal_edges_wflow++;
                            int count = 0;
                            for (auto pq : matching) {
                                if ((eps_close(pq.first.x, pq.first.y)
                                     && eps_close(pq.second.x, points[mygraph.id(mygraph.target(i))].x)
                                     && eps_close(pq.second.y,points[mygraph.id(mygraph.target(i))].y))) {
                                    count++;
                                }
                            }
                            correct_num_diag_matchings += std::min(ns.flow(i), count);
                            correct_cost += costs[i] * std::min(ns.flow(i), count);
                            original_cost += costs[i] * count;
                        }
                        else if (ns.flow(i) > 0 && mygraph.id(mygraph.source(i)) != Bbar &&
                            mygraph.id(mygraph.target(i)) != Abar) {
                            int count = 0;
                            for (auto pq : matching) {
                                if ((eps_close(pq.first.x, points[mygraph.id(mygraph.source(i))].x)
                                     && eps_close(pq.first.y,points[mygraph.id(mygraph.source(i))].y)
                                     && eps_close(pq.second.x,points[mygraph.id(mygraph.target(i))].x)
                                     && eps_close(pq.second.y,points[mygraph.id(mygraph.target(i))].y))

                                     //we shouldn't have to check these if the order of the datasets is correct
//                                    || (pq.second.x == points[mygraph.id(mygraph.source(i))].x
//                                        && pq.second.y == points[mygraph.id(mygraph.source(i))].y
//                                        && pq.first.x == points[mygraph.id(mygraph.target(i))].x
//                                        && pq.first.y == points[mygraph.id(mygraph.target(i))].y)
                                        ) {
                                    count++;
                                }
                            }
                            correct_num_nondiag_matchings += std::min(ns.flow(i), count);
                            correct_cost += costs[i] * std::min(ns.flow(i), count);
                            original_cost += costs[i] * count;
                        }
                        if (ns.flow(i) > 0) {
                            total_flows += ns.flow(
                                    i);//this doesn't really say anything since it is total flow, not the number of matchings
                            //printf("all edges: %d %d \n", mygraph.id(mygraph.source(i)) , mygraph.id(mygraph.target(i)));
                        }
                    }

                    printf("CORRECT DIAG matchings: %d\n", correct_num_diag_matchings);
                    printf("CORRECT NONDIAG matchings: %d\n", correct_num_nondiag_matchings);
                    printf("COST from EXACTLY CORRECT MATCHINGS: %lf\n", correct_cost);

                    printf("ORIGINAL COST: %lf\n", original_cost);
                    printf("diagonal_flows not including the diagonal to diagonal flow<=total supply= number of matchings total: %d\n",
                           diagonal_flows);

                    printf("total supply= total demand: %d\n", total_supply);
                    printf("total_flows: %d\n",
                           total_flows);//this can be bigger than the total supply+total demand. Consider following example: 3 points in A, 3 points in B exactly the same and on a straight line, now route flow from one point through another point to the third point. This will involve more total flow than matchings in the bipartite matching case.
                    printf("diagonal edges with flow count: %d\n", diagonal_edges_wflow);
                    std::cout << std::setprecision(15) << "finite points manual computed cost= " << cost <<
                              std::endl;
//            std::cout<< std::setprecision(15)<<"manual computed cost= "<<std::sqrt(cost)<<std::endl;
                    std::cout << std::setprecision(15) << "finite points cost computed by lemon=" << ns.totalCost()

                              <<
                              std::endl;
//                std:cout<<"k: "<<correct_cost
//            std::cout << std::setprecision(15)<< "cost computed by lemon=" << std::sqrt(ns.totalCost()) << std::endl;

                    break;
                case NS::UNBOUNDED:
                    std::cerr << "infinite flow" <<
                              std::endl;
                    break;
                default:
                    break;
            }
            std::cout << "We have a directed graph with " <<
                      countNodes(mygraph)
                      << " nodes "
                      << "and " <<
                      countArcs(mygraph)
                      << " arc." <<
                      std::endl;
            std::cout << std::endl <<
                      std::endl;
            return infinity_cost +
                   cost;
        }

        real_t launch_load_matching( std::vector<std::pair<Point,Point>>& matching) {
            while(true){
                Point a, b;
                std::cin >> a.x >> a.y >> b.x >> b.y;
                //std::cout<<std::setprecision(15)<<a.x<<" "<<a.y<<" "<<b.x<<" "<<b.y<<std::endl;
                if(a.x==-1 && a.y==-1 && b.x==-1 && b.y==-1){
                    break;
                }
                //matching.push_back(std::make_pair(a,b));
                matching.emplace_back(a,b);
            }
            real_t cost= 0.0;
            std::cout<<"matching.size() "<<matching.size()<<std::endl;
            for(int i=0; i<matching.size(); i++){
                if(matching[i].first.x != matching[i].first.y
                || matching[i].second.x != matching[i].second.y)
                cost+= matching[i].first.dist_l2(matching[i].second);
            }
            return cost;
        }
    }

////////////////////////////////////
//interface functions to run W1

    double compute_RWMD(std::vector<PDoptFlow::W1::Point> &diagramA, std::vector<PDoptFlow::W1::Point> &diagramB) {
        return (double) PDoptFlow::compute_RWMD(diagramA, diagramB);
    }

    double compute_centroid_dist(std::vector<PDoptFlow::W1::Point> &diagramA, std::vector<PDoptFlow::W1::Point> &diagramB){
        return (double)PDoptFlow::compute_centroid_dist(diagramA,diagramB);
    }
    //if delta_condense is changed from true to false, then a new set of ADDITIVE_COEFF and MULTIPLICATIVE_COEFF hyperparameters (larger) must be chosen
    //keep delta_condense as true if you do not want to change hyper parameters
    double
    wasserstein_dist(std::vector<PDoptFlow::W1::Point> diagramA, std::vector<PDoptFlow::W1::Point> diagramB,
                     double s, bool delta_condense, int additive, int multiplicative) {
        return (double)PDoptFlow::W1::compute_wasserstein_dist(diagramA, diagramB, s, delta_condense, additive, multiplicative);
    }

    double wasserstein_dist_fromfile(std::string f1, std::string f2, double s, bool delta_condense, int additive, int multiplicative) {
        std::vector<W1::Point> diagramA;
        std::vector<W1::Point> diagramB;
        int decPrecision = 0;
        if (!PDoptFlow::hera::readDiagramPointSet(f1, diagramA, decPrecision)) {
            exit(1);
        }
        if (!PDoptFlow::hera::readDiagramPointSet(f2, diagramB, decPrecision)) {
            exit(1);
        }
        return (double)W1::compute_wasserstein_dist(diagramA, diagramB, s, delta_condense, additive, multiplicative);
    }

    std::vector<PDoptFlow::W1::Point> check_points;

//these are for the k-experiments. WILL POSSIBLY REMOVE IN LATER VERSION
    /*
    double wasserstein_fullbipartite_generate_stats(std::string f1, std::string f2,
                                                    std::vector<std::pair<std::pair<int, int>, int>> &diag_pairs,
                                                    std::vector<std::pair<std::pair<int, int>, int>> &nondiag_pairs, std::vector<PDoptFlow::W1::Point> indextopoint) {
        return (double)W1::launch_wspd_fullbipartite_generate_stats_COMPATIBLE(f1, f2, diag_pairs, nondiag_pairs, indextopoint);
    }
     */

    double load_matching(std::string f3, std::vector<std::pair<PDoptFlow::W1::Point, PDoptFlow::W1::Point>>& matching) {
        //ignore f3 and read from stdin for now
        return (double)W1::launch_load_matching(matching);
    }

    double wasserstein_check_matching(std::string f1, std::string f2, double s, std::vector<std::pair<PDoptFlow::W1::Point, PDoptFlow::W1::Point>>& matching) {
        return (double) W1::launch_check_matching(f1, f2, s,matching);
    }
/*
double wasserstein_check_matching(std::string f1, std::string f2, double s,
                               std::vector<std::pair<std::pair<int, int>, int>> &diag_pairs,
                               std::vector<std::pair<std::pair<int, int>, int>> &nondiag_pairs, std::vector<PDoptFlow::W1::Point> indextopoint) {
    return (double)W1::launch_check_stats(f1, f2, s, diag_pairs, nondiag_pairs, indextopoint);
}*/
}
