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

#include <spanner/split_tree.hh>

namespace PDoptFlow {
    namespace W1 {
        bool is_well_separated(Node left, Node right, real_t s) {

            auto radius_l = left.bounding_box.get_l2_radius();
            auto radius_r = right.bounding_box.get_l2_radius();
            //auto radius_l = left.bounding_box.get_linfty_radius();
            //auto radius_r = right.bounding_box.get_linfty_radius();

            auto center_l = left.bounding_box.get_center();
            auto center_r = right.bounding_box.get_center();

            real_t dist = center_l.dist_l2(center_r) - (radius_l + radius_r);

            //real_t dist = center_l.dist_linfty(center_r) - (radius_l + radius_r);

            auto left_top_left = left.bounding_box.top_left;
            auto left_bottom_right = left.bounding_box.bottom_right;
            auto right_top_left = right.bounding_box.top_left;
            auto right_bottom_right = right.bounding_box.bottom_right;

            return dist >= s * std::max(radius_l, radius_r);//*2.0;
            //return dist*dist >= s * std::max(radius_l*radius_l, radius_r*radius_r)*4.0;//for l2^2 "distance"
        }

        void wspd_rec_par_count(int i, Node u, Node v, real_t s, std::vector<int> &count) {
            if (u.left == nullptr && u.right == nullptr && v.left == nullptr &&
                v.right == nullptr) {//a pair of leaves is always well separated
                count[i]++;
                return;
            }
            if (is_well_separated(u, v, s)) {
                count[i]++;
                return;

            }
            if (u.bounding_box.get_max_length() <= v.bounding_box.get_max_length()) {
                if (!v.left || !v.right)
                    return;

                auto right_l = v.left;
                auto right_r = v.right;

                wspd_rec_par_count(i, u, *right_l, s, count);
                wspd_rec_par_count(i, u, *right_r, s, count);
            } else {
                if (!u.left || !u.right)
                    return;

                auto left_l = u.left;
                auto left_r = u.right;

                wspd_rec_par_count(i, *left_l, v, s, count);
                wspd_rec_par_count(i, *left_r, v, s, count);
            }
        }

        void
        wspd_rec_par_write(int i, Node u, Node v, real_t s, std::vector<int> &offsets, std::vector <int_pair> &wspd) {
            if (u.left == nullptr && u.right == nullptr && v.left == nullptr &&
                v.right == nullptr) {
                wspd[offsets[i]++] = {u.vertex_repr, v.vertex_repr};
                return;
            }
            if (is_well_separated(u, v, s)) {
                wspd[offsets[i]++] = {u.vertex_repr, v.vertex_repr};
                return;
            }
            if (u.bounding_box.get_max_length() <= v.bounding_box.get_max_length()) {
                if (!v.left || !v.right)
                    return;

                auto right_l = v.left;
                auto right_r = v.right;

                wspd_rec_par_write(i, u, *right_l, s, offsets, wspd);
                wspd_rec_par_write(i, u, *right_r, s, offsets, wspd);
            } else {
                if (!u.left || !u.right)
                    return;

                auto left_l = u.left;
                auto left_r = u.right;

                wspd_rec_par_write(i, *left_l, v, s, offsets, wspd);
                wspd_rec_par_write(i, *left_r, v, s, offsets, wspd);
            }
        }

        void wspd_parallel_write_construction(SplitTree tree, real_t s, std::vector <int_pair> &wspd) {

            Stopwatch sw;
            auto nodes = tree.get_all_nodes();
            int nr_nodes = nodes.size();
#ifdef COUNTING
            std::cout<<"num nodes for the split tree: "<< nr_nodes<<std::endl;
#endif
            sw.start();
            std::vector<int> counts(nr_nodes + 1, 0);
#pragma omp parallel for schedule(static, 1)
            for (int i = 0; i < nr_nodes; i++) {
                auto node = nodes[i];

                if (node->left == nullptr || node->right == nullptr)
                    continue;
                wspd_rec_par_count(i, *(node->left), *(node->right), s, counts);

            }
            sw.stop();
#ifdef PROFILING
            std::cout<<"(part of time to form wspd) search split tree for wspd time: "<< sw.ms()/1000.0<<"s"<<std::endl;
#endif
            sw.start();
            std::vector<int> offsets(nr_nodes + 1, 0);

            for (int i = 1; i < nr_nodes + 1; i++) {
                offsets[i] += offsets[i - 1] + counts[i - 1];
            }
            wspd.resize(offsets[nr_nodes]);
            sw.stop();
#ifdef PROFILING
            std::cout << "(part of time to form wspd): prefix sum time: " << std::setprecision(15)<<sw.ms() / 1000.0 << "s"<<std::endl;
#endif
            sw.start();
#pragma omp parallel for schedule(static, 1)
            for (int i = 0; i < nr_nodes; i++) {
                auto node = nodes[i];

                if (node->left == nullptr || node->right == nullptr)
                    continue;
                wspd_rec_par_write(i, *(node->left), *(node->right), s, offsets, wspd);
            }
            sw.stop();
#ifdef PROFILING
            std::cout<<"wspd.size(): "<<wspd.size()<<std::endl;
            std::cout<<"(part of time to form wspd) write wspd time: "<< sw.ms()/1000.0<<"s"<<std::endl;
#endif
        }
    }
}