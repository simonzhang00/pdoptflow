/**
Copyright 2020, 2021 Tamal Dey and Simon Zhang

Contributed by Simon Zhang

note: split tree implementation largely modified from https://github.com/reyreaud-l/parallel_spanner

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
#include <spanner/rectangle.hpp>
#include <omp.h>
#include <profiling/stopwatch.h>

namespace PDoptFlow {
    namespace W1 {
        class Node {
        public:
            Node() {};

            Node(Rectangle, int);

            Node(std::shared_ptr <Node>, std::shared_ptr <Node>, Rectangle, int);

            std::vector <std::shared_ptr<Node>> get_all_nodes();

            std::shared_ptr <Node> left;
            std::shared_ptr <Node> right;
            Rectangle bounding_box;
            int vertex_repr;
        };

        class SplitTree {
        public:
            SplitTree(std::vector <Point>, Rectangle);

            std::shared_ptr <Node> calc_tree(std::vector <Point>, Rectangle);//, Point);

            std::vector <std::shared_ptr<Node>> get_all_nodes();

            std::shared_ptr <Node> root;

            std::vector <std::vector<Point>> nodeLists;
        };

///////////////////////////////////////////////////////////////

        Node::Node(Rectangle rec, int rep) {
            left = nullptr;
            right = nullptr;
            bounding_box = rec;
            vertex_repr = rep;
        }

        Node::Node(std::shared_ptr <Node> l, std::shared_ptr <Node> r, Rectangle rec, int rep) {
            left = l;
            right = r;
            bounding_box = rec;
            vertex_repr = rep;
        }

        std::vector <std::shared_ptr<Node>> Node::get_all_nodes() {
            std::vector <std::shared_ptr<Node>> res;
            if (left && left->right != nullptr && left->left != nullptr) {
                res.push_back(left);
                auto son = left->get_all_nodes();
                res.insert(res.end(),
                           std::make_move_iterator(son.begin()),
                           std::make_move_iterator(son.end()));
            }
            if (right && right->left != nullptr && right->right != nullptr) {
                res.push_back(right);
                auto son = right->get_all_nodes();
                res.insert(res.end(),
                           std::make_move_iterator(son.begin()),
                           std::make_move_iterator(son.end()));
            }
            return res;
        }

        SplitTree::SplitTree(std::vector <Point> points, Rectangle rectangle) {
            assert(points.size() > 0);
            root = calc_tree(points, rectangle);
        }

        std::shared_ptr <Node>
        SplitTree::calc_tree(std::vector <Point> points, Rectangle rectangle)//, Point point_closest_to_center)
        {
            assert(!points.empty());
            Rectangle bounding_box(points);

            if (points.size() == 1)
                return std::make_shared<Node>(Node(bounding_box, points[0].vertice));
            bool horizontal = bounding_box.get_dim_max_length();//is it a horizontal line or vertical line?
//#define DEBUG
#ifdef DEBUG
            std::cout << "\n\n";
            printf("points.size(): %d\n", points.size());
            for(int i=0; i<points.size(); i++){
                std::cout<<points[i]<<std::endl;
            }
            std::cout << "Bounding box:\n" << bounding_box << std::endl;
            std::cout << "Split Line:\n" << horizontal << std::endl;
#endif
            //x intercept
            real_t intercept = (bounding_box.top_left.x + bounding_box.bottom_right.x) / 2.0;
            if (horizontal) {//y intercept
                intercept = (bounding_box.top_left.y + bounding_box.bottom_right.y) / 2.0;
            }
            auto box_pair = rectangle.split(intercept, horizontal);
            auto r1 = box_pair.first;
            auto r2 = box_pair.second;
            std::vector <Point> s1;
            std::vector <Point> s2;

            for (auto point : points) {
                if (r1.contains(point)) {
                    s1.push_back(point);
                } else if (r2.contains(point)) {
                    s2.push_back(point);
                }
            }
#ifdef DEBUG
            std::cout << "Left:\n" << r1 << std::endl;
            std::cout<<"Left num ptns size: "<<s1.size()<<std::endl;
            std::cout << "Right:\n" << r2 << std::endl;
            std::cout<<"Right num ptns size: "<<s2.size()<<std::endl;
#endif
            auto left = std::shared_ptr<Node>(nullptr);
            auto right = std::shared_ptr<Node>(nullptr);;

            assert(s1.size() > 0);
            assert(s2.size() > 0);

            left = calc_tree(s1, r1);//, s1_repr);
            right = calc_tree(s2, r2);//, s2_repr);
            return std::make_shared<Node>(
                    Node(left, right, bounding_box, points[0].vertice));
        }

        std::vector <std::shared_ptr<Node>> SplitTree::get_all_nodes() {
            std::vector <std::shared_ptr<Node>> res;
            res.push_back(root);
            auto son = root->get_all_nodes();
            res.insert(res.end(),
                       std::make_move_iterator(son.begin()),
                       std::make_move_iterator(son.end()));
            return res;
        }
    }
}