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

namespace PDoptFlow {
    namespace W1 {


        __host__ __device__
        struct int_pair {//use this to represent vertex pairs for edges in static graph
            int first;
            int second;
        };

        struct compare_int_pair {
            __host__ __device__

            bool operator()(const int_pair &p1, const int_pair &p2) {
                return p1.first < p2.first || (p1.first == p2.first && p1.second < p2.second);
            }
        };

        struct equality_int_pair_first {
            __host__ __device__

            bool operator()(const int_pair &p1, const int_pair &p2) {
                return p1.first == p2.first;
            }
        };

        typedef struct int_pair int_pair;

///Point class put here?

        /*class Line {
        public:
            // dir == true => line is horizontal
            Line(Point s, Point d, bool dir)
                    : src(s), dst(d), horizontal(dir) {}

            friend std::ostream &operator<<(std::ostream &ostr_, Line &v) {
                ostr_ << "\tSrc " << v.src << "\n\tDst " << v.dst;
                return ostr_;
            }

            Point src;
            Point dst;
            bool horizontal;
        };*/

        class Rectangle {
        public:

            Rectangle() {};

            Rectangle(Point a,
                      Point d)
                    : top_left(a),
                      bottom_right(d) {}

            Rectangle(std::vector <Point>);

            bool contains(Point);

            bool get_dim_max_length();

            std::pair <Rectangle, Rectangle> split(real_t, bool);

            real_t get_max_length();


            Point get_center() {
                return Point((top_left.x + bottom_right.x) / 2.0, (top_left.y + bottom_right.y) / 2.0);
            }

            real_t get_l2_radius() {
                //return top_left.distance(bottom_right) / 2.0;
                return top_left.dist_l2(bottom_right)/2.0;
            }

            real_t get_linfty_radius() {
                return top_left.dist_linfty(bottom_right) / 2.0;
            }

            real_t get_linfty_diameter() {
                return top_left.dist_linfty(bottom_right);
            }

            friend std::ostream &operator<<(std::ostream &ostr_, Rectangle &v) {

                ostr_ << "\ttop_left: " << v.top_left.x << "," << v.top_left.y << std::endl;
                //ostr_ << "\ttop_right: " << v.top_right << std::endl;
                //ostr_ << "\tbottom_left: " << v.bottom_left << std::endl;
                ostr_ << "\tbottom_right: " << v.bottom_right.x << "," << v.bottom_right.y << std::endl;
                return ostr_;
            }

            Point top_left;
            Point bottom_right;
        };


        Rectangle::Rectangle(std::vector <Point> points) {
            real_t left_bound = std::numeric_limits<real_t>::max();//999999;//std::numeric_limits<real_t>::max();
//            real_t right_bound = std::numeric_limits<real_t>::min();//-std::numeric_limits<real_t>::max();
            real_t right_bound = -std::numeric_limits<real_t>::max();//-std::numeric_limits<real_t>::max();
            real_t bottom_bound = std::numeric_limits<real_t>::max();
//            real_t top_bound = std::numeric_limits<real_t>::min();//-std::numeric_limits<real_t>::max();
            real_t top_bound = -std::numeric_limits<real_t>::max();//-std::numeric_limits<real_t>::max();

            for (Point point : points) {
                left_bound = std::min(point.x, left_bound);
                bottom_bound = std::min(point.y, bottom_bound);
                right_bound = std::max(point.x, right_bound);
                top_bound = std::max(point.y, top_bound);
            }

//top_left and bottom_right suffice to generate the rectangle
            top_left = Point(left_bound, top_bound);
            bottom_right = Point(right_bound, bottom_bound);

        }

        bool Rectangle::contains(Point point) {
            return top_left.x <= point.x && point.x <= bottom_right.x
                   && bottom_right.y <= point.y && point.y <= top_left.y;
        }

        bool Rectangle::get_dim_max_length() {
            return !((bottom_right.x - top_left.x) > (top_left.y - bottom_right.y));
        }

        std::pair <Rectangle, Rectangle> Rectangle::split(real_t intercept, bool horizontal) {

            if (horizontal) {
                Rectangle top(top_left, Point(bottom_right.x, intercept));
                Rectangle bottom(Point(top_left.x, intercept), bottom_right);
                return std::make_pair<>(bottom, top);
            }

            Rectangle left(top_left, Point(intercept, bottom_right.y));
            Rectangle right(Point(intercept, top_left.y), bottom_right);
            return std::make_pair<>(left, right);
        }

        real_t Rectangle::get_max_length() {
            assert(bottom_right.x - top_left.x >= 0 && top_left.y - bottom_right.y >= 0);
            return std::max((bottom_right.x - top_left.x), (top_left.y - bottom_right.y));
        }
    }
}

///////////////////////////////////////////////////////////////

