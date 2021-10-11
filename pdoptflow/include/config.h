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

#include <map>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <climits>
#include <assert.h>
#include <limits>
#include <memory>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <profiling/stopwatch.h>


namespace PDoptFlow {
    namespace W1 {

//#define VERBOSE
#ifdef VERBOSE
#define PROFILING
#define COUNTING
#endif
        typedef int32_t integer_t;
        typedef double real_t;
        real_t EPS= 1e-10;
#define sqrt2div2 0.70710678118

//#define DEBUG


        std::pair< double, bool > trunc_n( double value, std::size_t digits_after_decimal = 0 )
        {
            static constexpr std::intmax_t maxv = std::numeric_limits<std::intmax_t>::max() ;
            static constexpr std::intmax_t minv = std::numeric_limits<std::intmax_t>::min() ;

            unsigned long long multiplier = 1 ;
            for( std::size_t i = 0 ; i < digits_after_decimal ; ++i ) multiplier *= 10 ;

            const auto scaled_value = value * multiplier ;

            const bool did_trunc =  scaled_value != scaled_value+0.5 && scaled_value != scaled_value-0.5 ;

            if( scaled_value >= minv && scaled_value <= maxv )
                return { double( std::intmax_t(scaled_value) ) / multiplier, did_trunc } ;
            else return { std::trunc(scaled_value) / multiplier, did_trunc } ;
        }
        __host__ __device__ class Point {
        public:
            __host__ __device__ Point() {}

            __host__ __device__ Point(real_t x, real_t y) : x(x), y(y), vertice(-1) {}

            __host__ __device__
            Point operator+(const Point& q) const {
                return {x+q.x,y+q.y};
            };
            //~Point() {}

            bool operator==(const Point &r) {
                return x == r.x && y == r.y;
                //return std::max(x-r.x,y-r.y)<EPS;
            }

            bool operator<(const Point &r) {
                return x < r.x || (x == r.x && y < r.y);
            }

            /*real_t distance(Point v) {
                return std::sqrt(std::pow(v.x - x, 2) + std::pow(v.y - y, 2));
            }*/

            __host__ __device__ real_t dist_linfty(Point q) {
                //return std::max(std::abs(x - q.x), std::abs(y - q.y));
                return max(fabs(x - q.x), fabs(y - q.y));
            }

            __host__ __device__ real_t dist_l2(Point q){
                return sqrt((x-q.x)*(x-q.x)+(y-q.y)*(y-q.y));
            }

            friend std::ostream &operator<<(std::ostream &ostr_, Point &v) {
                ostr_ << "x: " << v.x << " y: " << v.y;
                return ostr_;
            }


            __host__ __device__ real_t get_linfty_dist_to_diag(){
                return fabs(y-x)/2.0;
            }

            __host__ __device__ real_t get_l2_dist_to_diag(){
                return fabs(y-x)*sqrt2div2;
            }

            real_t x;
            real_t y;
            int vertice;//this can be removed so long as the points aren't resorted
        };
        bool eps_close(real_t a, real_t b){
            return std::abs(b-a)<EPS;
        }
        bool eps_close(std::pair<Point,int>& a, std::pair<Point,int>& b){
            return std::max(std::abs(a.first.x-b.first.x), std::abs(a.first.y-b.first.y))<EPS;
        }
        bool compare_Point_mass(const struct std::pair<Point,int>& l, const std::pair<Point,int>& r)
        {
            return (l.first.x<r.first.x) || ((l.first.x==r.first.x) && l.first.y<r.first.y);
        }

#define CUDACHECK(cmd) do {\
    cudaError_t e= cmd;\
    if( e != cudaSuccess ) {\
        printf("Failed: Cuda error %s:%d '%s'\n",\
        __FILE__,__LINE__,cudaGetErrorString(e));\
    exit(EXIT_FAILURE);\
    }\
    } while(0)

    }
}