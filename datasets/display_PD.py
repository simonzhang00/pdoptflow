'''
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

'''

'''
USAGE: python3 display_PD.py PD1.txt PD2.txt
displays a plot of the two PDs superimposed
'''

import sys
import os
import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:
    # x,y = [float(line) for line in next(f).split()]
    array = [[float(z) for z in line.split()] for line in f]
    print(array)
    PD1_num_ptns = len(array)
    x1, y1 = [[i for i, j in array],
              [j for i, j in array]]
    max_x = max(x1)
    max_y = max(y1)
with open(sys.argv[2]) as f:
    # x,y = [float(line) for line in next(f).split()]
    array = [[float(z) for z in line.split()] for line in f]
    print(array)
    PD2_num_ptns = len(array)
    x2, y2 = [[i for i, j in array],
              [j for i, j in array]]
    max_x = max(max_x, max(x2))
    max_y = max(max_y, max(y2))
    plt.figure(figsize=(8, 8))
    plt.axis('scaled')
    print("PD1 NUM POINTS: ", PD1_num_ptns)
    print("PD2 NUM POINTS: ", PD2_num_ptns)
    plt.plot((0, max(max_x, max_y)), (0, max(max_x, max_y)), linestyle='solid', color='black', linewidth=1.5)
    head1, tail1 = os.path.split(sys.argv[1])
    head2, tail2 = os.path.split(sys.argv[2])
    # p1= plt.scatter(x1,y1,marker='.', color= 'blue', alpha= 0.4)
    # p1= plt.scatter(x1,y1,marker='.', color= 'blue', alpha= 0.4, s=80)
    p2 = plt.scatter(x2, y2, marker='x', color='red', alpha=0.9, s=20)
    p1 = plt.scatter(x1, y1, marker='.', color='blue', alpha=0.4, s=80)

    plt.legend((p1, p2), ('PD1' + ", |PD1|=" + str(PD1_num_ptns), 'PD2' + ", |PD2|=" + str(PD2_num_ptns)),
               scatterpoints=1, prop={'size': 18}, loc='lower right')
    # plt.title(tail1+" X "+tail2)
    plt.xlim(xmin=-0.001, xmax=max(max_x, max_y))
    plt.ylim(ymin=-0.001, ymax=max(max_x, max_y))
    plt.grid()
    plt.show()
