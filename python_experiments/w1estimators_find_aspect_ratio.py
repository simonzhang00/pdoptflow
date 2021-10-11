import pd_estimators as pde
import os
import time
import random
import sys
import faulthandler
import math
import csv
import numpy as np

# Converts a csv file to a list of points
# Also returns the representation of the diagram required to calculate
# flowtree/embedding distance (i.e. [(a1, 1.0),... ]) where a1 is the index of
# the point in the total list of points.
def csv_to_diagram(file, dict, unique_points):
    p_list = []
    ft_diagram = []
    with open(file) as f:
        reader = csv.reader(f, delimiter= " ")
        for row in reader:
            birth = float(row[0])
            death = float(row[1])
            p = (birth, death)
            p_list.append((birth, death))
            if p not in dict:
                dict[p] = len(unique_points)
                unique_points.append(p)
            ft_diagram.append((dict[p], 1.0))#dict[p] is index of point p in total list of points
    f.close()
    diagram = np.asarray(p_list)

    return p_list, ft_diagram

if __name__ == '__main__':
    dict= {}
    unique_points= []
    p_list, index_diagram= csv_to_diagram(sys.argv[1],dict,unique_points)
    max_pairwise_dist= -1
    min_pairwise_dist= float("inf")
    
    for p in p_list:
        for q in p_list:
            if(p[0]!=q[0] and p[1]!=q[1]):
                d= math.sqrt((p[0]-q[0])*(p[0]-q[0])+(p[1]-q[1])*(p[1]-q[1]))
                if(d>max_pairwise_dist):
                    max_pairwise_dist= max(max_pairwise_dist,d)
                if(d<min_pairwise_dist):
                    min_pairwise_dist= min(min_pairwise_dist,d)
    print("max pairwise distance: ", max_pairwise_dist)
    print("min pairwise distance: ", min_pairwise_dist)

