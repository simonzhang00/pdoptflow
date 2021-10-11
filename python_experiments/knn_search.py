from pdoptflow import W1# change this to import W1 if importing manually
import numpy as np
import glob
import csv
import time
import gudhi.hera
import pd_estimators as pde
import random
import os
import sys
from tqdm import tqdm

#global qt_diagrams
global ft_diagrams
global query_ft_diagrams
global point_ft_diagrams
global unique_points
global point_diagrams
global query_diagrams

def diagram2ft_diagram(array, dict, unique_points):
    #p_list = []
    ft_diagram = []
    for row in array:
        birth = float(row[0])
        death = float(row[1])
        #added to see if aspect ratio kills memory:
        #birth= round(birth,5)
        #death= round(death,5)
        #birth-=np.random.random()
        #death+=np.random.random()
        p = (birth, death)
        #p_list.append((birth, death))
        if p not in dict:
            dict[p] = len(unique_points)
            unique_points.append(p)
        ft_diagram.append((dict[p], 1.0))#dict[p] is index of point p in total list of points
    #diagram = np.asarray(p_list)
    #return p_list, ft_diagram
    return ft_diagram

# Converts a csv file to a list of points
# Also returns the representation of the diagram required to calculate
# flowtree/embedding distance (i.e. [(a1, 1.0),... ]) where a1 is the index of
# the point in the total list of points.
def csv_to_diagrams(file, dict, unique_points):
    p_list = []
    ft_diagram = []
    with open(file) as f:
        reader = csv.reader(f,delimiter= ',')
        for row in reader:
            birth = float(row[0])
            death = float(row[1])
            #added to see if aspect ratio kills memory:
            #birth= round(birth,5)
            #death= round(death,5)
            #birth-=np.random.random()
            #death+=np.random.random()
            p = (birth, death)
            p_list.append((birth, death))
            if p not in dict:
                dict[p] = len(unique_points)
                unique_points.append(p)
            ft_diagram.append((dict[p], 1.0))#dict[p] is index of point p in total list of points
    f.close()
    diagram = np.asarray(p_list)

    return p_list, ft_diagram

def load_data(PD1, PD2):
    #basepath = folder
    all_files = []
    #for entry in os.listdir(basepath):
    #    if os.path.isfile(os.path.join(basepath, entry)):
    #        all_files.append(os.path.join(basepath, entry))
    all_files= [os.path.abspath(PD1), os.path.abspath(PD2)]
    diagrams = []
    ft_diagrams = []
    unique_pts = []
    dict_pt = {}
    for file in all_files:
        diagram, ft_diagram = csv_to_diagram(file, dict_pt, unique_pts)
        diagrams.append(diagram)
        ft_diagrams.append(ft_diagram)
    vocab = np.array(unique_pts)
    return vocab, diagrams, ft_diagrams

def find_candidates(method, n_resulting_candidates, query_diagram_index, candidate_diagram_indices):
    result= []
    #print("candidate diagrams: ",candidate_diagrams)#[tup[0] for tup in candidate_diagrams])
    for candidate in candidate_diagram_indices:
        p_index= int(candidate[1])
        #print(candidate)
        #print(p_index)
        if method=="wcd":
            wcd= W1.wcd(query_diagrams[query_diagram_index], point_diagrams[p_index])
            result.append([wcd, p_index])
        elif method=="rwmd":
            rwmd= W1.rwmd(query_diagrams[query_diagram_index],  point_diagrams[p_index])
            result.append([rwmd, p_index])
        elif method== "sparse":
            w1= W1.wasserstein1(query_diagrams[query_diagram_index],point_diagrams[p_index],1,10000)
            result.append([w1,p_index])
        elif method=="exact":
            w1= W1.wasserstein1(query_diagrams[query_diagram_index], point_diagrams[p_index], 18, 100000)
            result.append([w1, p_index])
        elif method=="exactexact":
            #w1= W1.wasserstein1(query_diagrams[query_diagram_index], point_diagrams[p_index], 30, 100000)
            w1= gudhi.hera.wasserstein_distance(query_diagrams[query_diagram_index], point_diagrams[p_index], order =1, internal_p=2)
            result.append([w1, p_index])
        elif method=="flowtree":
            w1= solver.flowtree_distance(query_ft_diagrams[query_diagram_index],point_ft_diagrams[p_index], 2)
            result.append([w1, p_index])
        elif method=="quadtree":
            w1= solver.embedding_distance(query_diagram_index,p_index)  
            result.append([w1,p_index])
    result.sort(key=lambda tup: tup[0])
    #print("result: ",result)#[tup[0] for tup in result])
    #if not set([tup[1] for tup in result])==set([tup[1] for tup in candidate_diagrams]):
    #    print("ERROR")
    #    exit(1)
    return np.array(result[:n_resulting_candidates])

if __name__ == "__main__":
    all_files= list(glob.glob("others/w1estimators/python/sample_data/*.csv"))
    seed =int(sys.argv[1])
    random.seed(seed)
    random.shuffle(all_files)
    points_files= all_files[:100]
    ##points_files= sorted(glob.glob("knn-dataset/*.txt"))[:10]
    # points_files= ["knn-dataset/queries/Synthesized_FLASH25_Axial_0370.tiff.txt"]#['../datasets/PD_mri_750-perturbed.txt']#
    #points_files= ['../datasets/PD_lower_star_Beijing_1654x2270-csv.txt']#
    # points_files= ['../datasets/PD_mri_750-perturbed.txt']#
    ##query_points_files= sorted(glob.glob("knn-dataset/queries/*.txt"))
    query_points_files= all_files[100:]
    #query_points_files= ["knn-dataset/Synthesized_FLASH25_Axial_0371.tiff.txt"]#['../datasets/PD_mri_751-perturbed.txt']#
    #query_points_files= ['../datasets/PD_lower_star_Athens_760x585-csv.txt']
    # query_points_files= ['../datasets/PD_mri_751-perturbed.txt']#

    point_diagrams= []
    query_diagrams= []
    ft_diagrams= []
    query_ft_diagrams= []
    point_ft_diagrams= []
    unique_points= []
    dict= {}

    start= time.time()
    for qf in query_points_files:
        pd, ft_diagram= csv_to_diagrams(qf, dict, unique_points)
        ft_diagrams.append(ft_diagram)
        query_diagrams.append(pd)
        query_ft_diagrams.append(ft_diagram)
    
    for pf in points_files:
        pd, ft_diagram=csv_to_diagrams(pf, dict, unique_points)
        ft_diagrams.append(ft_diagram)
        point_diagrams.append(pd)
        point_ft_diagrams.append(ft_diagram)
    vocab= np.array(unique_points).astype(np.float32)
    stop= time.time()
    print("loading time: ", stop-start)
    '''
    np.random.seed(0)
    for i in range(50):
        pd= np.random.normal(0,10*(i+1)*(i+1)*(i+1),(1000,2))
        ft_diagram= diagram2ft_diagram(pd, dict, unique_points)
        point_diagrams.append(pd)
        point_ft_diagrams.append(ft_diagram)
        ft_diagrams.append(ft_diagram)
    for i in range(100):
        pd= np.random.normal(0,10*(i+1)*(i+1)*(i+1),(1001,2))
        ft_diagram= diagram2ft_diagram(pd, dict, unique_points)
        query_diagrams.append(pd)
        query_ft_diagrams.append(ft_diagram)
        ft_diagrams.append(ft_diagram)
    vocab= np.array(unique_points).astype(np.float32)
    '''
    #print(len(ft_diagrams[0]),len(ft_diagrams[1]))
    #print(not ft_diagrams[0]==ft_diagrams[1])
    solver = pde.PDEstimators()
    start = time.time()
    solver.load_points(vocab)
    end = time.time()
    print("Points in flowtree: ", len(vocab))
    print("Time for building the flowtree: ", end - start)
    start = time.time()
    solver.load_diagrams(ft_diagrams)
    end = time.time()
    print("Time for loading all embedding representations: ", end - start)
    topK= 1
    n_wcd= 95
    n_rwmd= 10
    n_ft= 10
    n_qt= 95
    n_sparse= 5
    n_exact= topK
    n_correct= 0
    n_correct_qt= 0
    n_correct_ft= 0
    n_correct_rwmd= 0
    n_correct_wcd= 0
    n_correct_sp= 0
    accuracy= []
    exactexact_time= 0
    wcd_time= 0
    rwmd_time=0
    exact_time= 0
    ft_time=0
    qt_time= 0
    sp_time= 0
    ALL_gt_candidates= []
    if os.path.isfile("gt_candidates_seed"+str(seed)+".npz"):
        print("LOADING GT CANDIDATES FROM FILE")
        container= np.load("gt_candidates_seed"+str(seed)+".npz")
        ALL_gt_candidates = [container[key] for key in container]
    for i,q in enumerate(tqdm(query_diagrams)):
        #print(list(zip([0]*len(point_diagrams),point_diagrams)))
        dummy_candidates= np.asarray(list(zip([0]*len(point_diagrams),range(len(point_diagrams)))))

        if os.path.isfile("gt_candidates_seed"+str(seed)+".npz"):
            gt_candidates= ALL_gt_candidates[i]#np.load("exact_candidates_seed"+str(seed)+".npz")#["ALL_gt_candidates"][i]
            print(gt_candidates.shape)
            print("LOADED GT FOR SEED: %d at query index %d"%(seed,i))
        else:
            start= time.time()
            gt_candidates= find_candidates("exactexact", len(point_diagrams), i, dummy_candidates)
            stop= time.time()
            exactexact_time+= (stop-start)
            ALL_gt_candidates.append(gt_candidates)

        start=time.time()
        wcd_candidates=find_candidates("wcd", n_wcd, i, dummy_candidates)
        stop= time.time()
        wcd_time+= (stop-start)

        start= time.time()
        rwmd_candidates=find_candidates("rwmd", n_rwmd, i, dummy_candidates)#wcd_candidates)
        stop= time.time()
        rwmd_time+= (stop-start)
        start= time.time()
        quadtree_candidates= find_candidates("quadtree", n_qt, i, dummy_candidates)
        stop= time.time()
        qt_time+= (stop-start)

        start= time.time()
        flowtree_candidates= find_candidates("flowtree", n_ft, i, dummy_candidates)
        stop= time.time()
        ft_time+= (stop-start)

        start= time.time()
        sparse_candidates= find_candidates("sparse", n_sparse, i, dummy_candidates)
        stop= time.time()
        sp_time+= (stop-start)
        start= time.time()
        
        final_candidates= find_candidates("exact", n_exact, i, dummy_candidates)#flowtree_candidates)
        stop= time.time()
        exact_time+= (stop-start)


        #print(len(final_candidates))
        #print(final_candidates[0][1], exact_candidates[0][1])
        print("PDoptFlow final_candidates [approx, NN index]: ", final_candidates[0:topK])
        print("flowtree_candidates [approx, NN index]: ", flowtree_candidates[0:topK])
        print("gt candidates [ans, NN index]: ", gt_candidates[0:topK+1])
        match= 0
        '''
        for c in exact_candidates[0:K,1]:
            print("exact answer: ",c)
            print("candidates to compare with: ", final_candidates[0:K,1])
            if c in final_candidates[0:K,1]:
                match+=1.0
        '''

        if gt_candidates[0,1] in final_candidates[0:topK,1]:
            n_correct+= 1
        if gt_candidates[0,1] in rwmd_candidates[0:topK,1]:
            n_correct_rwmd+=1
        if gt_candidates[0,1] in wcd_candidates[0:topK,1]:
            n_correct_wcd+= 1
        if gt_candidates[0,1] in flowtree_candidates[0:topK,1]:
            n_correct_ft+=1
        if gt_candidates[0,1] in quadtree_candidates[0:topK,1]:
            n_correct_qt+=1
        if gt_candidates[0,1] in sparse_candidates[0:topK,1]:
            n_correct_sp+=1    
            #print("fail case:")
            #print(exact_candidates[0,1])
            #print(final_candidates[0:topK,1])
        #print('*'*int(100/len(query_diagrams)),end = '')
        #if(match>0.9*topK):
        #    n_correct+= 1
        #print("% correct candidates: ",match/K)
        #accuracy.append(match/K)
        #if(final_candidates[0][1]==exact_candidates[0][1]):
        #    print("CORRECT!")
        #    n_correct+=1
    #print(accuracy)
    if not os.path.isfile("gt_candidates_seed"+str(seed)+".npz"):
        np.savez("gt_candidates_seed"+str(seed)+".npz", *ALL_gt_candidates)

    print("PDoptFlow(s=18) CORRECT 1-NNs: ", n_correct)
    print("QUADTREE CORRECT 1-NNs: ", n_correct_qt)
    print("FLOWTREE CORRECT 1-NNs: ", n_correct_ft)
    print("RWMD CORRECT 1-NNs: ", n_correct_rwmd)
    print("WCD CORRECT 1-NNs: ", n_correct_wcd)
    print("PDoptFlow(s=1) CORRECT 1-NNs: ", n_correct_sp)
    print("hera(gt) time (0 if load from cached file): ", exactexact_time)
    print("wcd time: ", wcd_time)
    print("rwmd time: ", rwmd_time)
    print("PDoptFlow(s=18) exact time: ", exact_time)
    print("PDoptFlow(s=1) time: ", sp_time)
    print("flowtree time: ", ft_time)
    print("quadtree time: ", qt_time)
