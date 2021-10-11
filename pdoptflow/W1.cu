#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

//#include <IO/read_asciidiagram.h>
#include <wasserstein.h>
//#include <exception>


double wasserstein1(pybind11::array_t<double> numpyA, pybind11::array_t<double>numpyB, double s, int additive){
    pybind11::buffer_info bufferA = numpyA.request();
    pybind11::buffer_info bufferB = numpyB.request();
    if(!(bufferA.shape[1]==2 && bufferB.shape[1]==2)){
        throw std::runtime_error("Input shapes must be of shape nx2");
    }

    double *ptrA = (double *) bufferA.ptr;
    double *ptrB = (double *) bufferB.ptr;
    //int additive= 250000;
    int multiplicative= 0;
    //double s= 5;
    std::vector<PDoptFlow::W1::Point> diagramA;
    diagramA.clear();
//    diagramA.reserve(bufferA.shape[0]);

//#pragma omp parallel for schedule(static,1)
    for(int i=0; i<bufferA.shape[0]; i++) {
        PDoptFlow::W1::Point p;
        p.x= ptrA[i*2+0];
        p.y= ptrA[i*2+1];
//        //std::cout<<"diagramA["<<i<<"] is: "<<p.x<<" "<<p.y<< std::setprecision(15) <<std::endl;
//        //diagramA[i]= p;
        diagramA.emplace_back(p);
//        diagramA[i].x= ptrA[i*2+0];
//        diagramA[i].y= ptrA[i*2+1];
    }
//    std::cout<<"diagramA.size(): "<<diagramA.size()<<std::endl;
    std::vector<PDoptFlow::W1::Point> diagramB;
    diagramB.clear();
//    diagramB.reserve(bufferB.shape[0]);
//#pragma omp parallel for schedule(static,1)
    for(int i=0; i<bufferB.shape[0];i++) {
        PDoptFlow::W1::Point p;
        p.x= ptrB[i*2+0];
        p.y= ptrB[i*2+1];
//        //std::cout<<"diagramB["<<i<<"] is: "<<p.x<<" "<<p.y<< std::setprecision(15) <<std::endl;
//        //diagramB[i]= p;
        diagramB.emplace_back(p);
//        diagramB[i].x= ptrB[i*2+0];
//        diagramB[i].y= ptrB[i*2+1];
    }
    double w1dist = PDoptFlow::wasserstein_dist(diagramA, diagramB, s, true, additive,multiplicative);//1.00248756219);//1+0.01);
    //std::cout<<w1dist<< std::setprecision(15) <<std::endl;
    return w1dist;
}

double centroid_distance(pybind11::array_t<double> numpyA, pybind11::array_t<double>numpyB){
    pybind11::buffer_info bufferA = numpyA.request();
    pybind11::buffer_info bufferB = numpyB.request();
    if(!(bufferA.shape[1]==2 && bufferB.shape[1]==2)){
        throw std::runtime_error("Input shapes must be of shape nx2");
    }

    double *ptrA = (double *) bufferA.ptr;
    double *ptrB = (double *) bufferB.ptr;
    std::vector<PDoptFlow::W1::Point> diagramA(bufferA.shape[0]);
    std::vector<PDoptFlow::W1::Point> diagramB(bufferB.shape[0]);
#pragma omp parallel for schedule(static,1)
    for(int i=0; i<bufferA.shape[0]; i++) {
        PDoptFlow::W1::Point p;
        p.x= ptrA[i*2+0];
        p.y= ptrA[i*2+1];
        diagramA[i]= p;
    }
#pragma omp parallel for schedule(static,1)
    for(int i=0; i<bufferB.shape[0];i++) {
        PDoptFlow::W1::Point p;
        p.x= ptrB[i*2+0];
        p.y= ptrB[i*2+1];
        diagramB[i]= p;
    }

    double PD_centroid_dist = PDoptFlow::W1::compute_centroid_dist(diagramA, diagramB);//1.00248756219);//1+0.01);
    return PD_centroid_dist;
}

double relaxed_wmd(pybind11::array_t<double> numpyA, pybind11::array_t<double>numpyB){
    pybind11::buffer_info bufferA = numpyA.request();
    pybind11::buffer_info bufferB = numpyB.request();
    if(!(bufferA.shape[1]==2 && bufferB.shape[1]==2)){
        throw std::runtime_error("Input shapes must be of shape nx2");
    }

    double *ptrA = (double *) bufferA.ptr;
    double *ptrB = (double *) bufferB.ptr;
    std::vector<PDoptFlow::W1::Point> diagramA(bufferA.shape[0]);
    std::vector<PDoptFlow::W1::Point> diagramB(bufferB.shape[0]);
#pragma omp parallel for schedule(static,1)
    for(int i=0; i<bufferA.shape[0]; i++) {
        PDoptFlow::W1::Point p;
        p.x= ptrA[i*2+0];
        p.y= ptrA[i*2+1];
        diagramA[i]= p;
    }
#pragma omp parallel for schedule(static,1)
    for(int i=0; i<bufferB.shape[0];i++) {
        PDoptFlow::W1::Point p;
        p.x= ptrB[i*2+0];
        p.y= ptrB[i*2+1];
        diagramB[i]= p;
    }

    double rwmd = PDoptFlow::W1::compute_RWMD(diagramA, diagramB);//1.00248756219);//1+0.01);
    return rwmd;
}

PYBIND11_MODULE(W1, m) {
m.doc() = "pybind11 example plugin"; // optional module docstring

m.def("wasserstein1", &wasserstein1, "Compute the Wasserstein-1 distance between two persistence diagrams with sparsity factor s.");
m.def("rwmd", &relaxed_wmd, "Compute the relaxed word mover's distance between persistence diagrams in parallel.");
m.def("wcd", &centroid_distance, "Compute the word centroid distance between persistence diagrams.");

}
