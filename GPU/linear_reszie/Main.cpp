#include "linear_probe.h"
#include "dy_hash.h"
#include <iostream>
#include <assert.h>
#include <memory.h>
//#include <stdlib>
#include <cuda_runtime.h>
#include "api.h"

#define SIZE_TEST 1000
using namespace std;

bool  test_size(int size){
    printf("the size is :%d\n",size);
    checkCudaErrors(cudaGetLastError());
    auto *key=new int[size];
    auto *value=new int[size];
    auto *search=new int[size];
    auto *chck=new int[size];

    for(int i=0;i<size;i++){
        key[i]=2*i+3+1;
        value[i]=3*i+3+1;
        search[i]=i+3+1;
        chck[i]=0;
    }
//     //ADD_FAILURE()<<"test_simple"<<"\n";
//     // table_size = size/4
     hashAPI h(size/4);
    //insert size/8
   // h.hash_insert(key,value,size/8);
    
//     //insert size/8 ~ size/4
//     h.hash_insert(key,value,size/4);
//     //insert left 3/4
    
    double diff,diff1;
    struct  timespec start, end;
    struct  timespec start1, end1;

    clock_gettime(CLOCK_MONOTONIC, &start);

    h.hash_insert(key,value,size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    diff = 1000000 * (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/1000;
    printf("INSERT, the time is %.2lf us, speed is %.2f Mops\n", 
        (double)diff, (double)(size) / diff);

    h.hash_getnum();
    

   
    clock_gettime(CLOCK_MONOTONIC, &start1);

    h.hash_search(key,chck,size);

    clock_gettime(CLOCK_MONOTONIC, &end1);
    diff1 = 1000000 * (end1.tv_sec-start1.tv_sec) + (double)(end1.tv_nsec-start1.tv_nsec)/1000;
    printf("SEARCH, the time is %.2lf us, speed is %.2f Mops\n", 
        (double)diff1, (double)(size) / diff1);
   
   // int tmp=0;
    //for(int i=0;i<size;i++){
    //    if(chck[i]!=value[i]){
            //printf("check error %d :%d %d \n",key[i],chck[i],value[i]);
    //        tmp++ ;
            //(chck[i]==value[i]);
    //    }
    //}
    //printf("test %d pass,%d check not pass\n",size,tmp);

    checkCudaErrors(cudaGetLastError());
    delete[] key;
    delete[] value;
    delete[] search;
    delete[] chck;
    return true;
}

int main(){
    
    //assert(test_size(50));
    assert(test_size(100));
    assert(test_size(1000));
    assert(test_size(10000));
    assert(test_size(100000));
    assert(test_size(1000000));
    assert(test_size(10000000));
    assert(test_size(100000000));
    


    
}