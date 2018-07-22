


#include "linear_probe.h"
#include "dy_hash.h"
#include <iostream>
#include <assert.h>
#include <memory.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "api.h"

#define SIZE_TEST 1000

// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)

bool  test_size(int size){

    int adds=50;
    int deletes=50;
    int i;

   	if (adds+deletes > 100) {
    	printf("Sum of add and delete precentages exceeds 100.\nAborting...\n");
     	exit(1);
  	}


    printf("the size is :%d\n",size);
    checkCudaErrors(cudaGetLastError());
    auto *op=new int[size];

  	// Populate the op sequence
  	for(i=0;i<(size*adds)/100;i++){
    	op[i]=ADD;
  	}
  	for(;i<(size*(adds+deletes))/100;i++){
    	op[i]=DELETE;
  	}
  	for(;i<size;i++){
    	op[i]=SEARCH;
  	}

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


     hashAPI h(size/4);


    struct  timespec start, end;
    double diff;

    clock_gettime(CLOCK_MONOTONIC, &start);

    h.hash_kernel(key,value,size,op);

    clock_gettime(CLOCK_MONOTONIC, &end);
    diff = 1000000 * (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/1000;
    printf("INSERT, the time is %.2lf us, speed is %.2f Mops\n", 
        (double)diff, (double)(size) / diff);


    checkCudaErrors(cudaGetLastError());
    delete[] key;
    delete[] value;
    delete[] search;
    delete[] chck;
    return true;
}


int main(){
    
    
    assert(test_size(100));
    assert(test_size(1000));
    assert(test_size(10000));
    assert(test_size(100000));
    assert(test_size(1000000));
    assert(test_size(10000000));
    assert(test_size(100000000));
    
    return 0;

    
}