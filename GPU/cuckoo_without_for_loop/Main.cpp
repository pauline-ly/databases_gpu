//
// Created by jing on 2018/7/1.
//

#include "cuckoo.h"
#include "api.h"

#define SIZE_TEST 1000
// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)

using namespace std;

bool  test_size(int size){
    
    int adds=0;
    int deletes=0;
    int i;

   	if (adds+deletes > 100) {
    	printf("Sum of add and delete precentages exceeds 100.\nAborting...\n");
     	exit(1);
  	}
    
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
    
    printf("\tsize:%d\n",size);
    cudaDeviceReset();
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
     hashAPI h(size*2);
    //insert size/8
    //h.hash_insert(key,value,size/8);
    
//     //insert size/8 ~ size/4
//     h.hash_insert(key,value,size/4);
//     //insert left 3/4
     //h.hash_insert(key,value,size);
     struct  timespec start, end;
     double diff;
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    h.hash_insert(key,value,size);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    diff = 1000000 * (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/1000;
    printf("insert：the time is %.2lf us, speed is %.2f Mops\n", 
        (double)diff, (double)(size) / diff);
    
    
    h.hash_search(key,chck,size);
   /*
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    h.hash_kernel(key,value,size,op);
    
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    diff = 1000000 * (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/1000;
    printf("the time is %.2lf us, speed is %.2f Mops\n", 
        (double)diff, (double)(size) / diff);
    
    
    
   */
    int tmp=0;
    for(int i=0;i<size;i++){
        if(chck[i]!=value[i]){
            //printf("check error %d :%d %d \n",key[i],chck[i],value[i]);
            tmp++ ;
            //(chck[i]==value[i]);
        }
    }

    printf("¶¶¶test %d complete ,%d not pass:abort %.2f\n\n\n\n\n",size,tmp,tmp*1.0/size);

    checkCudaErrors(cudaGetLastError());
    delete[] key;
    delete[] value;
    delete[] search;
    delete[] chck;
    return true;
}



/*
int main(){



    assert(test_size(1<<6));
    assert(test_size(1<<8));
    assert(test_size(1<<10));
    assert(test_size(1<<12));
    assert(test_size(1<<14));
    assert(test_size(1<<16));
    assert(test_size(1<<18));
    assert(test_size(1<<20));
//


    
}
*/


int main(int argc, char** argv) {  
    

    int size=0;

    if (argc >=2 ) 
    {
        size=atoi(argv[1]);
    }
    
    test_size(size);
    
    return 0;

    
}
