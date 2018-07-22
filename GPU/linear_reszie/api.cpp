//
// Created by jing on 2018/6/8.
//

#include "api.h"

hashAPI::hashAPI(int size) {
    
    checkCudaErrors(cudaGetLastError());
    table_size=size;
    cudaMalloc((void**)&dkey, size*sizeof(TTT));
    cudaMemset(dkey,0,size*sizeof(TTT));

    cudaMalloc((void**)&dvalue, size*sizeof(TTT));
    cudaMemset(dvalue,0,size*sizeof(TTT));

    //rehash will be init by every insert function
    cudaMalloc((void**)&rehash,sizeof(TTT));
    //cudaMemset(remove(),0,size*sizeof(TTT));
    checkCudaErrors(cudaGetLastError());

    gpu_lp_init(dkey,dvalue);
    checkCudaErrors(cudaGetLastError());
    //printf("init ok\n");
}

void hashAPI::hash_insert(TTT *key, TTT *value,int size) {
    //copy
checkCudaErrors(cudaGetLastError());
    TTT* d_keys;
    cudaMalloc((void**)&d_keys, sizeof(TTT)*size);
    cudaMemcpy(d_keys, key, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    TTT* d_value;
    cudaMalloc((void**)&d_value, sizeof(TTT)*size);
    cudaMemcpy(d_value, value, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    // does size need be copy first
    gpu_lp_insert(d_keys,d_value,size,rehash,table_size);
    //printf("self check success\n");
    checkCudaErrors(cudaGetLastError());
    //printf("insert ok\n");
}

void hashAPI::hash_search(TTT *key, TTT *value,int size){
    checkCudaErrors(cudaGetLastError());
    TTT* d_keys;
    cudaMalloc((void**)&d_keys, sizeof(TTT)*size);
    cudaMemcpy(d_keys, key, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    TTT* d_value;
    cudaMalloc((void**)&d_value, sizeof(TTT)*size);
    cudaMemcpy(d_value, value, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    //kernel
    gpu_lp_search(d_keys,d_value,size,table_size);
    
    
    cudaMemcpy(value, d_value, sizeof(TTT)*size, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}



//void hashAPI::hash_delete(int *key,int *ans,int size) {
//    for(int i=0;i<size;i++){
//        ans[i]=table->remove(key[i]);
//    }
//}

hashAPI::~hashAPI() {
    cudaFree(dkey);
    cudaFree(dvalue);
    cudaFree(rehash);
}

void hashAPI::hash_getnum() {
    gpu_lp_getnum(table_size);
}
