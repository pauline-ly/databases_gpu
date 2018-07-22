


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

void hashAPI::hash_kernel(TTT *key, TTT *value,int size,TTT *op) {
    //copy
    checkCudaErrors(cudaGetLastError());
    TTT* d_keys;
    cudaMalloc((void**)&d_keys, sizeof(TTT)*size);
    cudaMemcpy(d_keys, key, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    TTT* d_value;
    cudaMalloc((void**)&d_value, sizeof(TTT)*size);
    cudaMemcpy(d_value, value, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    TTT* d_op;
    cudaMalloc((void**)&d_op, sizeof(TTT)*size);
    cudaMemcpy(d_op, op, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    // does size need be copy first
    gpu_lp_kernel(d_keys,d_value,size,table_size,d_op);
    //printf("self check success\n");
    checkCudaErrors(cudaGetLastError());
    //printf("insert ok\n");
}




hashAPI::~hashAPI() {
    cudaFree(dkey);
    cudaFree(dvalue);
    cudaFree(rehash);
}

void hashAPI::hash_getnum() {
    gpu_lp_getnum(table_size);
}
