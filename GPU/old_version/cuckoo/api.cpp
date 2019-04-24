//
// Created by jing on 2018/7/1.
//

#include "api.h"

/// size : the key  in the table
hashAPI::hashAPI(int size) {
    
    checkCudaErrors(cudaGetLastError());


    //rehash will be init by every insert function
    cudaMalloc((void**)&rehash,sizeof(TTT));
    //cudaMemset(remove(),0,size*sizeof(TTT));
    checkCudaErrors(cudaGetLastError());
    if(size<1000) size = 20 * TABLE_NUM * BUCKET_SIZE;
    /// malloc table
    int s_bucket  =  size  / TABLE_NUM / BUCKET_SIZE;
    int s_size = s_bucket *BUCKET_SIZE ;
    table_size= s_size * TABLE_NUM;

    cuckoo * h_table=(cuckoo*)malloc(sizeof(cuckoo));
    for(int i=0;i<TABLE_NUM;i++){
        cudaMalloc((void **) &h_table->table[i], sizeof(TYPE) * s_size * 2);
        cudaMemset(h_table->table[i], 0, sizeof(TYPE) * s_size * 2);
        h_table->Lsize[i]=s_size;
    }

    cudaMalloc((void**)&hash_table,sizeof(cuckoo));
    cudaMemcpy(hash_table,h_table,sizeof(cuckoo),cudaMemcpyHostToDevice);

    /// TODO : free cpu hash table :h_table
    checkCudaErrors(cudaGetLastError());
    printf("init ok\n");
}

void hashAPI::hash_insert(TTT *key, TTT *value,int size) {

    checkCudaErrors(cudaGetLastError());

//    /// overflow ,TODO ...
//    if(size+num_size > table_size*NUM_OVERFLOW_ratio){
//        TTT old_size=table_size;
//        table_size *= NUM_grow_ratio;
//        gpu_rehash( old_size, table_size);
//    }

    num_size+=size;
    TTT* d_keys;
    cudaMalloc((void**)&d_keys, sizeof(TTT)*size);
    cudaMemcpy(d_keys, key, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    TTT* d_value;
    cudaMalloc((void**)&d_value, sizeof(TTT)*size);
    cudaMemcpy(d_value, value, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    // does size need be copy first
    gpu_lp_insert(d_keys,d_value,size,rehash,hash_table,table_size);
    //printf("self check success\n");
    checkCudaErrors(cudaGetLastError());
    printf("insert ok\n");
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
   /// TODO free hash table;
    cudaFree(rehash);
}
