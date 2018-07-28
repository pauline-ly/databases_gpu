//
// Created by jing on 2018/7/1.
//

#include "api.h"

bool isPrime(int num)
{
    if (num == 2 || num == 3) return true;
    if (num % 6 != 1 && num % 6 != 5) return false;
    for (int i = 5; i*i <= num; i += 6)
        if (num % i == 0 || num % (i+2) == 0) return false;
    return true;
}

int nextPrime(int n)
{
    bool state=isPrime(n);
    while(!state)
        state=isPrime(++n);
#if size_debug
    printf("find prime %d\n",n);
#endif
    return n;
}


/// size : the key  in the table
hashAPI::hashAPI(int size) {
    
    checkCudaErrors(cudaGetLastError());
    //rehash will be init by every insert function
    cudaMalloc((void**)&rehash,sizeof(TTT));
    //cudaMemset(remove(),0,size*sizeof(TTT));
    checkCudaErrors(cudaGetLastError());

    /// adjust size to prime
    int s=size /( TABLE_NUM * BUCKET_SIZE);
#if size_debug
    printf("s:%d",s++);
#endif
    s=nextPrime(s>3?s:3);

    int s_bucket  =  s;
    int s_size = s_bucket *BUCKET_SIZE ;

    table_size= s_size * TABLE_NUM;

    /// malloc table
    checkCudaErrors(cudaGetLastError());
    cuckoo * h_table=(cuckoo*)malloc(sizeof(cuckoo));
    printf("size: %d tablesize:%d \n",size,table_size);
    for(int i=0;i<TABLE_NUM;i++){
        cudaMalloc((void **) &h_table->table[i], sizeof(TYPE) * s_size * 2);
        cudaMemset(h_table->table[i], 0, sizeof(TYPE) * s_size * 2);
        h_table->Lsize[i]=s_size;
    }
    checkCudaErrors(cudaGetLastError());
    /// malloc lock   TODO size of lock
    cudaMalloc((void**)&h_table->Lock,sizeof(int)*table_size/32);
    cudaMemset(h_table->Lock,0,sizeof(int)*table_size/32);

    /// copy to gpu table
    cudaMalloc((void**)&hash_table,sizeof(cuckoo));
    cudaMemcpy(hash_table,h_table,sizeof(cuckoo),cudaMemcpyHostToDevice);

    /// TODO : free cpu hash table :h_table
    checkCudaErrors(cudaGetLastError());
    printf("init ok\n");
}

void hashAPI::hash_insert(TTT *key, TTT *value,int size) {

//    checkCudaErrors(cudaGetLastError());

//    /// overflow ,TODO  rehash...
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
//    checkCudaErrors(cudaGetLastError());
//    printf("insert ok\n");
}

void hashAPI::hash_search(TTT *key, TTT *value,int size){
//    checkCudaErrors(cudaGetLastError());
    TTT* d_keys;
    cudaMalloc((void**)&d_keys, sizeof(TTT)*size);
    cudaMemcpy(d_keys, key, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    TTT* d_value;
    cudaMalloc((void**)&d_value, sizeof(TTT)*size);
    cudaMemcpy(d_value, value, sizeof(TTT)*size, cudaMemcpyHostToDevice);

    //kernel
    gpu_lp_search(d_keys,d_value,size,hash_table);
    
    
    cudaMemcpy(value, d_value, sizeof(TTT)*size, cudaMemcpyDeviceToHost);
//    checkCudaErrors(cudaGetLastError());
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
