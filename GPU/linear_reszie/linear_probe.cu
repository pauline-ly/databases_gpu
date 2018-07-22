//
// Created by jing on 2018/6/4.
///
// TODO: add remove
// TODO: add big data


#include "linear_probe.h"

//table
__device__ TTT* tkey;
__device__ TTT* tvalue;




//hash function
inline __device__  unsigned
hash_function(TTT key, TTT size)
{
    return key % _PRIME % size ;
}

// insert sub-fun
__device__   bool
Add(TTT k, TTT v,TTT table_size,TTT* resize) {
    int hash = hash_function(k, table_size);

    TTT old_T;

    for(int attempt=0;attempt<MAX_ITERATOR;attempt++){


        old_T=atomicCAS(tkey+hash,0,k);

        if(old_T==0){
            tvalue[hash]=v;
            return true;
        }

        if(old_T==k){
            tvalue[hash]=v;
            return true;
        }

        // differ
        hash++;
        hash %= table_size;
        //hash += attempt * attempt;
    }
    *resize=1;
    return false;
}


// search sub-fun
__device__  TTT
Search(TTT k, TTT table_size) {
    int hash = hash_function(k, table_size);



    for(int attempt=0;attempt<MAX_ITERATOR;attempt++) {


        //tmp only return t/f
        if (tkey[hash] == k) return tvalue[hash];
        // differ
        hash++;
        hash %= table_size;
        //hash += attempt * attempt;
    }

    return -1;
}



//init set gpu hash table
__global__
void lp_init(TTT *k,TTT* v){
    tkey=k;
    tvalue=v;
}

//insert kernel
__global__ void
lp_insert(TTT* key,TTT* value,TTT size,TTT* resize,TTT table_size){
    //printf("lp insert\n");
    *resize=0;
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    while(tid<size){
        Add(key[tid],value[tid],table_size,resize);
        int a=Search(key[tid],table_size);
        if(a!=value[tid]){
           // printf("!!!error  \t%d %d s: %d\n",key[tid],value[tid],a);
            
        }
        tid+=NUM_BLOCK*NUM_THREADS;
    }
    tid=blockIdx.x*blockDim.x+threadIdx.x;
    //if(tid==0 && *resize==1)
        //printf("insert--insert tkv\n");
}

//search kernel
__global__ void 
lp_search(TTT* key,TTT* ans,TTT size,TTT table_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid<size){
        ans[tid] =Search(key[tid], table_size);
        tid+=NUM_BLOCK*NUM_THREADS;
    }
}

__global__ void
lp_getnum(TTT table_size)
{
    int num=0;
    for(int i=0;i<table_size;i++){
        if(tkey[i]!=0)
            num++;
    }
    double load=(double)num/table_size;
    printf("the table size is %d\n",table_size);
    printf("the load is %lf\n",load);
    
}

__global__ void
rehash(TTT* key,TTT* value,TTT size,TTT* resize,TTT* rkey,TTT* rvalue,TTT old_size,TTT table_size){
//(TTT* key,TTT* value,TTT size,TTT* resize,TTT* rkey,TTT* rvalue,TTT table_size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //tmp tkv
    TTT* ikey=tkey;
    TTT* ivalue=tvalue;
    // copy with empty
    tkey=rkey;
    tvalue=rvalue;
//     if(tid==0)
//         printf("rehash-0-insert tkv\n");
        
    //insert table
    while(tid<old_size){
        if(ikey[tid]!=0)
            Add(ikey[tid],ivalue[tid],2*table_size,resize);
        tid+=NUM_BLOCK*NUM_THREADS;
    }
    
    tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(tid==0)
//         printf("rehash-1-insert tkv\n");
    
    //insert key
    while(tid<size){
        Add(key[tid],value[tid],table_size,resize);
        int a=Search(key[tid],table_size);
        if(a!=value[tid]){
           // printf("!!!error  \t%d %d s: %d\n",key[tid],value[tid],a);
        }
        tid+=NUM_BLOCK*NUM_THREADS;
    }

//     tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(tid==0)
//         printf("rehash-2-return tkv\n");
}

void gpu_rehash(TTT* key,TTT* value,TTT size,TTT old_size,TTT* resize,TTT &table_size){
    //malloc
    printf("----rehash size:  %d --> %d\n",size,table_size);
    TTT* d_key,*d_value;
    cudaMalloc((void**)&d_key, sizeof(TTT)*table_size);
    cudaMalloc((void**)&d_value, sizeof(TTT)*table_size);
    cudaMemset(d_key,0, sizeof(TTT)*table_size);
    cudaMemset(d_value,0, sizeof(TTT)*table_size);
    
    checkCudaErrors(cudaGetLastError());
    rehash<<<NUM_BLOCK,NUM_THREADS>>>(key,value,size,resize,d_key,d_value,old_size,table_size);
    checkCudaErrors(cudaGetLastError());
    
    
}


void gpu_lp_insert(TTT* key,TTT* value,TTT size,TTT* resize,TTT &table_size){
    if(size>table_size){
       int old_size=table_size;
       table_size=2*size;
       gpu_rehash(key,value,size,old_size,resize,table_size);
       return ;
    }
   
    //in main
    // st is you operator num
    unsigned int real_block=((unsigned int)size+NUM_THREADS-1)/NUM_THREADS;
    dim3 block=real_block>NUM_BLOCK ? NUM_BLOCK : real_block;

    //printf("start gpulpi\n");
    checkCudaErrors(cudaGetLastError());
    lp_insert<<<block,NUM_THREADS>>>(key,value,size,resize,table_size);
    int* a=new int[1];
    checkCudaErrors(cudaGetLastError());
    cudaMemcpy(a,resize,sizeof(TTT),cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
    if(*a==1){
       *a=0;
       int old_size=table_size;
       table_size *= 2;
       gpu_rehash(key,value,size,old_size,resize,table_size);
    }
    checkCudaErrors(cudaGetLastError());
    
  
}

//void gpu_lp_delete();

void gpu_lp_search(TTT* key,TTT* ans,TTT size,TTT table_size){
    unsigned int real_block=(size+NUM_THREADS-1)/NUM_THREADS;
    dim3 block=real_block>NUM_BLOCK ? NUM_BLOCK : real_block;
    lp_search<<<block,NUM_THREADS>>>(key,ans,size,table_size);
}

void gpu_lp_init(TTT* k,TTT* v){
    lp_init<<<1,1>>>(k,v);
}
void gpu_lp_getnum(TTT table_size){
    lp_getnum<<<1,1>>>(table_size);
}


