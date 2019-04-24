


#include "linear_probe.h"


// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)

//hash table
__device__ TTT* tkey;
__device__ TTT* tvalue;



//hash function
inline __device__  unsigned
hash_function(TTT key, TTT size)
{
    return key % _PRIME % size ;
}

//init set gpu hash table
__global__
void lp_init(TTT *k,TTT* v){
    tkey=k;
    tvalue=v;
}

// insert sub-fun
__device__   bool
Add(TTT k, TTT v,TTT table_size) {
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
    //*resize=1;
    return false;
}


__device__   bool
Delete(TTT k, TTT table_size) {
    
    int hash=hash_function(k,table_size);

    for(int attempt=0;attempt<MAX_ITERATOR;attempt++) {

        //tmp only return t/f
        if (tkey[hash] == k) 
        {
            tvalue[hash]=0;
            return true;
        }    
        // differ
        hash++;
        hash %= table_size;
        //hash += attempt * attempt;
    }

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




__global__ void kernel(TTT* key, TTT*value, TTT size,TTT table_size,TTT* op)
{

    //*resize=0;
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=size) return;
    while(tid<size){
        if(op[tid]==ADD){
      		//Insert(itm, &(result[op_id])); // return 1 or 0 or WRONG_POS(need rehash)
            Add(key[tid],value[tid],table_size);
   	 	} else if(op[tid]==DELETE){
            Delete(key[tid],table_size);
    	} else if(op[tid]==SEARCH){
            Search(key[tid], table_size);
    	}

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



void gpu_lp_kernel(TTT* key,TTT* value,TTT size,TTT table_size,TTT* op){
    unsigned int real_block=(size+NUM_THREADS-1)/NUM_THREADS;
    dim3 block=real_block>NUM_BLOCK ? NUM_BLOCK : real_block;
    kernel<<<block,NUM_THREADS>>>(key,value,size,table_size,op);
}

void gpu_lp_init(TTT* k,TTT* v){
    lp_init<<<1,1>>>(k,v);
}

void gpu_lp_getnum(TTT table_size){
    lp_getnum<<<1,1>>>(table_size);
}