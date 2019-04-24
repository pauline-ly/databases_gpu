//
// Created by jing on 2018/7/1.
//


#include "cuckoo.h"
#include <assert.h>
#include <device_launch_parameters.h>



////hash function
//inline __device__  unsigned
//hash_function(TTT key, TTT size)
//{
//    return key % _PRIME % size ;
//}

// TODO hash fun
/// table size is dynamic
__device__ TYPE
get_next_loc(TYPE k,TYPE v,TYPE num_table,cuckoo* table){
    num_table%=TABLE_NUM;
    //printf("hash: k:%d size:%d ,hash :%d \n",(( k%_PRIME )+num_table),(table->Lsize[num_table]/BUCKET_SIZE),(( k%_PRIME) +num_table) % (table->Lsize[num_table]/BUCKET_SIZE));
    return (( k%_PRIME )+num_table) % (table->Lsize[num_table]/BUCKET_SIZE);
}


__device__ void pbucket(bucket *b,int num,int hash){
    printf("table%d,%d \n",num,hash);
    for(int i=0;i<BUCKET_SIZE;i++){
        printf("%d,%d ",b->key[i],b->value[i]);
    }
    printf("\n");
}


__global__ void
cuckoo_insert(TTT* key, /// key to insert
              TTT* value, /// value to insert
              TTT size, /// insert size
              TTT* resize, /// insert error?
              cuckoo* table, /// hash table
              TTT table_size) {
    *resize = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /// for every k
    if(tid>= size) return;


    int lan_id = tid & 0x0000001f;
    int wrap_id = tid >> 5;

    while (tid < size) {

        int myk = key[tid];
        int myv = value[tid];
        int is_active = 1;/// mark for work

        int work_k = 0;
        int work_v = 0;

        /// for insert
        int hash;
        int hash_table_num;
        int ballot;


        /// for voting , TODO size??
        volatile __shared__ int wrap[(NUM_BLOCK * NUM_THREADS)>>5 ];

        /// while have work to do
        while (__any(is_active != 0)) {

            hash_table_num=0;

            //printf("lan_id: %d, active:%d \n",lan_id,is_active);

/// step1   start voting ==================================
            if (is_active != 0)
                wrap[wrap_id] = lan_id;

            work_k = myk;
            work_v = myv;
            /// over ======



/// step2   broadcast ====================================
            work_k=__shfl(work_k, wrap[wrap_id]);
            work_v=__shfl(work_v, wrap[wrap_id]);

           
/// step3   insert to the table ===========================
            hash_table_num = 0;
            hash = get_next_loc(work_k, work_v, 0,table);

            /// find null or too long
            for (int i = 0; i < MAX_ITERATOR; i++) {

/// step3.1     TODO:lock and compress

                //assert(0);
                /// block
                bucket *b = &(table->table[hash_table_num][hash]);

                assert(b);

/// step3.2     check exist & insert
                ballot = __ballot(b->key[lan_id] == work_k);   
                if (ballot != 0) { /// update
                    if (lan_id == wrap[wrap_id])
                        is_active = 0;
                    break;
                }


                
/// step3.3      check null & insert
                ballot = __ballot(b->key[lan_id] == 0);
                if (ballot != 0) {


                    // TODO: choose a lan to insert,now is the least
                    /// set kv
                    if (lan_id == __ffs(ballot)-1) {
                        b->key[lan_id] = work_k;
                        b->value[lan_id] = work_v;
                    }


                    /// mark active false

                    if (lan_id == wrap[wrap_id])
                        is_active = 0;

                    /// insert ok ,
                    break;
                }/// insert



/// step3.4     other,we need  cuckoo evict

                if(lan_id==wrap[wrap_id]){
                    work_k=b->key[lan_id];
                    work_v=b->value[lan_id];
                    b->key[lan_id]=myk;
                    b->value[lan_id]=myv;
                }
                __shfl(work_k, wrap[wrap_id]);
                __shfl(work_v, wrap[wrap_id]);


/// step3.5     keep evicted kv and reinsert
                hash = get_next_loc(work_k, work_v, hash_table_num++,table);

            }

        }


        tid += NUM_BLOCK * NUM_THREADS;
    }
}



//__global__ void
//lp_rehash()(cu_linear_probe *t){
//    table=t;
//
//}

__global__ void
rehash(TTT* rkey,TTT* rvalue,TTT old_size,TTT table_size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;



    //insert table
    while(tid<old_size){


        tid+=NUM_BLOCK*NUM_THREADS;
    }
}

void gpu_rehash(TTT old_size,TTT new_table_size){
    //malloc
    printf("----rehash size:  %d --> %d\n",old_size,new_table_size);
    TTT* d_key,*d_value;
    cudaMalloc((void**)&d_key, sizeof(TTT)*new_table_size);
    cudaMalloc((void**)&d_value, sizeof(TTT)*new_table_size);
    cudaMemset(d_key,0, sizeof(TTT)*new_table_size);
    cudaMemset(d_value,0, sizeof(TTT)*new_table_size);
    
    checkCudaErrors(cudaGetLastError());
    rehash<<<NUM_BLOCK,NUM_THREADS>>>(d_key,d_value,old_size,new_table_size);
    checkCudaErrors(cudaGetLastError());


}


__global__ void show_table(cuckoo* table){
    if(blockIdx.x * blockDim.x + threadIdx.x>0 ) return;

    for(int i=0;i<TABLE_NUM;i++){
        printf("\n\n\ntable:%d\n",i);
        for(int j=0;j<(table->Lsize[i])/BUCKET_SIZE;j++){
            for(int t=0;t<BUCKET_SIZE;t++)
                printf(" %d,%d ",table->table[i][j].key[t],table->table[i][j].value[t]);
            printf("\n");
        }

    }
}


void gpu_lp_insert(TTT* key,TTT* value,TTT size,TTT* resize,cuckoo *table,TTT &table_size){

   
    //in main
    // st is you operator num
    unsigned int real_block=((unsigned int)size+NUM_THREADS-1)/NUM_THREADS;
    dim3 block=real_block>NUM_BLOCK ? NUM_BLOCK : real_block;

    //printf("start gpulpi\n");
    checkCudaErrors(cudaGetLastError());
    cuckoo_insert<<<block,NUM_THREADS>>>(key,value,size,resize,table,table_size);
    int* a=new int[1];
    checkCudaErrors(cudaGetLastError());
    cudaMemcpy(a,resize,sizeof(TTT),cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
//    if(*a != 0){
//        *a=0;
//        printf("need resize:！！！");
//        int old_size=table_size;
//        table_size *= NUM_grow_ratio;
//        gpu_rehash(old_size,table_size);
//        gpu_lp_insert(key,value,size,resize,table_size);
//    }
    checkCudaErrors(cudaGetLastError());

    //show_table<<<1,1>>>(table);
    
  
}

//void gpu_lp_delete();

void gpu_lp_search(TTT* key,TTT* ans,TTT size,TTT table_size){
    unsigned int real_block=(size+NUM_THREADS-1)/NUM_THREADS;
    dim3 block=real_block>NUM_BLOCK ? NUM_BLOCK : real_block;

}




