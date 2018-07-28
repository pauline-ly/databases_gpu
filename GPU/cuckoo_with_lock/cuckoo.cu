//
// Created by jing on 2018/7/1.
//


#include "cuckoo.h"
#include <assert.h>
#include <device_launch_parameters.h>





/// hash function
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


__device__ void pbucket(bucket *b,int num,int hash,int t_size){
    printf("table%d,%d/%d \n",num,hash,t_size);
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

#if    head_info
if(tid==0) {
    printf("\n\ninsert kernel:\nsize:%d ,tsize:%d \n", size, table_size);
    printf("table:%x ,t1:%x,t2:%x,t3:%x,t4:%x,t5:%x\n",
           table, table->table[0], table->table[1], table->table[2], table->table[3], table->table[4]);
}
#endif

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
        int old_cas;

#if 0
if(lan_id==0)
        printf("\nstart insert kv:%d %d ,my wrap_id:%d ,lan_id: %d \n", wrap_id, lan_id, myk, myv);
#endif
        /// for voting , TODO size??
        volatile __shared__ int wrap[(NUM_BLOCK * NUM_THREADS)>>5 ];

        /// while have work to do
        while (__any(is_active != 0)) {

            hash_table_num=0;

#if insert_debug
            //printf("lan_id: %d, active:%d \n",lan_id,is_active);
ballot=__ballot(is_active != 0);
if(lan_id==0)
printf("\n\n\nstart insert  ,my wrap_id:%d ,ballot:%x\n" ,wrap_id,ballot);
#endif

/// step1   start voting ==================================
            if (is_active != 0 && wrap[wrap_id]!=lan_id )
                wrap[wrap_id] = lan_id;
#if insert_debug
if(lan_id==0)
    printf("voting: %d   ",wrap[wrap_id] );
#endif

            work_k = myk;
            work_v = myv;
            /// over ======
#if insert_debug
if(lan_id==0)
printf("step 2  ", wrap_id, lan_id, myk, myv,wrap[wrap_id]);
#endif

/// step2   broadcast ====================================
            work_k=__shfl(work_k, wrap[wrap_id]);
            work_v=__shfl(work_v, wrap[wrap_id]);

#if insert_debug
            if(lan_id==0)
                printf("kv: %d ,%d  ",work_k,work_v );
#endif
/// step3   insert to the table ===========================
            hash_table_num = work_k%TABLE_NUM;
            hash = get_next_loc(work_k, work_v, hash_table_num,table);

            /// find null or too long
            for (int i = 0; i < MAX_ITERATOR; i++) {

/// step3.1     TODO:lock and compress
                ///  lock  ,otherwise revote
                if(lan_id==wrap[wrap_id]){
                    old_cas=atomicCAS(&(table->Lock[hash_table_num*(table->Lsize[hash_table_num]/32)+hash]),0,1);
                }
                old_cas=__shfl(old_cas, wrap[wrap_id]);
                if(old_cas==1) break;
//                if(lan_id==0)
//                    printf("work lock:%d / %d \n",hash_table_num*(table->Lsize[hash_table_num]/32)+hash,(table->Lsize[hash_table_num]/32)*TABLE_NUM);
                //assert(0);


                /// block
                bucket *b = &(table->table[hash_table_num][hash]);

                assert(b);
#if 0
if(lan_id==0)
printf("step_3.2 hash:%d  ",hash, wrap_id, lan_id, myk, myv);
#endif

                //printf("lan_id: %d, active:%d , b[key]: %d ,work_k %d \n",lan_id,is_active,b->key[lan_id],work_k);
/// step3.2     check exist & insert
                ballot = __ballot(b->key[lan_id] == work_k);   
                if (ballot != 0) { /// update

                    if(lan_id==wrap[wrap_id]){
#if debug_big_data
                        printf("update: key:%d,v:%d->%d\tt:%d,h:%d\n",myk,b->value[lan_id],myv,hash_table_num,hash);
#endif
                        b->value[lan_id]=myv;

                    }
#if insert_debug
if(lan_id==0)
printf("3.2.1 workK: %d ,work_kv :%d ,vot: %d ,ballot:%d  \n", work_k,work_v,wrap[wrap_id],ballot);
#endif
                    if (lan_id == wrap[wrap_id])
                        is_active = 0;
                    table->Lock[hash_table_num*(table->Lsize[hash_table_num]/32)+hash]=0;
                    break;
                }


                ballot = __ballot(b->key[lan_id] == 0);

#if insert_debug
if(lan_id==0)
printf("3.3 ballot:%x \n ", ballot,wrap_id, lan_id, myk, myv);
if(lan_id==0)
    pbucket(b,hash_table_num,hash,table->Lsize[hash_table_num]/32);
#endif

/// step3.3      check null & insert
                if (ballot != 0) {

#if insert_debug
if(lan_id==0)
printf("3.3.1 my wrap_id:%d,ballot: %x ,ffs:%d\n", wrap_id, ballot,__ffs(ballot)-1,lan_id, myk, myv);
#endif

                    // TODO: choose a lan to insert
                    /// set kv

                    if (lan_id == __ffs(ballot)-1) {
#if debug_big_data
                        printf("insert null: bucket:%d,%d kv:%d,%d\tt:%d,h:%d\n",b->key[lan_id],b->value[lan_id],work_k,work_v,hash_table_num,hash);
#endif
                        b->key[lan_id] = work_k;
                        b->value[lan_id] = work_v;
                    }


                    /// mark active false

                    if (lan_id == wrap[wrap_id])
                        is_active = 0;

                    table->Lock[hash_table_num*(table->Lsize[hash_table_num]/32)+hash]=0;
                    /// insert ok ,
                    break;
                }/// insert

#if insert_debug
if(lan_id==0)
printf("3.4  ", wrap_id, lan_id, myk, myv);
#endif

/// step3.4     other,we need  cuckoo evict
                if(lan_id==wrap[wrap_id]){
#if debug_big_data
                    printf("insert evict: %d,%d -> %d,%d\tt:%d,h:%d\n",b->key[lan_id],b->value[lan_id],work_k,work_v,hash_table_num,hash);
#endif
                    work_k=b->key[lan_id];
                    work_v=b->value[lan_id];
                    b->key[lan_id]=myk;
                    b->value[lan_id]=myv;
                }
                work_k=__shfl(work_k, wrap[wrap_id]);
                work_v=__shfl(work_v, wrap[wrap_id]);
#if insert_debug
if(lan_id==0)
    printf("3.5  work_kv: %d %d\n",work_k,work_v);
#endif

                table->Lock[hash_table_num*(table->Lsize[hash_table_num]/32)+hash]=0;
/// step3.5     keep evicted kv and reinsert
                hash_table_num++;
                hash_table_num%=TABLE_NUM;
                hash = get_next_loc(work_k, work_v,hash_table_num ,table);

            }

        }


        tid += NUM_BLOCK * NUM_THREADS;
    }
}



__global__ void
cuckoo_search(TTT* key, /// key to s
              TTT* value, /// value to key
              TTT size, /// s size
              cuckoo* table) { /// hash table
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /// for every k
#if head_info
    if(tid==0) {
        printf("\n\nfind kernel\n\nsize:%d\n", size);
        printf("table:%x ,t1:%x,t2:%x,t3:%x,t4:%x,t5:%x\n",
               table, table->table[0], table->table[1], table->table[2], table->table[3], table->table[4]);
    }
#endif
    int lan_id = tid & 0x0000001f;
    int wrap_id = tid >> 5;

    while (tid < size) {

        int myk = key[tid];

        int is_active = 1;/// mark for work

        int work_k = 0;
        int work_v;


        /// for s
        int hash;
        int hash_table_num;
        int ballot;
        bucket *b;
#if head_info
        if(lan_id==0)
            printf("my wrap_id:%d ,lan_id: %d,k:%d \n", wrap_id, lan_id, myk);
#endif
        /// for voting , TODO size??
        volatile __shared__ int wrap[(NUM_BLOCK * NUM_THREADS)>>5 ];

        /// while have work to do
        while (__any(is_active != 0)) {

            hash_table_num=0;

            //printf("lan_id: %d, active:%d \n",lan_id,is_active);
            ballot=__ballot(is_active != 0);
#if search_debug
            if(lan_id==0)
                printf("\n\n1 my wrap_id:%d  work_kv:%d %d ,ballot:%x\n", wrap_id, work_k,work_v,ballot);
#endif

/// step1   start voting ==================================
            if (is_active != 0)
                wrap[wrap_id] = lan_id;
#if search_debug
            if(lan_id==0)
                printf("voting: %d\t",wrap[wrap_id] );
#endif

            work_k = myk;

            /// over ======


/// step2   broadcast ====================================
            work_k=__shfl(work_k, wrap[wrap_id]);


            //printf("lan_id: %d, active:%d  ,work_k %d \n",lan_id,is_active,work_k);
/// step3   find in 5 table ===========================
            hash_table_num = work_k % TABLE_NUM;
            hash = get_next_loc(work_k, work_v, hash_table_num,table);

            /// find null or too long
            for (int i = 0; i < TABLE_NUM; i++) {
                b=&table->table[hash_table_num][hash];
                ballot=__ballot(b->key[lan_id]==work_k);

                /// find it
                if(ballot!=0){
                    if(lan_id==wrap[wrap_id]){
                        value[tid]=b->value[__ffs(ballot)-1];
#if search_debug
                        printf("find %d: %d\n",key[tid],value[tid]);
#endif
                        is_active=0;
                    }
                    break;
                }
                hash_table_num++;
                hash_table_num%=TABLE_NUM;
                hash=get_next_loc(work_k, work_v,hash_table_num,table);
            }
            if(lan_id==wrap[wrap_id]){
                is_active=0;
            }
        }
        tid += NUM_BLOCK * NUM_THREADS;
    }

}


/// TODO: reshah
__global__ void
rehash(TTT* rkey,TTT* rvalue,TTT old_size,TTT table_size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;



    //insert table
    while(tid<old_size){


        tid+=NUM_BLOCK*NUM_THREADS;
    }
}


/// TODO: rehash
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



/// show table by key,value
__global__ void show_table(cuckoo* table){
    if(blockIdx.x * blockDim.x + threadIdx.x>0 ) return;
    for(int i=0;i<TABLE_NUM;i++){
        printf("\n\n\ntable:%d\n",i);
        for(int j=0;j<(table->Lsize[i])/BUCKET_SIZE;j++){
            for(int t=0;t<BUCKET_SIZE;t++) {
                if(t==16) printf("\n    ");
                printf(" %d,%d ", table->table[i][j].key[t], table->table[i][j].value[t]);
            }
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
//    checkCudaErrors(cudaGetLastError());
    cuckoo_insert<<<block,NUM_THREADS>>>(key,value,size,resize,table,table_size);
#if __show_table
    show_table<<<1,1>>>(table);
#endif
    int* a=new int[1];
    cudaDeviceSynchronize();
//    checkCudaErrors(cudaGetLastError());
    cudaMemcpy(a,resize,sizeof(TTT),cudaMemcpyDeviceToHost);
//    checkCudaErrors(cudaGetLastError());
//    if(*a != 0){
//        *a=0;
//        printf("need resize:！！！");
//        int old_size=table_size;
//        table_size *= NUM_grow_ratio;
//        gpu_rehash(old_size,table_size);
//        gpu_lp_insert(key,value,size,resize,table_size);
//    }
//    checkCudaErrors(cudaGetLastError());


  
}

//void gpu_lp_delete();

void gpu_lp_search(TTT* key,TTT* ans,TTT size,cuckoo* table){
    unsigned int real_block=(size+NUM_THREADS-1)/NUM_THREADS;
    dim3 block=real_block>NUM_BLOCK ? NUM_BLOCK : real_block;

//    checkCudaErrors(cudaGetLastError());
    cuckoo_search<<<block,NUM_THREADS>>>(key,ans,size,table);
    cudaDeviceSynchronize();
//    checkCudaErrors(cudaGetLastError());

}




