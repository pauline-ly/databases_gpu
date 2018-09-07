//
// Created by jing on 2018/7/1.
//

// add vscode sup



#include "cuckoo.h"
#include <assert.h>
#include <device_launch_parameters.h>

// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)

#define debug_num 30

#define single_BUCKET 15629 


/// hash table
__constant__ cuckoo table;

#define  get_table_length(i)  get_table_bucket_length(i)
#define  get_table_bucket_length(i) (table.Lsize[i]/BUCKET_SIZE)
/// Lsize0 is the biggest
#define  Lock_pos(num,hash) ((num) * (get_table_length(0)) + hash)


/// hash functiong
__device__ __forceinline__ TYPE
get_next_loc(TYPE k,
             TYPE v,
             TYPE num_table)
{
    return (k^(table.hash_fun[num_table].x) + table.hash_fun[num_table].y) % _PRIME %get_table_length(num_table);
}


__device__ void pbucket(bucket *b,int num,int hash,int t_size)
{
    printf("table.%d,%d/%d \n",num,hash,t_size);
    for(int i=0;i<BUCKET_SIZE;i++){
        if(i%8==0) printf("\n\t");
        printf("%d,%d ",b->key[i],b->value[i]);
    }
    printf("\n");
}




__global__ void
cuckoo_insert(TTT* key, /// key to insert
              TTT* value, /// value to insert
              TTT size, /// insert size
              TTT* resize) /// insert error?
{
    *resize = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /// for every k
    if (tid >= size) return;

#if head_info
    if(tid==0) {
    printf(">>>insert kernel:\n>>>size:%d  \n", size);
    printf(">>>s_size:t1:%d, t2:%d, t3:%d, t4:%d\n",
            table.Lsize[0], table.Lsize[1], table.Lsize[2], table.Lsize[3]);
    }
#endif


    /// warp cooperation
    int lan_id = threadIdx.x & 0x0000001f;
    int warp_id = threadIdx.x >> 5;


    /// work kv
    TTT work_k, work_v ;

    ///  leader thread num
    int ansv;

    
    /// keep kv every thread
    TTT myk, myv;
    /// add when evict ,set to 0 when exist or null
    TTT evict_time_of_one_thread =0;

    /// for insert
    int hash;
    int hash_table_num=0;

    /// using for ballot & CAS
    int tmp;


    volatile __shared__ int warp[ (NUM_THREADS) >> 5 ];


    while (tid < size) {

        evict_time_of_one_thread =0;

        int is_active = 1;/// mark for work 
        myk = key[tid];
        myv = value[tid];

        /// while have work to do
        while (__any(is_active != 0)) {

            /// TODO: there need check resize?

/// step1   start voting ==================================
            /// if there is one active thread  , work
            if (is_active != 0 && warp[warp_id]!=lan_id )
                warp[warp_id] = lan_id;


            ansv = warp[warp_id];
            work_k = myk;
            work_v = myv;
            /// over ======


/// step2   broadcast ====================================
            work_k = __shfl(work_k, ansv);
            work_v = __shfl(work_v, ansv);

#if insert_debug
            if(lan_id==0 && work_k==debug_num)
                printf("2-kv: %d ,%d  \n",work_k,work_v);
#endif

/// step3   insert to the table. ===========================
            hash_table_num ++;
            hash_table_num %= TABLE_NUM;
            hash = get_next_loc(work_k, work_v, hash_table_num);




/// step3.1   TODO: compress
            ///  lock  ,otherwise revote
            if (lan_id == ansv) {
                /// TODO: different length need to sum ,using double length
                /// tmp 0: free 1: busy
                tmp = atomicCAS(&(table.Lock[Lock_pos(hash_table_num,hash)]), 0, 1);
            }//end if
            tmp = __shfl(tmp, ansv);
            if (tmp == 1)
                continue;


#if insert_debug

            if(lan_id==0 && work_k==debug_num)
                printf("cas-kv: %d ,%d ,cas:%d \n",work_k,work_v,tmp);
#endif


            /// block
            bucket *b = &(table.table[hash_table_num][hash]);



            //printf("lan_id: %d, active:%d , b[key]: %d ,work_k %d \n",lan_id,is_active,b->key[lan_id],work_k);
/// step3.2     check exist & insert
            tmp = __ballot(b->key[lan_id] == work_k);
#if insert_debug
            if(lan_id==0 && work_k==debug_num)
                printf("start-exist-kv: %d ,%d ,ballot:%x \n",work_k,work_v,tmp);
#endif
            if (tmp != 0) { /// update
                if (lan_id == ansv) {
                    /// update value
                    b->value[__ffs(tmp) - 1] = myv;
                    is_active = 0;
                    evict_time_of_one_thread=0;
                }// end if ,upadte

#if insert_debug
                if(lan_id==0 && work_k==debug_num)
                    printf("exist-kv: %d ,%d ,ballot:%x \n",work_k,work_v,tmp);
#endif
                /// TODO: the lock free: one thread / all thread
                table.Lock[Lock_pos(hash_table_num,hash)] = 0;
                continue;
            }//end check update

/// step3.3      check null & insert
            tmp = __ballot(b->key[lan_id] == 0);

#if insert_debug
            if(lan_id==0 && work_k==debug_num) {
                printf("start-null-kv: %d ,%d ,ballot:%x \n", work_k, work_v, tmp);
                pbucket(b, hash_table_num, hash, get_table_length(hash_table_num));
            }
#endif

            if (tmp != 0) {

#if insert_debug
                if(lan_id==0 && work_k==debug_num)
                    printf("null-kv: %d ,%d ,ballot:%x \n",work_k,work_v,tmp);
#endif
                /// set kv
                if (lan_id == __ffs(tmp) - 1) {
                    b->key[lan_id] = work_k;
                    b->value[lan_id] = work_v;
                }// insert


                /// mark active false

                if (lan_id == ansv){
                    evict_time_of_one_thread=0;
                    is_active = 0;
                    table.Lock[Lock_pos(hash_table_num,hash)]= 0;
                }


                /// insert ok ,
                continue;
            }/// null insert over

#if insert_debug
            if(lan_id==0 && work_k==debug_num)
                printf("evict-kv: %d ,%d ,ballot:%x \n",work_k,work_v,tmp);
#endif

/// step3.4     other,we need  cuckoo evict
                if (lan_id == ansv) {

                    myk = b->key[lan_id];
                    myv = b->value[lan_id];
                    b->key[lan_id] = work_k;
                    b->value[lan_id] = work_v;
                    evict_time_of_one_thread++;
                    /// when one always get leader , mark rehash
                    if(evict_time_of_one_thread >= MAX_ITERATOR){
                        *resize=1;
                        evict_time_of_one_thread=0;
                        is_active=0;
                    }
                } // evict


                table.Lock[Lock_pos(hash_table_num,hash)] = 0;
/// step3.5     keep evicted kv and reinsert




//        /// can not insert in here
//        if (warp[warp_id] == lan_id){
//            is_active = 0;
//            *resize=1;
//        }

        }


        tid += NUM_BLOCK * NUM_THREADS;
    }
}



__global__ void
cuckoo_search(TTT* key, /// key to s
              TTT* value, /// value to key
              TTT size) /// s size
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    /// for every k
#if head_info
    if(tid==0) {
    printf(">>>search kernel:\n>>>size:%d  \n", size);
    printf(">>>s_size:t1:%d, t2:%d, t3:%d, t4:%d\n",
            table.Lsize[0], table.Lsize[1], table.Lsize[2], table.Lsize[3]);
    }
#endif
    int lan_id = threadIdx.x & 0x0000001f;
    int warp_id = threadIdx.x >> 5;

    int myk;

    int is_active;

    int work_k = 0;
    int work_v;


    /// for s
    int hash;
    int hash_table_num;
    int ballot;
    bucket *b;
    /// for voting ,using in one block
    volatile  __shared__ int warp[( NUM_THREADS)>>5 ];

    while (tid < size) {

        myk = key[tid];
        is_active = 1;/// mark for work




        

        /// while have work to do
        while (__any(is_active != 0)) {

            hash_table_num=0;

            //printf("lan_id: %d, active:%d \n",lan_id,is_active);
            ballot=__ballot(is_active != 0);


/// step1   start voting ==================================
            if (is_active != 0)
                warp[warp_id] = lan_id;
#if search_debug
            if(lan_id==0)
                printf("voting: %d\t",warp[warp_id] );
#endif

            work_k = myk;

            /// over ======


/// step2   broadcast ====================================
            work_k=__shfl(work_k, warp[warp_id]);


            //printf("lan_id: %d, active:%d  ,work_k %d \n",lan_id,is_active,work_k);
/// step3   find in 5 table ===========================
            hash_table_num = work_k % TABLE_NUM;
            hash = get_next_loc(work_k, work_v, hash_table_num);

            /// find null or too long
            for (int i = 0; i < TABLE_NUM; i++) {
                b=&table.table[hash_table_num][hash];
                ballot=__ballot(b->key[lan_id]==work_k);

                /// find it
                if(ballot!=0){
                    if(lan_id==warp[warp_id]){
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
                hash=get_next_loc(work_k, work_v,hash_table_num);
            }
            if(lan_id==warp[warp_id]){
                is_active=0;
            }
        }
        tid += NUM_BLOCK * NUM_THREADS;
    }

}

void __global__
cuckoo_resize_up(bucket* old_table, /// new table has been set to table
                 int old_size,
                 TTT num_table_to_resize)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
///step1 取新表  ======================
    bucket* new_table=table.table[num_table_to_resize];
    /// end ,next : old->new


    bucket *b;
    /// take kv to insert
    TTT key,value;
    /// insert position
    int hash;
    old_size/=BUCKET_SIZE;

///step2 每个线程处理一个bucket ======================
    while(tid<old_size){

///step2.1  获取自己的bucket ======================
        /// tid is hash_value
        b=&old_table[tid];


///step2.2 对bucket中各插入对应的位置======================

        for(int i=0;i<BUCKET_SIZE;i++){
            key=b->key[i];
            if(key==0) continue;
            value=b->value[i];
            /// how to use tid & hash fun
            hash=get_next_loc(key,value,num_table_to_resize);



            /// store to hash loc
            new_table[hash].key[i]=key;
            new_table[hash].value[i]=value;
        }

        tid += NUM_BLOCK * NUM_THREADS;
    }

}

void __global__
cuckoo_resize_down(bucket* old_table,  /// small
                   int old_size,
                   int num_table_to_resize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

#if head_info
    if(tid==0) {
        printf(">>>down_size kernel: %d->%d\n",old_size,table.Lsize[num_table_to_resize]);
        printf(">>>s_size:t1:%d, t2:%d, t3:%d, t4:%d\n",
               table.Lsize[0], table.Lsize[1], table.Lsize[2], table.Lsize[3]);
    }
#endif

    bucket *b=NULL;
    bucket *des_b=NULL;

    /// take kv to insert
    TTT key, value;

    /// insert position
    int hash;
    int new_bucket_size = table.Lsize[num_table_to_resize] / BUCKET_SIZE;

    /// warp coopration
    int warp_num_in_all = tid >> 5;
    int warp_num_in_block = threadIdx.x >> 5;
    int lan_id = tid & 0x1f;
    int is_active;
    int ballot;

    /// in block , for voting
    volatile __shared__ int warp[(NUM_THREADS) >> 5];

///step1 置换新表  ======================
    /// 与新表对应的表长已设置好
    bucket *new_table = table.table[num_table_to_resize];
    /// end ,next : old->new

#if down_size_debug
if (tid==0)
    printf("step start \n");
#endif

///step2 每个warp处理2个bucket->一个bucket ======================
    /// 分别将 旧表 tid tid+new_bucket_size 两个bucket插入到新表的 tid bucket中

    /// PROBLEM： 这里默认 new_bucket_size * 2 = old_size (api.cpp line 47)
    /// 方法 部分条件下可将old_size 设置为偶数，这样只有在多次downsize之后才会不符合上述条件

    /// PROBLEM: 将两个bucket映射到一个bucket，在元素较多的情况下势必造成部分
    ///   溢出，除了将溢出部分插入到其他表，我们还需要合理安排两个到一个的映射关系使之高
    ///   效转换。
    /// 方法1. 逐个查询，使用原子add
    /// 方法2. 对空位置和非空kv scan，直接得到相应位置，：需要shared或其他数组支持
    /// 方法3. 首先进行简单插入，然后使用warp通信找到空位置插入

    /// one thread one block
    while (warp_num_in_all < new_bucket_size) {  /// new size is smaller


///step2.1 获取新表的bucket  ======================
        /// warp_num_in_all is hash_value
        des_b = &new_table[warp_num_in_all];
#if down_size_debug
        if (tid==0)
            printf("step2.1 start \n");
#endif

///step2.2  获取第一个旧表的bucket ======================
        /// 读入第一个bucket中kv到变量
        b = &old_table[warp_num_in_all];
        key = b->key[lan_id];
        value = b->value[lan_id];
#if down_size_debug
        if(tid==0){
            printf("old table1\n");
            pbucket(b,0,0,0);
        }
        if(warp_num_in_all==0)
            printf("b1-%d: %d,%d\n",lan_id,key,value);
#endif

        int crose_lan_id=31-lan_id;
        /// 空kv再此读入第二个bucket
        b = &old_table[warp_num_in_all + new_bucket_size];
        if (key == 0) {
            key = b->key[crose_lan_id];
            value = b->value[crose_lan_id];
        }

        ///到这里，第一个bucket全部会被读入后面接着写入，第二个部分还未读入

#if down_size_debug
        if(tid==0){
            printf("old table2\n");
            pbucket(b,0,0,0);
        }
        if(warp_num_in_all==0)
            printf("b1-%d: %d,%d\n",lan_id,key,value);
#endif

///step2.3   将不为空的kv插入新表=====================
        des_b->key[lan_id] = key;
        des_b->value[lan_id] = value;
#if down_size_debug
        if(tid==0)
            printf("write\n");
        if(tid==0)
            pbucket(des_b,num_table_to_resize,tid,get_table_length(num_table_to_resize));
#endif

        is_active=0;



///step2.4  读取第二个bucket中未存入的kv ======================
        if (key != b->key[crose_lan_id]  /// 从未写入过
            && b->key[crose_lan_id] !=0)  /// 存在值
        {
            key = b->key[crose_lan_id];
            value = b->value[crose_lan_id];
            is_active = 1;
        }
#if down_size_debug
        if(warp_num_in_all==0)
            printf("b1-%d: %d,%d - %d\n",lan_id,key,value,is_active);
#endif
///step2.5 对新bucket还有的空位进行插入kv======================
        /// PROBLEM: how about skip step2.5 use step3 directly

        /// 如果空位置比较少会比较快，否则可能使用scan会更快
        /// 如果还有空位

        ballot = __ballot(des_b->key[lan_id] == 0);
#if down_size_debug
        if( tid==0 && ballot == 0 )
            printf("step 2.5 , full\n");
#endif

        while (__any(des_b->key[lan_id] == 0)) {
#if down_size_debug
            if(tid==0)
                printf("step 2.5 \n");
#endif
            if(!__any(is_active==1)) break;
#if down_size_debug
            if(tid==0)
                pbucket(des_b,num_table_to_resize,tid,get_table_length(num_table_to_resize));
#endif


            /// 找出空位
            ballot = __ballot(des_b->key[lan_id] == 0);

            /// use hash as tmp to decrease register
            /// 选一个空位
            hash = __ffs(ballot) - 1;

            /// 选一个线程
            if (is_active == 1)
                warp[warp_num_in_block] = lan_id;

            /// insert
            if (warp[warp_num_in_block] == lan_id) {
                des_b->key[hash] = key;
                des_b->value[hash] = value;
                is_active=0;
            }
        }


///step 3  如果位将第二个表中元素全部插入完成，插入到其他表中======================
#if down_size_debug
        if (tid==0)
            printf("step3 start \n");
#endif
        /// key value has kv to insert
        TTT work_k,work_v;
        int hash_table_num=num_table_to_resize;
        int ansv;
#if down_size_casinsert_debug
        if(warp_num_in_all==0) {
            printf("b1-%d: %d,%d - %d\n",lan_id,key,value,is_active);
        }
#endif

        while (__any(is_active != 0)) {
            /// using logic of  cuckoo_insert  (__global__)
            /// how to reuse the code ?

            /// prepare to be leader
            /// key value  hold  kv to insert
            /// so evict kv is sent key value
            work_k = key;
            work_v = value;

/// step3.1 start voting ==================================
            if (is_active != 0)//&& warp[warp_num_in_block]!=lan_id )
                warp[warp_num_in_block] = lan_id;


            /// leader is ansv
            ansv = warp[warp_num_in_block];


/// step3.2   broadcast ====================================
            work_k = __shfl(work_k, ansv);
            work_v = __shfl(work_v, ansv);

/// step3.3   insert to the table. ===========================
            hash_table_num ++;
            hash_table_num %= TABLE_NUM;
            hash = get_next_loc(work_k, work_v, hash_table_num);

/// step3.4   lock   TODO: compress  ===========================
            /// using ballot as tmp to decrease register
            ///  lock  ,otherwise revote
            if (lan_id == ansv) {
                /// TODO: different length need to sum ,tmp using double length
                ballot = atomicCAS(&(table.Lock[Lock_pos(hash_table_num,hash)]), 0, 1);
            }//end if
            ballot = __shfl(ballot, ansv);
            if (ballot == 1)
                continue;

            b = &(table.table[hash_table_num][hash]);

#if down_size_casinsert_debug
            ballot=__ballot(is_active==1);
            if(tid==0){
                printf("\n\nactive ballot:%x kv %d,%d ansv:%d\n",
                       ballot,work_k,work_v,ansv );
                pbucket(b,hash_table_num,hash,get_table_length(hash_table_num));

            }
#endif
/// step3.5   check exist & insert

            ballot = __ballot(b->key[lan_id] == work_k);
            if (ballot != 0) { /// update

                if (lan_id == ansv) {
                    b->value[__ffs(ballot) - 1] = value;
                    is_active = 0;
#if down_size_casinsert_debug
                    if(warp_num_in_all==0) {
                        printf("exit after insert \n");
                        pbucket(b, hash_table_num, hash, get_table_length(hash_table_num));
                    }
#endif
                    table.Lock[Lock_pos(hash_table_num,hash)] = 0;
                }// end if ,upadte

                continue;
            }//end check update

/// step3.6   check null & insert
            ballot = __ballot(b->key[lan_id] == 0);

#if down_size_casinsert_debug
            if(warp_num_in_all==0)  printf("%d,",lan_id);
            if(tid==0){
                printf("\n\nnull ballot:%x kv %d,%d ansv:%d \n",
                       ballot,work_k,work_v,ansv);
            }
#endif
            if (ballot != 0) {

                /// set kv
                if (lan_id == __ffs(ballot) - 1) {
                    b->key[lan_id] = work_k;
                    b->value[lan_id] = work_v;

                    /// free
                    table.Lock[Lock_pos(hash_table_num,hash)] = 0;
#if down_size_casinsert_debug
                    if(warp_num_in_all==0) {
                        printf("null after insert \n");
                        pbucket(b, hash_table_num, hash, get_table_length(hash_table_num));
                    }
#endif
                }// insert

                /// mark active false
                if (lan_id == ansv)
                    is_active = 0;

                continue;
            }/// null insert over


/// step3.7     other,we need  cuckoo evict
            if (lan_id == ansv){
                key = b->key[lan_id];
                value = b->value[lan_id];
                b->key[lan_id] = work_k;
                b->value[lan_id] = work_v;
#if down_size_casinsert_debug
                if(warp_num_in_all==0) {
                    printf("evict after insert \n");
                    pbucket(b, hash_table_num, hash, get_table_length(hash_table_num));
                }
#endif
                table.Lock[Lock_pos(hash_table_num,hash)] = 0;
            } // evict


        }

        /// TODO:auto configure ,what should be add to tid
        tid += NUM_BLOCK * NUM_THREADS;
        warp_num_in_all = tid >> 5;
    }

}


void GPU_cuckoo_resize_up(int num_table_to_resize,int old_size,bucket* new_table,cuckoo *h_table,cudaStream_t stream){

//    checkCudaErrors(cudaGetLastError());
    int new_size=old_size*2;

    ///  set table & size it needed
    bucket* old_table=h_table->table[num_table_to_resize];
    h_table->Lsize[num_table_to_resize]=new_size;
    h_table->table[num_table_to_resize]=new_table;
    cudaMemcpyToSymbolAsync(table,h_table,sizeof(cuckoo),0,cudaMemcpyHostToDevice,stream);

//    checkCudaErrors(cudaGetLastError());
    /// TODO: auto configure
    /// kernel Configuration

    dim3 block=choose_block_num(old_size);

    /// kernel launch


    GpuTimer timer;
    timer.Start();

    cuckoo_resize_up<<<block,NUM_THREADS>>>(old_table,old_size,num_table_to_resize);

    if(stream==0){
        cuckoo_resize_up<<<block,NUM_THREADS>>>(old_table,old_size,num_table_to_resize);
    }else{
        cuckoo_resize_up<<<block,NUM_THREADS ,0,stream>>>(old_table,old_size,num_table_to_resize);
    }
    cudaDeviceSynchronize();
    timer.Stop();

    /// sec
    double diff = timer.Elapsed()*1000000;
    printf("kernel <<<upsize>>>：the time is %.2lf us, ( %.2f Mops)s\n",
           (double)diff, (double)(new_size) / diff);

    checkCudaErrors(cudaGetLastError());
}

void GPU_cuckoo_resize_down(int num_table_to_resize,int old_size,bucket* new_table,cuckoo *h_table,cudaStream_t stream){

    printf("starting down_size \n");


    int new_size=old_size/2;


    ///  set table & size it needed
    bucket* old_table=h_table->table[num_table_to_resize];
    h_table->Lsize[num_table_to_resize]=new_size;
    h_table->table[num_table_to_resize]=new_table;
    cudaMemcpyToSymbolAsync(table,h_table,sizeof(cuckoo),0,cudaMemcpyHostToDevice,stream);

    /// TODO: auto configure
    /// kernel Configuration
    dim3 block=choose_block_num(old_size);

    GpuTimer ktimer;

    ktimer.Start(stream);

    /// kernel launch
    if(stream==0){
        cuckoo_resize_down<<<block,NUM_THREADS>>>(old_table,old_size,num_table_to_resize);
    }else{
        cuckoo_resize_down<<<block,NUM_THREADS,0,stream>>>(old_table,old_size,num_table_to_resize);
    }

    ktimer.Stop(stream);

    double diff = 1000000 * ktimer.Elapsed();
    printf("kernel <<<downsize>>>：the time is %.2lf us ( %.2f Mops)\n",
           (double) diff, (double) (old_size) / diff);

    checkCudaErrors(cudaGetLastError());
}

//
//void gpu_rehash(TTT old_size,TTT new_table_size){
//    //malloc
//    printf("----rehash size:  %d --> %d\n",old_size,new_table_size);
//    TTT* d_key,*d_value;
//    cudaMalloc((void**)&d_key, sizeof(TTT)*new_table_size);
//    cudaMalloc((void**)&d_value, sizeof(TTT)*new_table_size);
//    cudaMemset(d_key,0, sizeof(TTT)*new_table_size);
//    cudaMemset(d_value,0, sizeof(TTT)*new_table_size);
//
//    checkCudaErrors(cudaGetLastError());
//    rehash<<<NUM_BLOCK,NUM_THREADS>>>(d_key,d_value,old_size,new_table_size);
//    
//checkCudaErrors(cudaGetLastError());
//
//}



/// show table by key,value
__global__ void show_table() {
    if (blockIdx.x * blockDim.x + threadIdx.x > 0) return;
    /// i is the table num
    for (int i = 0; i < TABLE_NUM; i++) {
        printf("\n\n\ntable:%d\n", i);
        /// j is the bucket num
        for (int j = 0; j < get_table_length(i); j++) {
            printf("bucket:%d\n", j);
            /// t is every slot(one bucket has 32 slot)
            for (int t = 0; t < BUCKET_SIZE; t++) {
                if (t % 8 == 0) printf("\n\t\t");
                printf(" %d,%d ", table.table[i][j].key[t], table.table[i][j].value[t]);
            }
            printf("\n");
        }
    }
}

void GPU_show_table(){
    show_table<<<1,1>>>();
}

void gpu_lp_insert(TTT* key,TTT* value,TTT size,TTT* resize,cudaStream_t stream) {

    //in main
    // st is you operator num
    dim3 block=choose_block_num(size);

    //printf("start gpulpi\n");
    //checkCudaErrors(cudaGetLastError());

    //printf("confige:b:%d t:%d",block,NUM_THREADS);

    if(stream==0){
        cuckoo_insert <<< block, NUM_THREADS >>> (key, value, size, resize);
    }else{
        cuckoo_insert <<< block, NUM_THREADS , 0, stream>>> (key, value, size, resize);
    }

 
    int *a = new int[1];

    //checkCudaErrors(cudaGetLastError());
    cudaMemcpyAsync(a, resize, sizeof(TTT), cudaMemcpyDeviceToHost,stream);
    checkCudaErrors(cudaGetLastError());
}


void gpu_lp_search(TTT* key,TTT* ans,TTT size,cudaStream_t stream){


    dim3 block=choose_block_num(size);

    GpuTimer ktimer;

    ktimer.Start(stream);

    if(stream==0) {
        //checkCudaErrors(cudaGetLastError());
        cuckoo_search <<< block, NUM_THREADS >>> (key, ans, size);
    }else{
        cuckoo_search <<< block, NUM_THREADS ,0,stream >>> (key, ans, size);
    }
    ktimer.Stop(stream);



    double diff = 1000000 * ktimer.Elapsed();
    printf("kernel <<<search>>>：the time is %.2lf us, ( %.2f Mops)s\n",
        (double)diff, (double)(size) / diff);
//    cudaDeviceSynchronize();

//    checkCudaErrors(cudaGetLastError());

}

/// only used in api/init
void gpu_lp_set_table(cuckoo *h_table) {
    //printf("seting table\n");
    cudaMemcpyToSymbol(table,h_table,sizeof(cuckoo));
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

int choose_block_num(TTT size){
    unsigned int real_block=(size+NUM_THREADS-1)/NUM_THREADS;
    /// 不能超过 NUM_BLOCK
    int block=real_block>NUM_BLOCK ? NUM_BLOCK : real_block;
    ///
    block=block<1?1:block;
    return block;
}




