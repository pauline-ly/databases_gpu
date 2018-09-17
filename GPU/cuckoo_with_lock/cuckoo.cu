//
// Created by jing on 2018/7/1.
//

// add vscode sup



#include "cuckoo.h"
#include <assert.h>
#include <device_launch_parameters.h>
#include "api.h"

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


#define parameter_of_hash_function_a(num) (table.hash_fun[num].x)
#define parameter_of_hash_function_b(num) (table.hash_fun[num].y)


/// hash functiong
__device__ __forceinline__ TYPE
get_next_loc(TYPE k,
             TYPE num_table)
{
    return ( k^ parameter_of_hash_function_a(num_table)
               + parameter_of_hash_function_b(num_table)
           ) % PRIME_uint
           % get_table_length(num_table);
}

/// for debug
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
cuckoo_insert(TYPE* key, /// key to insert
              TYPE* value, /// value to insert
              TYPE size, /// insert size
              int* resize) /// insert error?
{

//    insertok=0;
//    inserterror=0;
    *resize = 0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /// for every k

#if head_info_debug
    if(tid==0) {
    printf(">>>insert kernel:\n>>>size:%d  \n", size);
    printf(">>>s_size:t1:%d, t2:%d, t3:%d, t4:%d\n",
            table.Lsize[0], table.Lsize[1], table.Lsize[2], table.Lsize[3]);
    }
#endif

    /// warp cooperation
    int lan_id = threadIdx.x & 0x0000001f;
    int warp_num_in_block = threadIdx.x >> 5;
    volatile __shared__ int warp[(THREAD_NUM) >> 5];

    /// 0 work_over,1 work to be done ,2 work error
    /// this can bu used to accelerate resize by long chain
    int is_active=0;

    /// work kv
    TYPE work_k, work_v;

    ///  leader thread num
    int lead_thread_num;

    /// keep kv every thread
    TYPE myk, myv;

    /// add when evict ,set to 0 when exist or null
    TYPE evict_time_of_one_thread = 0;

    /// for insert
    int hash;
    TYPE operator_hash_table_num = 0;

    /// using for ballot & CAS
    int tmp;

    /// ((size+31)>>5)<<5 :keep  a warp to active
    while ( tid < (((size + 31) >> 5) << 5) ) {

#if MAX_ITERATOR_over_to_break_insert
        if(*resize==1)
            break;
#endif
        evict_time_of_one_thread = 0;


        /// read data to insert
        if (tid < size) {
            is_active = 1;/// mark for work
            myk = key[tid];
            myv = value[tid];
        }
        /// if key==0 , not need insert
        is_active= (myk==0) ? 0:is_active;


        /// while have work to do
        while (__any(is_active == 1)) {
            /// reisze break line:122 can be here , some kv may disappear

            work_k = myk;
            work_v = myv;

/// step1   start voting ==================================
            /// if there is one active thread  , work
            if (is_active == 1 ){//&& warp[warp_num_in_block] != lan_id) {
                warp[warp_num_in_block] = lan_id;
            }
            lead_thread_num = warp[warp_num_in_block];


/// step2   broadcast ====================================
            work_k = __shfl(work_k, lead_thread_num);
            work_v = __shfl(work_v, lead_thread_num);
#if insert_debug
            if(lan_id==0 && work_k==debug_num)
                printf("2-kv: %d ,%d  \n",work_k,work_v);
#endif


/// step3   insert to the table. ===========================
            operator_hash_table_num++;
            operator_hash_table_num %= TABLE_NUM;
            hash = get_next_loc(work_k,  operator_hash_table_num);


/// step3.1 lock & un compress  TODO: compress
            ///  lock  ,otherwise revote
            if (lan_id == lead_thread_num) {
                /// TODO: different length need to sum ,tmp using double length
                /// tmp 0: free 1: busy
                tmp = atomicCAS(&(table.Lock[Lock_pos(operator_hash_table_num, hash)]), 0, 1);
            }//end if
            tmp = __shfl(tmp, lead_thread_num);
            if (tmp == 1){
                // revoting
                continue;
            }

#if insert_debug
            if(lan_id==0 && work_k==debug_num)
                printf("cas-kv: %d ,%d ,cas:%d \n",work_k,work_v,tmp);
#endif
            /// block
            bucket *b = &(table.table[operator_hash_table_num][hash]);



/// step3.2     check exist & insert
            tmp = __ballot(b->key[lan_id] == work_k);

            if (tmp != 0) { /// update
                if (lan_id == lead_thread_num) {
                    /// update value
                    //printf("update value,%d %d->%d ,tmpk %d\n",myk,b->value[__ffs(tmp) - 1],myv,b->key[__ffs(tmp) - 1]);
                    b->value[__ffs(tmp) - 1] = myv;
                    is_active = 0;
                    evict_time_of_one_thread = 0;
                }// end if ,upadte

                /// TODO: the lock free: one thread / all thread
                table.Lock[Lock_pos(operator_hash_table_num, hash)] = 0;
                continue;
            }//end check update



/// step3.3      check null & insert
            tmp = __ballot(b->key[lan_id] == 0);
#if insert_debug
            if(lan_id==0 && work_k==debug_num) {
                printf("start-null-kv: %d ,%d ,ballot:%x \n", work_k, work_v, tmp);
                pbucket(b, operator_hash_table_num, hash, get_table_length(operator_hash_table_num));
            }
#endif
            if (tmp != 0) {
                /// set kv
                if (lan_id == __ffs(tmp) - 1) {
                    b->key[lan_id] = work_k;
                    b->value[lan_id] = work_v;
                }// insert
                /// mark active false
                if (lan_id == lead_thread_num) {
                    evict_time_of_one_thread = 0;
                    is_active = 0;
                    table.Lock[Lock_pos(operator_hash_table_num, hash)] = 0;
                }
                /// insert ok ,
                continue;
            }/// null insert over

#if insert_debug
            if(lan_id==0 && work_k==debug_num)
                printf("evict-kv: %d ,%d ,ballot:%x \n",work_k,work_v,tmp);
#endif

/// step3.4     other,we need  cuckoo evict
            if (lan_id == lead_thread_num) {
                /// choose pos:lan_id evict ,TODO: choose rand?
                myk = b->key[lan_id];
                myv = b->value[lan_id];
                b->key[lan_id] = work_k;
                b->value[lan_id] = work_v;
                evict_time_of_one_thread++;

                /// when one always get leader , mark rehash
                /// check long chain
                if (evict_time_of_one_thread >= MAX_ITERATOR) {
#if record_cannot_insert_num
                    atomicAdd(resize,1);
                    printf(">>>evict_time_of_one_thread,need resize:%d,tid:%d\n",*resize,tid);
#else
                    *resize=1;
                    printf("need resize\n");
#endif
                    evict_time_of_one_thread = 0;
                    is_active = 2;
                }
            } // evict
            table.Lock[Lock_pos(operator_hash_table_num, hash)] = 0;

        }//while any owrk

#if MAX_ITERATOR_over_to_break_insert
        key[tid] = (is_active==2) ? myk : 0;
        key[tid] = (is_active==2) ? myv : 0;
#endif
        tid += BLOCK_NUM * THREAD_NUM;
    }//while size
}//cucukoo insert


__global__ void
cuckoo_search(TYPE* key, /// key to s
              TYPE* value, /// value to key
              TYPE size) /// s size
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /// for every k
#if head_info_debug
    if(tid==0) {
    printf(">>>search kernel:\n>>>size:%d  \n", size);
    printf(">>>s_size:t1:%d, t2:%d, t3:%d, t4:%d\n",
            table.Lsize[0], table.Lsize[1], table.Lsize[2], table.Lsize[3]);
    }
#endif

    int lan_id = threadIdx.x & 0x0000001f;
    int warp_num_in_block = threadIdx.x >> 5;
    volatile  __shared__ int warp[( THREAD_NUM)>>5 ];

    TYPE myk;
    TYPE myv;

    int is_active;
    TYPE work_k = 0;

    /// for search
    int hash;
    int operator_hash_table_num;
    int ballot;
    bucket *b;



    /// ((size+31)>>5)<<5 :keep  a warp to active
    while ( tid < (((size + 31) >> 5) << 5) ) {

        if(tid<size) {
            myk = key[tid];
            is_active = 1;/// mark for work
        }
        is_active = (myk==0) ? 0 : is_active;

        /// while have work to do
        while (__any(is_active != 0)) {

            operator_hash_table_num=0;


/// step1   start voting ==================================
            if (is_active != 0)
                warp[warp_num_in_block] = lan_id;
#if search_debug
            if(lan_id==0)
                printf("voting: %d\t",warp[warp_num_in_block] );
#endif
            work_k = myk;

/// step2   broadcast ====================================
            work_k=__shfl(work_k, warp[warp_num_in_block]);


/// step3   find in 5 table ===========================

            /// find null or too long
            for (int i = 0; i < TABLE_NUM; i++) {
                operator_hash_table_num = i;
                hash = get_next_loc(work_k, operator_hash_table_num);
                b=&table.table[operator_hash_table_num][hash];

                ballot=__ballot(b->key[lan_id]==work_k);

                /// find it
                if(ballot!=0){
                    if(lan_id==warp[warp_num_in_block]){
                        myv=b->value[__ffs(ballot)-1];
#if search_debug
                        printf("find %d: %d\n",key[tid],value[tid]);
#endif
                        is_active=0;
                    }
                    break;
                }

            }/// end for


            /// can not find
            if(lan_id==warp[warp_num_in_block]){
                if(is_active==1) myv=2;
                    //printf("cannot find k: %d  ,tid:%d ",myk,tid);
                //pbucket(b,operator_hash_table_num,hash,get_table_length(operator_hash_table_num));
                is_active=0;
            }
        }

        value[tid]=myv;
        tid += BLOCK_NUM * THREAD_NUM;
    }

}//cuckoo_search


/// del and return value
__global__ void
cuckoo_delete(TYPE* key, /// key to del
              TYPE* value, /// value to return
              TYPE size) /// size
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /// for every k
#if head_info_debug
    if(tid==0) {
    printf(">>>delete kernel:\n>>>size:%d  \n", size);
    printf(">>>s_size:t1:%d, t2:%d, t3:%d, t4:%d\n",
            table.Lsize[0], table.Lsize[1], table.Lsize[2], table.Lsize[3]);
    }
#endif

    int lan_id = threadIdx.x & 0x0000001f;
    int warp_num_in_block = threadIdx.x >> 5;
    volatile  __shared__ int warp[( THREAD_NUM)>>5 ];

    TYPE myk;
    TYPE myv;

    int is_active;
    TYPE work_k = 0;

    /// for search
    int hash;
    int operator_hash_table_num;
    int ballot;
    bucket *b;


    /// ((size+31)>>5)<<5 :keep  a warp to active
    while ( tid < (((size + 31) >> 5) << 5) ) {

        if(tid<size) {
            myk = key[tid];
            is_active = 1;/// mark for work
        }
        is_active = (myk==0)? 0 : is_active;

        /// while have work to do
        while (__any(is_active != 0)) {

            operator_hash_table_num=0;

/// step1   start voting ==================================
            if (is_active != 0)
                warp[warp_num_in_block] = lan_id;
#if search_debug
            if(lan_id==0)
                printf("voting: %d\t",warp[warp_num_in_block] );
#endif
            work_k = myk;

/// step2   broadcast ====================================
            work_k=__shfl(work_k, warp[warp_num_in_block]);


/// step3   find in 5 table ===========================

            /// find null or too long
            for (int i = 0; i < TABLE_NUM; i++) {
                operator_hash_table_num = i;
                hash = get_next_loc(work_k, operator_hash_table_num);
                b=&table.table[operator_hash_table_num][hash];

                ballot=__ballot(b->key[lan_id]==work_k);

                /// find it
                if(ballot!=0){
                    if(lan_id==warp[warp_num_in_block]){
                        myv=b->value[__ffs(ballot)-1];
#if search_debug
                        printf("find %d: %d\n",key[tid],value[tid]);
#endif
///step3.1   if find, set to zero ===========================
                        b->key[__ffs(ballot)-1]=0;
                        is_active=0;
                    }
                    break;
                }

            }/// end for

            /// can not find
            if(lan_id==warp[warp_num_in_block]){
                    is_active=0;
            }
        }
        value[tid]=myv;


        tid += BLOCK_NUM * THREAD_NUM;
    }
}//cuckoo_delete



void __global__
cuckoo_resize_up(bucket* old_table, /// new table has been set to table
                 int old_size,
                 TYPE num_table_to_resize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    int warp_num_in_all = tid >> 5;
    int lan_id = tid & 0x1f;


    /// take kv to insert
    TYPE key, value;

    int hash;


    /// hold old one bucket to op
    bucket *b;

///step1 取新表  ======================
    bucket *new_table = table.table[num_table_to_resize];

///step2 每个warp处理一个bucket ======================
    old_size /= BUCKET_SIZE;
    while (warp_num_in_all < old_size) {

///step2.1  获取自己的bucket ======================
        b = &old_table[warp_num_in_all];

///step2.2 对bucket中各插入对应的位置======================
        key = b->key[lan_id];
        value = b->value[lan_id];
        if (key != 0) {
            /// how to use tid & hash fun
            hash = get_next_loc(key,  num_table_to_resize);
            new_table[hash].key[lan_id] = key;
            new_table[hash].value[lan_id] = value;
        }

        tid += BLOCK_NUM * THREAD_NUM;
        warp_num_in_all = tid >> 5;
    }

}//cuckoo_resize_up

void __global__
cuckoo_resize_down(bucket* old_table,  /// small
                   int old_size,
                   int num_table_to_resize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

#if head_info_debug
    if(tid==0) {
        printf(">>>down_size kernel: %d->%d\n",old_size,table.Lsize[num_table_to_resize]);
        printf(">>>s_size:t1:%d, t2:%d, t3:%d, t4:%d\n",
               table.Lsize[0], table.Lsize[1], table.Lsize[2], table.Lsize[3]);
    }
#endif

    bucket *b=NULL;
    bucket *des_b=NULL;

    /// take kv to insert
    TYPE key, value;

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
    volatile __shared__ int warp[(THREAD_NUM) >> 5];

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
        /// 空kv再此读入第二个bucket 交叉读取
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
#if  down_size_debug || down_size_cas_insert_debug
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
#if down_size_debug || down_size_cas_insert_debug
        if(warp_num_in_block==0)
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
#if  down_size_debug || down_size_cas_insert_debug
        if(tid==0)
            printf("after2.5 start3\n");
        if(tid==0)
            pbucket(des_b,num_table_to_resize,tid,get_table_length(num_table_to_resize));
#endif
        /// key value has kv to insert
        TYPE work_k,work_v;
        int operator_hash_table_num=0;
        int lead_thread_num;
#if down_size_cas_insert_debug
        if(warp_num_in_all==0) {
            printf("b1-%d: %d,%d - %d\n",lan_id,key,value,is_active);
        }
#endif

        while (__any(is_active != 0)) {
            /// using logic of  cuckoo_insert  (__global__)
            /// how to reuse the code ?

            /// TODO , check too long evict

            work_k = key;
            work_v = value;

/// step3.1 start voting ==================================
            if (is_active != 0)//&& warp[warp_num_in_block]!=lan_id )
                warp[warp_num_in_block] = lan_id;


            /// leader is lead_thread_num
            lead_thread_num = warp[warp_num_in_block];


/// step3.2   broadcast ====================================
            work_k = __shfl(work_k, lead_thread_num);
            work_v = __shfl(work_v, lead_thread_num);

/// step3.3   insert to the table. ===========================
            operator_hash_table_num ++;
            /// donot insert to table:num_table_to_resize full
            if (operator_hash_table_num==num_table_to_resize ) {
                operator_hash_table_num++;
            }
            operator_hash_table_num %= TABLE_NUM;
            hash = get_next_loc(work_k,  operator_hash_table_num);

/// step3.4   lock   TODO: compress  ===========================
            /// using ballot as tmp to decrease register
            ///  lock  ,otherwise revote
            if (lan_id == lead_thread_num) {
                /// TODO: different length need to sum ,tmp using double length
                ballot = atomicCAS(&(table.Lock[Lock_pos(operator_hash_table_num,hash)]), 0, 1);
            }//end if
            ballot = __shfl(ballot, lead_thread_num);
            if (ballot == 1)
                continue;

            b = &(table.table[operator_hash_table_num][hash]);

#if down_size_cas_insert_debug
            ballot=__ballot(is_active==1);
            if(warp_num_in_block==0 && lan_id==0){
                printf("\n\nactive ballot:%x kv %d,%d lead_thread_num:%d\n",
                       ballot,work_k,work_v,lead_thread_num );
                pbucket(b,operator_hash_table_num,hash,get_table_length(operator_hash_table_num));
            }
#endif
/// step3.5   check exist & insert
            ballot = __ballot(b->key[lan_id] == work_k);
            if (ballot != 0) { /// update

                if (lan_id == lead_thread_num) {
                    b->value[__ffs(ballot) - 1] = value;
                    is_active = 0;
#if down_size_cas_insert_debug
                    if(warp_num_in_block==0) {
                        printf("exit after insert \n");
                        pbucket(b, operator_hash_table_num, hash, get_table_length(operator_hash_table_num));
                    }
#endif
                    table.Lock[Lock_pos(operator_hash_table_num,hash)] = 0;
                }// end if ,upadte

                continue;
            }//end check update

/// step3.6   check null & insert
            ballot = __ballot(b->key[lan_id] == 0);

#if down_size_cas_insert_debug
            if(warp_num_in_block==0)  printf("%d,",lan_id);
            if(tid==0){
                printf("\n\nnull ballot:%x kv %d,%d lead_thread_num:%d \n",
                       ballot,work_k,work_v,lead_thread_num);
            }
#endif
            if (ballot != 0) {
                /// set kv
                if (lan_id == __ffs(ballot) - 1) {
                    b->key[lan_id] = work_k;
                    b->value[lan_id] = work_v;
                    /// free
                    table.Lock[Lock_pos(operator_hash_table_num,hash)] = 0;
#if down_size_cas_insert_debug
                    if(warp_num_in_block==0) {
                        printf("null after insert \n");
                        pbucket(b, operator_hash_table_num, hash, get_table_length(operator_hash_table_num));
                    }
#endif
                }// insert

                /// mark active false
                if (lan_id == lead_thread_num)
                    is_active = 0;
                continue;
            }/// null insert over


/// step3.7     other,we need  cuckoo evict
            if (lan_id == lead_thread_num){
                key = b->key[lan_id];
                value = b->value[lan_id];
                b->key[lan_id] = work_k;
                b->value[lan_id] = work_v;
#if down_size_cas_insert_debug
                if(warp_num_in_block==0) {
                    printf("evict after insert \n");
                    pbucket(b, operator_hash_table_num, hash, get_table_length(operator_hash_table_num));
                }
#endif
                table.Lock[Lock_pos(operator_hash_table_num,hash)] = 0;
            } // evict
        }

        /// TODO:auto configure ,what should be add to tid
        tid += BLOCK_NUM * THREAD_NUM;
        warp_num_in_all = tid >> 5;
    }

}//cuckoo_resize_down

int choose_block_num(TYPE size);

void GPU_cuckoo_resize_up(int num_table_to_resize,
                          TYPE old_size,
                          bucket* new_table,
                          cuckoo *h_table)
{

    checkCudaErrors(cudaGetLastError());
    TYPE new_size=old_size*2;

    ///  set table & size it needed
    bucket* old_table=h_table->table[num_table_to_resize];
    h_table->Lsize[num_table_to_resize]=new_size;
    h_table->table[num_table_to_resize]=new_table;
    cudaMemcpyToSymbol(table,h_table,sizeof(cuckoo));

    /// TODO: auto configure
    /// kernel Configuration

    dim3 block=choose_block_num(old_size);

    /// kernel launch


    GpuTimer timer;
    timer.Start();

    cuckoo_resize_up<<<block,THREAD_NUM>>>(old_table,old_size,num_table_to_resize);

    timer.Stop();
    double diff = timer.Elapsed()*1000000;
    printf("kernel <<<rehash>>>：the time is %.2lf us, ( %.2f Mops)s\n",
           (double)diff, (double)(new_size) / diff);

}//GPU_cuckoo_resize_up

void GPU_cuckoo_resize_down(int num_table_to_resize,
                            TYPE old_size,
                            bucket* new_table,
                            cuckoo *h_table)
{
    /// bucket to size : << 5
    int new_size=((get_table_bucket_size(num_table_to_resize)+1)/2) << 5;
    //printf("down_size : %d : szie%d->%d.",num_table_to_resize,old_size,new_size);

    ///  set table & size it needed
    bucket* old_table=h_table->table[num_table_to_resize];
    h_table->Lsize[num_table_to_resize]=new_size;
    h_table->table[num_table_to_resize]=new_table;
    cudaMemcpyToSymbol(table,h_table,sizeof(cuckoo));

    dim3 block=choose_block_num(old_size);

    /// kernel launch
    cuckoo_resize_down<<<block,THREAD_NUM>>>(old_table,old_size,num_table_to_resize);


}//GPU_cuckoo_resize_down





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
                ///  8 slot a line
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


void gpu_lp_insert(TYPE* key,
                   TYPE* value,
                   TYPE size,
                   int* resize)
{
    dim3 block=choose_block_num(size);

    GpuTimer time;
    time.Start();

    cuckoo_insert <<< block, THREAD_NUM >>> (key, value, size, resize);

    time.Stop();
    double diff = time.Elapsed() * 1000000;
    printf("kernel <<<insert>>>：the time is %.2lf us ( %.2f Mops)\n",
           (double) diff, (double) (size) / diff);
}//gpu_lp_insert



void gpu_lp_search(TYPE* key,
                    TYPE* ans,
                    TYPE size){
    dim3 block=choose_block_num(size);

    GpuTimer time;
    time.Start();

    cuckoo_search<<<block,THREAD_NUM>>>(key,ans,size);

    time.Stop();
    double diff = time.Elapsed() * 1000000;
    printf("kernel <<<search>>>：the time is %.2lf us, ( %.2f Mops)s\n",
           (double)diff, (double)(size) / diff);
    //    checkCudaErrors(cudaGetLastError());
}

void gpu_lp_delete(TYPE* key,
                   TYPE* ans,
                   TYPE size){
    dim3 block=choose_block_num(size);

    GpuTimer time;
    time.Start();

    cuckoo_delete<<<block,THREAD_NUM>>>(key,ans,size);

    time.Stop();
    double diff = time.Elapsed() * 1000000;
    printf("delete <<<delete>>>：the time is %.2lf us, ( %.2f Mops)s\n",
           (double)diff, (double)(size) / diff);
    //    checkCudaErrors(cudaGetLastError());
}

void gpu_lp_set_table(cuckoo *h_table) {
    //printf("seting table\n");
    cudaMemcpyToSymbol(table,h_table,sizeof(cuckoo));
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

int choose_block_num(TYPE size){
    unsigned int real_block=(size+THREAD_NUM-1)/THREAD_NUM;
    /// 不能超过 BLOCK_NUM
    int block=real_block>BLOCK_NUM ? BLOCK_NUM : real_block;
    ///
    block=block<1?1:block;
    return block;
}




