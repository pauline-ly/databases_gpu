#include <assert.h>
//#include <device_launch_parameters.h>
#include <curand_kernel.h> // cu rand
#include "api.h"
#include "cuckoo.h"

#define gtid (blockIdx.x * blockDim.x + threadIdx.x)

#define bucketk(b, i) ((uint32_t)((b->sigloc[i])>>32))
#define bucketv(b, i) ((uint32_t)((b->sigloc[i])&0xffffffff))
#define bucket_all(b, i) (b->sigloc[i])
#define bitmove (threadIdx.x & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P))


/// hash table
__constant__ cuckoo table;
#define  get_table_length(i)  get_table_bucket_length(i)
#define  get_table_bucket_length(i) (table.Lsize[i]>>BUCKET_SIZE_P)
/// Lsize0 is the biggest /// resize it need change
#define  Lock_pos(num, hash) ((num) * (get_table_length(0)) + hash)

/// hash para
#define parameter_of_hash_function_a(num) (table.hash_fun[num].x)
#define parameter_of_hash_function_b(num) (table.hash_fun[num].y)


__device__ __forceinline__ uint32_t
get_hash1(uint32_t key) {
    key = ~key + (key << 15); // key = (key << 15) - key - 1;
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057; // key = (key + (key << 3)) + (key << 11);
    key = key ^ (key >> 16);
    return (key);
}


__device__ __forceinline__ uint32_t
get_hash2(uint32_t a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__device__ __forceinline__ uint32_t
get_hash3(uint32_t sig) {
    return ((sig ^ 59064253) + 72355969) % PRIME_uint;
}


__device__ __forceinline__ uint32_t
get_hash4(uint32_t a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

/// hash functiong
// __device__ __forceinline__ TYPE
// hash_function(TYPE k,
//               TYPE num_table)
// {
//     return (k ^ parameter_of_hash_function_a(num_table)
//                 + parameter_of_hash_function_b(num_table)
//            ) % PRIME_uint
//            % get_table_length(num_table);
// }
__device__ __forceinline__ TYPE
hash_function(TYPE k,
              TYPE num_table)
{
    switch(num_table%4)
    {
        case 0:return get_hash1(k) % PRIME_uint  % get_table_length(num_table);
            
        case 1:return get_hash2(k) % PRIME_uint  % get_table_length(num_table);
    
        case 2:return get_hash3(k) % PRIME_uint  % get_table_length(num_table);
    
        case 3:return get_hash4(k) % PRIME_uint  % get_table_length(num_table);
    }
}



//__device__ __forceinline__ uint32_t
//hash_function(uint32_t sig, int num) {
////    fill 1.05 25->20
//    switch (num & 0x3) {
//        case 0:
//            return get_hash1(sig) % get_table_length(0);
//        case 1:
//            return get_hash2(sig % get_table_length(1));
//        case 2:
//            return get_hash3(sig) % get_table_length(2);
//        case 3:
//            return get_hash4(sig) % get_table_length(3);
//    }
//}


__device__ __forceinline__ uint32_t
get_hash5(uint32_t a) {
    a -= (a << 6);
    a ^= (a >> 17);
    a -= (a << 9);
    a ^= (a << 4);
    a -= (a << 3);
    a ^= (a << 10);
    a ^= (a >> 15);
    return a;
}

#if n_choose_2

// //n=2
// // 其实无需选择
// __device__ __forceinline__ uint32_t
// choose_hash(uint32_t sig) {
//     return sig&1;
// }

// __device__ __forceinline__ uint32_t
// pos1(uint32_t pos) {
//     return pos;
// }

// __device__ __forceinline__ uint32_t
// pos2(uint32_t pos) {
//     return pos^1;
// }
// 
// //n=3  if 位移版本
// __device__ __forceinline__ uint32_t
// choose_hash(uint32_t sig) {
//     return (get_hash5(sig) % 3);
// }

// __device__ __forceinline__ uint32_t
// pos1(uint32_t pos) {
//     if (pos & 1)  // xx1 12
//         return 1;
//     return 0;
// }

// __device__ __forceinline__ uint32_t
// pos2(uint32_t pos) {
//     if (pos & 1)  // xx1 01 02 03: 001 011 101 :00 01 10 :+1  1 2 3
//         return 2;
//     return (pos >> 1) + 1;  // 01 02
// }

// //n=3 switch case 版本
// __device__ __forceinline__ uint32_t
// choose_hash(uint32_t sig) {
//     return (get_hash5(sig) % 3);
// }

// __device__ __forceinline__ uint32_t
// pos1(uint32_t pos) {
//     switch (pos){
//         case 0:
//         case 1:// 01 02 
//             return 0;
//         case 2:    // 12
//             return 1;
//     }
// }

// __device__ __forceinline__ uint32_t
// pos2(uint32_t pos) {
//     switch (pos){
//         case 0:
//         case 1:
//             return pos+1;
//         case 2:
//             return 2;
//     }
// }

// n=4
__device__ __forceinline__ uint32_t
choose_hash(uint32_t sig) {
    return (get_hash5(sig) % 6);
}


// xx1 01 02 03 x00 12 13 x10 23
__device__ __forceinline__ uint32_t
pos1(uint32_t pos) {
    if (pos & 1)  // xx1 01 02 03
        return 0;
    if (pos & 2)  // 23
        return 2;
    return 1;  // 12 13
}

__device__ __forceinline__ uint32_t
pos2(uint32_t pos) {
    if (pos & 1)  // xx1 01 02 03: 001 011 101 :00 01 10 :+1  1 2 3
        return (pos >> 1) + 1;
    if (pos & 2)  // 23
        return 3;
    return (pos >> 2) + 2;  // 12 13
}


// // n=5
// __device__ __forceinline__ uint32_t
// choose_hash(uint32_t sig) {
//     return (get_hash5(sig) % 10);
// }


// __device__ __forceinline__ uint32_t
// pos1(uint32_t pos) {
// 	if(pos == 9) 
// 		return 3;
//     if (pos & 1)  // xx1 01 02 03 04
//         return 0;
//     if (pos & 2)  // 23 24
//         return 2;
//     return 1;  // 12 13 14
// }

// __device__ __forceinline__ uint32_t
// pos2(uint32_t pos) {
// 	if(pos == 9) 
// 		return 4;
//     if (pos & 1)  // xx1 01 02 03 04
//         return (pos >> 1) + 1;
//     if (pos & 2)  // 23 24
//         return (pos >> 2) + 3;
//     return (pos >> 2) + 2;  // 12 13 14
// }



// // n=6
// __device__ __forceinline__ uint32_t
// choose_hash(uint32_t sig) {
//     return (get_hash5(sig) % 15);
// }

// __device__ __forceinline__ uint32_t
// pos1(uint32_t pos) {
// 	switch (pos){
// 		case 0:
//         case 1:
//         case 2:
//         case 3:
//         case 4:
// 			return 0;
// 		case 5:
//         case 6:
//         case 7:
//         case 8:
// 			return 1;
// 		case 9:
//         case 10:
//         case 11:
// 			return 2;
// 		case 12:
//         case 13:
// 			return 3;
// 		case 14:
// 			return 4;
// 	}
// }

// __device__ __forceinline__ uint32_t
// pos2(uint32_t pos) {
// 	switch (pos){
// 		case 0:
//         case 1:
//         case 2:
//         case 3:
//         case 4:
// 			return pos+1;
// 		case 5:
//         case 6:
//         case 7:
//         case 8:
// 			return pos-3;
// 		case 9:
//         case 10:
//         case 11:
// 			return pos-6;
// 		case 12:
//         case 13:
// 			return pos-8;
// 		case 14:
// 			return 5;
// 	}
// }
//#define pos22(t) ((t)>>2)
//
//
//#define not_same_pos(pre,now,t) (( pos1(t)!=now && pos1(t)!=pre ) \
//                                ||  ( pos2(t)!=now && pos2(t)!=pre ))
//
#define get_pos(hash, pre) ((pre==pos1(hash))? \
                                pos2(hash):   \
                                pos1(hash))
#define not_same_pos(p1,p2) ((p1) != (p2))
//
//
//#define pos1(t) ((t)&3)
//#define pos2(t) ( (pos1(t)==(pos22(t)))? \
//                            ((pos1(t)+1)&3): \
//                            pos22(t) )

#else // end of n choose 2

#endif
__global__ void
cuckoo_insert(
        TYPE *keys,
        TYPE *values,
        TYPE elem_num,
        int *rehash,
        int *iterator_count)
{
    int lan_num_in_all = (blockIdx.x * blockDim.x + threadIdx.x) >> ELEM_NUM_P;
    int step = (gridDim.x * blockDim.x) >> ELEM_NUM_P;

    for (; lan_num_in_all < elem_num; lan_num_in_all += step) {
        TYPE pre_table_no = 100;
        int k = keys[lan_num_in_all];
        int v = values[lan_num_in_all];
        /// last bit == 1 , first insert to pos 2
#if n_choose_2
        if (k & 1) pre_table_no = get_pos(choose_hash(k), 100);
#else
        pre_table_no = k % TABLE_NUM;
#endif
        Entry entry = makeEntry(k, v);
        
        int count=0;
        warp_insert(entry, pre_table_no, rehash, count);
#if ITERAOTOR_COUNT_FLAG
        iterator_count[lan_num_in_all] = count;
//        if(lan_num_in_all < 1000 && threadIdx.x==0) printf("%d:%d ",lan_num_in_all,count);
#endif 
    }
    return;
}


/**
 * a warp -- ELEM_NUM thread insert one entry to the kv
 * try MAX_CUCKOO_NUM times
 * #1 cal hash, get insert to bucket b
 * #2 find  pos :find null 0 ,find new hash_table != hash , choose  k ^ evictnum
 * #3 atomic EXCH , shfl
 */
void __device__ __forceinline__
warp_insert(Entry &entry,
            TYPE pre_table_no,
            int* rehash,
            int &count)
{
#if cu_random
    curandState state;
    curand_init(gtid>>ELEM_NUM_P, 0, 0, &state);
#endif
    int evict_num;
    int simd_lane = threadIdx.x & ((1 << ELEM_NUM_P) - 1);
    for (evict_num = 0; evict_num < MAX_CUCKOO_NUM; ++evict_num) {
#if n_choose_2
        TYPE table_no_hash = get_pos(choose_hash(getk(entry)), pre_table_no);
#else
        TYPE table_no_hash = (table_no_hash + 1) % TABLE_NUM;
#endif
        uint32_t hash = hash_function(getk(entry), table_no_hash);
        /// get insert bucket_t
        bucket *b = (table.table[table_no_hash] + hash);
        int key_in_bucket = bucketk(b, simd_lane);  // key in bucket

        /// #2 find one pos
        int ballot = __ballot(key_in_bucket == 0);
        ballot = (ballot >> bitmove) & ((1 << ELEM_NUM) - 1);
#if n_choose_2
        if (ballot == 0) {
//                         //n=2
//             TYPE table_no = choose_hash(key_in_bucket);
//             ballot = __ballot(true);
//             ballot = (ballot >> bitmove) & ((1 << ELEM_NUM) - 1);
            //n=3-6
            TYPE table_no = choose_hash(key_in_bucket);
            ballot = __ballot(table_no_hash != table_no);
            ballot = (ballot >> bitmove) & ((1 << ELEM_NUM) - 1);
        }
#endif
#if cu_random
        /// #3 exch
        int chosen_simd = ballot != 0 ? (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1)
                                      : (curand(&state)) & ((1 << ELEM_NUM_P) - 1);
#else
        int chosen_simd = ballot != 0 ? (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1)
                                      : evict_num & ((1 << ELEM_NUM_P) - 1);
#endif
        Entry tmpCAS;
        if (simd_lane == chosen_simd) {
            tmpCAS = atomicExch(&bucket_all(b, simd_lane), entry);
        }
        entry = __shfl(tmpCAS, chosen_simd, ELEM_NUM);
#if ITERAOTOR_COUNT_FLAG
        count ++;
#endif
        if (entry == 0) return;
        pre_table_no = table_no_hash;
    }
    if (evict_num >= MAX_CUCKOO_NUM && simd_lane == 0) {
#if small_slot
        if (!warp_small_insert(entry)) {
#endif
            *rehash = 1;
            //            if(gtid<p_num)  printf("-----cannnot insert%d: %d,%d \n",lan_num_in_all,getk(entry),getv(entry));
#if small_slot
        }
#endif
    }
}

#if small_slot
bool __device__ __forceinline__
warp_small_insert(Entry &entry) {
    int simd_lane = threadIdx.x & ((1 << ELEM_NUM_P) - 1);

    uint32_t hash = get_hash5(getk(entry)) % SLOT_SIZE;
    bucket *b = (table.slot + hash);
    int key_in_bucket = bucketk(b, simd_lane);  // key in bucket
    int ballot =  __ballot(key_in_bucket == 0);
    ballot = (ballot >> bitmove) & ((1 << ELEM_NUM) - 1);
    if(ballot == 0) return false;

    /// insert into small table
    int chosen_simd = ballot !=  (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
    Entry tmpCAS;
    if (simd_lane == chosen_simd) {
        tmpCAS = atomicExch(&bucket_all(b, simd_lane), entry);
    }
    entry = __shfl(tmpCAS, chosen_simd, ELEM_NUM);

    return (entry == 0);
}

#endif


__global__ void
cuckoo_search(
        TYPE *keys,
        TYPE *values,
        TYPE size) {
#if head_info
    if(gtid==0){
            printf(">>>kernel del ,\n table %d  insert %d\n",bucket_num_of_single_table<<ELEM_NUM_P,size);
        }
#endif
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = (gridDim.x * blockDim.x) >> ELEM_NUM_P;
    int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
    int elem_id = idx >> ELEM_NUM_P;


    /// get insert bucket_t

    for (int id = elem_id; id < size; id += step) {
        TYPE k = keys[id];
        int finded = 0;
#if n_choose_2
        int table_no = pos1(choose_hash(k));
        int hash = hash_function(k, table_no);
        bucket_t *b = (table.table[table_no] + hash);
        if (bucketk(b, simd_lane) == k) {
            values[id] = bucketv(b, simd_lane);
            finded = 1;
        };
        if (((__ballot(finded == 1) >> bitmove) & ((1 << ELEM_NUM) - 1)) != 0) continue;

        table_no = pos2(choose_hash(k));
        hash = hash_function(k, table_no);
        b = (table.table[table_no] + hash);
        if (bucketk(b, simd_lane) == k) {
            values[id] = bucketv(b, simd_lane);
            finded = 1;
        }
#else
        for(int i=0;i<TABLE_NUM;i++){
            int hash = hash_function(k, i);
            bucket_t *b = (table.table[i] + hash);
            if (bucketk(b, simd_lane) == k) {
                values[id] = bucketv(b, simd_lane);
                finded = 1;
            };
            if (((__ballot(finded == 1) >> bitmove) & ((1 << ELEM_NUM) - 1)) != 0) break;
        }
#endif
#if small_slot
        if (((__ballot(finded == 1) >> bitmove) & ((1 << ELEM_NUM) - 1)) != 0) continue;

        hash = get_hash5(k) % SLOT_SIZE;
        b = (table.slot + hash);
        if (bucketk(b, simd_lane) == k) {
            values[id] = bucketv(b, simd_lane);
        }
#endif
    }
    return ;
}


/// check if exist delete key else ignore it
__global__ void
cuckoo_delete(
        TYPE *keys,
        TYPE *ans,  /// tmp not used (TODO) we can return the value when delete key value
        TYPE total_elem_num) {
#if head_info
    if(gtid==0){
            printf(">>>kernel del ,\n table %d  insert %d\n",bucket_num_of_single_table<<ELEM_NUM_P,total_elem_num);
        }
#endif
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int id;
/// threads to cooperate for one element
    int step = (gridDim.x * blockDim.x) >> ELEM_NUM_P;
    int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
    int elem_id = idx >> ELEM_NUM_P;

    for (id = elem_id; id < total_elem_num; id += step) {
        TYPE k = keys[id];
        int deleted = 0;
        if (k == 0) continue;

#if n_choose_2
        int table_no = pos1(choose_hash(k));
        int hash = hash_function(k, table_no);
        bucket_t *b = (table.table[table_no] + hash);
        if (bucketk(b, simd_lane) == k) {
            bucket_all(b, simd_lane) = makeEntry(0, 0);
            deleted = 1;
        }
        if (((__ballot(deleted == 1) >> bitmove) & ((1 << ELEM_NUM) - 1)) != 0) continue;


        table_no = pos2(choose_hash(k));
        hash = hash_function(k, table_no);
        b = (table.table[table_no] + hash);
        if (bucketk(b, simd_lane) == k) {
            bucket_all(b, simd_lane) = makeEntry(0, 0);
            deleted = 1;
        }
#else
        for(int i=0;i<TABLE_NUM;i++){
            int hash = hash_function(k, i);
            bucket_t *b = (table.table[i] + hash);
            if (bucketk(b, simd_lane) == k) {
                bucket_all(b, simd_lane) = makeEntry(0, 0);
                deleted = 1;
            };
            if (((__ballot(deleted == 1) >> bitmove) & ((1 << ELEM_NUM) - 1)) != 0) break;
        }
#endif
#if small_slot
        if (((__ballot(deleted == 1) >> bitmove) & ((1 << ELEM_NUM) - 1)) != 0) continue;

        hash = get_hash5(k) % SLOT_SIZE;
        b = (table.slot + hash);
        if (bucketk(b, simd_lane) == k) {
            bucket_all(b, simd_lane) = makeEntry(0, 0);
        }
#endif

    }
    return;
}


__device__ void
show_b(bucket_t *b) {
    //if((threadIdx.x & ((1<<BUCKET_SIZE_P)-1)) && (gtid > BUCKET_SIZE)) return ;
    if ((gtid != 0)) return;
    for (int i = 0; i < BUCKET_SIZE; i++) {
        printf("%d,%d\n", getk(b->sigloc[i]), getv(b->sigloc[i]));
    }
    printf("\n");
}

__global__ void
cuckoo_resize_up(bucket_t* old_table, /// new table has been set to table
                 int old_size,
                 TYPE num_table_to_resize) {
    int group_id_in_all = gtid >> BUCKET_SIZE_P;
    int id_in_group = threadIdx.x & ((1 << BUCKET_SIZE_P) - 1);
    int step = (gridDim.x * blockDim.x) >> BUCKET_SIZE_P;

    ///step1 取新表  ======================
    bucket_t *new_table = table.table[num_table_to_resize];

    ///step2 每个warp处理一个bucket ======================
    old_size >>= BUCKET_SIZE_P;
    for (;group_id_in_all < old_size;group_id_in_all += step) {

        ///step2.1  获取自己的bucket ======================
        bucket_t *b = old_table + group_id_in_all;

        ///step2.2 对bucket中各插入对应的位置======================
        Entry entry = bucket_all(b, id_in_group);
        if (entry != 0) {
            /// how to use tid & hash fun
            int hash = hash_function(getk(entry), num_table_to_resize);
            new_table[hash].sigloc[id_in_group] = entry;
        }
    }

}//cuckoo_resize_up

/**
 * down size kernel
 * pre new table is settted to __constant__ table
 * bucket_t old_table is old , and cannot changed -- other cannot ? (FIXME)
 *
 * #1 read pos1 , read over
 * #2 read pos2 , may not over , need re read
 * #3 warp voter to insert , call warp_inert_api
 */
__global__ void
cuckoo_resize_down(bucket_t *old_table,  /// small
                   int old_size,
                   int num_table_to_resize) {
    /// warp coopration
    int step = (gridDim.x * blockDim.x) >> BUCKET_SIZE_P;
    int id_in_group = threadIdx.x & ((1 << BUCKET_SIZE_P) - 1);

    ///step1 get old table  ======================
    bucket_t *new_table = table.table[num_table_to_resize];
    int new_bucket_size = get_table_length(num_table_to_resize);

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

    int group_id_in_all = gtid >> BUCKET_SIZE_P;
    for (; group_id_in_all < new_bucket_size; group_id_in_all += step) {
        /// group_id_in_all is hash_value

        bucket_t *des_b = &new_table[group_id_in_all];
        bucket_t *b = old_table + group_id_in_all;
#if down_size_debug
        show_b(b);
#endif
        Entry entry = bucket_all(b, id_in_group);


        int crose_id_in_group = BUCKET_SIZE - id_in_group - 1;
        /// 空kv再此读入第二个bucket 交叉读取
        b = old_table + group_id_in_all + new_bucket_size;
#if down_size_debug
        show_b(b);
#endif
        if (entry == 0) {
            entry = bucket_all(b, crose_id_in_group);
        }
        ///到这里，第一个bucket全部会被读入后面接着写入，第二个部分还未读入

        ///step2.3   将不为空的kv插入新表=====================
        des_b->sigloc[id_in_group] = entry;


        int is_active = 0;
        ///step2.4  读取第二个bucket中未存入的kv and mark ======================
        if (entry != b->sigloc[crose_id_in_group]  /// 从未写入过
            && b->sigloc[crose_id_in_group] != 0)  /// 存在值
        {
            entry = bucket_all(b, crose_id_in_group);
            is_active = 1;
        }

        unsigned workq = (__ballot(is_active == 1) >> bitmove) & ((1 << BUCKET_SIZE) - 1);
//        /// step2.5 ===========
//        unsigned nul_pos = (__ballot(des_b->sigloc[id_in_group] == 0) >> bitmove) & ((1 << BUCKET_SIZE) - 1);
//        while (nul_pos != 0 && workq != 0) {
//            int leader = __ffs(workq) - 1;
//            if (id_in_group == leader) {
//                des_b->sigloc[ __ffs(nul_pos) - 1 ] =entry;
//                is_active = 0;
//            }
//            workq = (__ballot(is_active == 1) >> bitmove) & ((1 << BUCKET_SIZE) - 1);
//            nul_pos = (__ballot(des_b->sigloc[id_in_group] == 0) >> bitmove) & ((1 << BUCKET_SIZE) - 1);
//        }


        /// step3 warp insert with voter
        while (workq != 0) {
            Entry work_entry = entry;
            int leader = __ffs(workq) - 1;
            work_entry = __shfl(work_entry, leader, BUCKET_SIZE);
            int a = 0;
            warp_insert(work_entry, num_table_to_resize, &a,a);
            if (id_in_group == leader) is_active = 0;
            workq = (__ballot(is_active == 1) >> bitmove) & ((1 << BUCKET_SIZE) - 1);
        }
#if down_size_debug
        show_b(des_b);
#endif
    }

}


void GPU_cuckoo_resize_up(int num_table_to_resize,
                          TYPE old_size,
                          bucket_t *new_table,
                          cuckoo *h_table) {

    checkCudaErrors(cudaGetLastError());
    TYPE new_size = old_size * 2;

    ///  set table & size it needed
    bucket_t * old_table = h_table->table[num_table_to_resize];
    h_table->Lsize[num_table_to_resize] = new_size;
    h_table->table[num_table_to_resize] = new_table;
    cudaMemcpyToSymbol(table, h_table, sizeof(cuckoo));


    /// kernel launch
#if cu_info
    GpuTimer timer;
    timer.Start();
#endif
    cuckoo_resize_up << < BLOCK_NUM_k, THREAD_NUM_k >> > (old_table, old_size, num_table_to_resize);
#if cu_info
    timer.Stop();
    double diff = timer.Elapsed() * 1000000;
//     printf("kernel <<<rehash>>>：the time is %.2lf us, ( %.2f Mops)s\n",
//            (double) diff, (double) (new_size) / diff);
#endif
}//GPU_cuckoo_resize_up



void GPU_cuckoo_resize_down(int num_table_to_resize,
                            TYPE old_size,
                            bucket_t *new_table,
                            cuckoo *h_table) {
    /// bucket_t to size : << BUCKET_SIZE_P
    int new_size = ((get_table_bucket_size(num_table_to_resize) + 1) / 2) << BUCKET_SIZE_P;
    if( new_size * 2 != old_size) printf("!!!error , down size for init table\n"
                                         "!!!may cause error in cuckoo.cu line 611\n");

    ///  set table & size it needed
    bucket_t * old_table = h_table->table[num_table_to_resize];
    h_table->Lsize[num_table_to_resize] = new_size;
    h_table->table[num_table_to_resize] = new_table;
    cudaMemcpyToSymbol(table, h_table, sizeof(cuckoo));
#if cu_info
    GpuTimer timer;
    timer.Start();
    /// kernel launch
#endif
    cuckoo_resize_down << < BLOCK_NUM_k, THREAD_NUM_k >> > (old_table, old_size, num_table_to_resize);
#if cu_info
    timer.Stop();
    double diff = timer.Elapsed() * 1000000;
    printf("kernel <<<downsize>>>：the time is %.2lf us, ( %.2f Mops)s\n",
           (double) diff, (double) (new_size) / diff);
#endif
}//GPU_cuckoo_resize_down





/// show table by key,value
__global__ void show_table() {
    if (blockIdx.x * blockDim.x + threadIdx.x > 0) return;
    /// i is the table num
    for (int i = 0; i < TABLE_NUM; i++) {
        printf("\n\n\ntable:%d\n", i);
        /// j is the bucket_t num
        for (int j = 0; j < get_table_length(i); j++) {
            printf("bucket_t:%d\n", j);
            /// t is every slot(one bucket_t has 32 slot)
            for (int t = 0; t < BUCKET_SIZE; t++) {
                ///  8 slot a line
                if (t % 8 == 0) printf("\n\t\t");
                printf(" %d,%d ", getk(table.table[i][j].sigloc[i]),getv(table.table[i][j].sigloc[i]));
            }
            printf("\n");
        }
    }
}

void GPU_show_table(){
    show_table<<<1,1>>>();
}

__device__ int sumall=0;
// every thread one table , 
__global__ void computer_table() {
    sumall=0;
    __syncthreads();
    int sum;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx>= TABLE_NUM ) return; // only need TABLE_NUM threads 

    sum=0;
    /// j is the bucket_t num
    for (int j = 0; j < get_table_length(idx); j++) {
        /// t is every slot(one bucket_t has 32 slot)
        for (int t = 0; t < BUCKET_SIZE; t++) {
            ///  8 slot a line
            if(table.table[idx][j].sigloc[t] != 0) sum++;
        }
    }
    printf("table:%d %d/%d ",idx,sum,get_table_length(idx)*BUCKET_SIZE);
    __syncthreads();
    atomicAdd(&sumall,sum);
    __syncthreads();
    if(gtid == 0) {
        int tablesize=0;
        for(int i=0;i<TABLE_NUM;i++){
            tablesize +=  (get_table_length(i)*BUCKET_SIZE);
        }
        printf("\n%d/%d : %lf \n",sumall,tablesize,
               sumall*1.0/(tablesize));
    }
}

//
void gpu_computer_table() {
    computer_table<<<1,TABLE_NUM>>>();
    cudaDeviceSynchronize();
//     printf("\n");
}


void gpu_lp_insert(TYPE *key,
                   TYPE *value,
                   TYPE size,
                   int *resize,
                   int *iterator_count) {
#if cu_info
    GpuTimer timer;
    timer.Start();
#endif
    cuckoo_insert << < BLOCK_NUM_k, THREAD_NUM_k >> > (key, value, size, resize, iterator_count);
#if cu_info
    timer.Stop();
    double diff = timer.Elapsed() * 1000000;
    printf("<insert>%.2lf %.2f\n",
           (double) diff, (double) (size) / diff);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
#endif

}//gpu_lp_insert



void gpu_lp_search(TYPE *key,
                   TYPE *ans,
                   TYPE size) {
#if cu_info
    GpuTimer time;
    time.Start();
#endif
    cuckoo_search << < BLOCK_NUM_k, THREAD_NUM_k >> > (key, ans, size);
#if cu_info
    time.Stop();
    double diff = time.Elapsed() * 1000000;
    printf("search:<search>%.2lf %.2f\n",
           (double) diff, (double) (size) / diff);
    //    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
#endif

}

void gpu_lp_delete(TYPE *key,
                   TYPE *ans,
                   TYPE size) {
#if cu_info
    GpuTimer time;
    time.Start();
#endif
    cuckoo_delete << < BLOCK_NUM_k, THREAD_NUM_k >> > (key, ans, size);
#if cu_info
    time.Stop();
    double diff = time.Elapsed() * 1000000;
    printf("delete <<<delete>>>：the time is %.2lf us, ( %.2f Mops)s\n",
           (double) diff, (double) (size) / diff);
    checkCudaErrors(cudaGetLastError());
#endif
}

void gpu_lp_set_table(cuckoo * h_table) {
    //printf("seting table\n");
    cudaMemcpyToSymbol(table, h_table, sizeof(cuckoo));
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}





