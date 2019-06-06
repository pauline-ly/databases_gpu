
#ifndef LINEAR_LINEAR_PROBE_H
#define LINEAR_LINEAR_PROBE_H

#include "dy_hash.h"
#include "cuckoo.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdint.h>



#define Entry unsigned long long
#define makeEntry(a,b) ((((uint64_t)a)<<32)+(b))
#define getk(a) ((uint32_t)((a)>>32))
#define getv(a) ((uint32_t)((a)&0xffffffff))


#define ELEM_NUM_P		ELEM_NUM_P_gpu // 2^ELEM_NUM_P elements per bucket
#define ELEM_NUM		(1 << ELEM_NUM_P)


#define UPDATE_FLAG_ON 0 /// 0 off 1 on
#define hash_fun_num TABLE_NUM
#define MAX_CUCKOO_NUM	MAX_ITER/* maximum cuckoo evict number */



/// isolated kv  no
typedef struct bucket{
//    TYPE key[BUCKET_SIZE];
//    TYPE value[BUCKET_SIZE];
    Entry sigloc[ELEM_NUM];
}bucket_t;



/// definition structure
typedef struct mycuckoo{
    /// table , store kv ,
    /// in detail , kv is  isolated
    bucket_t *table[TABLE_NUM];
    /// size of sub table , determine rehash : L min resize
#if small_slot
    bucket_t *slot;
#endif
    TYPE Lsize[TABLE_NUM];
    /// Lock for bucket lock,use atomicCAS
    TYPE *Lock;
    uint2 hash_fun[TABLE_NUM];
}cuckoo;


int choose_block_num(TYPE size);
void __device__ __forceinline__
warp_insert(Entry& entry,
            TYPE pre_table_no,
            int* rehash,
            int &count);

#if small_slot
    bool __device__ __forceinline__
    warp_small_insert(Entry &entry);
#endif
__global__ void
cuckoo_insert(TYPE* key, /// key to insert
              TYPE* value, /// value to insert
              TYPE size, /// insert size
              int* resize,/// insert error?
              int* iterator_count=NULL);

__global__ void
cuckoo_search(TYPE* key, /// key to s
              TYPE* value, /// value to s
              TYPE siz); /// s size

__device__ __forceinline__ bool
exch_of_choose_simd(
        bucket_t *b,
        int simd_lane,
        int table_no_hash,
        Entry &entry,
        int evict_num);


void GPU_cuckoo_resize_up(int num_table_to_resize,
                          TYPE old_size,
                          bucket* new_table,
                          cuckoo *h_table);

void GPU_cuckoo_resize_down(int num_table_to_resize,
                            TYPE old_size,
                            bucket* new_table,
                            cuckoo *h_table);
/// dubug
void GPU_show_table();
void gpu_computer_table();

// all the pointer are device pointer
void gpu_lp_insert(TYPE* key,
                   TYPE* value,
                   TYPE size,
                   int* resize,
                   int* iterator_count=NULL);

void gpu_lp_search(TYPE* key,
                   TYPE* ans,
                   TYPE size);

void gpu_lp_delete(TYPE* key,
                   TYPE* ans,
                   TYPE size);

void gpu_lp_set_table(cuckoo* h_table);


#endif //LINEAR_LINEAR_PROBE_H
