//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_LINEAR_PROBE_cu_H
#define LINEAR_LINEAR_PROBE_cu_H

#include "dy_hash.h"
#include "cuckoo.h"
#include <helper_functions.h>
#include <helper_cuda.h>


/// isolated kv
struct bucket{
    TYPE key[BUCKET_SIZE];
    TYPE value[BUCKET_SIZE];
};

/// definition structure
typedef struct mycuckoo{
    /// table , store kv ,
    /// in detail , kv is  isolated
    bucket *table[TABLE_NUM];
    /// size of sub table , determine rehash : L min resize
    TTT Lsize[TABLE_NUM];
    /// Lock for bucket lock,use atomicCAS
    TTT *Lock;
    uint2 hash_fun[TABLE_NUM];
}cuckoo;



__device__ TYPE get_next_loc(TYPE k,TYPE v,TYPE num_table);

__device__  bool
Add(TTT k,TTT v,TTT size);

//__device__ __host__ bool
// Delete(TTT k,TTT size );

__device__  int
Search(TTT k,TTT size);

__global__ void
cuckoo_insert(TTT* key, /// key to insert
              TTT* value, /// value to insert
              TTT size, /// insert size
              TTT* resize); /// insert error?

__global__ void
cuckoo_search(TTT* key, /// key to s
              TTT* value, /// value to s
              TTT size); /// s size


void GPU_cuckoo_resize_up(int num_table_to_resize,
                          int old_size,
                          bucket* new_table,
                          cuckoo *h_table,
                          cudaStream_t stream=0);

void GPU_cuckoo_resize_down(int num_table_to_resize,
                            int old_size,
                            bucket* new_table,
                            cuckoo *h_table,
                            cudaStream_t stream=0);



int choose_block_num(TTT size);

void GPU_show_table();

// all the vertor are device pointer
void gpu_lp_insert(TTT* key,
                   TTT* value,
                   TTT size,
                   TTT* resize,  /// for rehash
                   cudaStream_t stream=0);

//void gpu_lp_delete();
void gpu_lp_search(TTT* key,
                   TTT* ans,
                   TTT size,
                   cudaStream_t stream=0);

/// only used in api/init
void gpu_lp_set_table(cuckoo* h_table);


#endif //LINEAR_LINEAR_PROBE_H
