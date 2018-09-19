//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_LINEAR_PROBE_H
#define LINEAR_LINEAR_PROBE_H

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
    TYPE Lsize[TABLE_NUM];
    /// Lock for bucket lock,use atomicCAS
    TYPE *Lock;
    uint2 hash_fun[TABLE_NUM];
}cuckoo;



__global__ void
cuckoo_insert(TYPE* key, /// key to insert
              TYPE* value, /// value to insert
              TYPE size, /// insert size
              int* resize); /// insert error?

__global__ void
cuckoo_search(TYPE* key, /// key to s
              TYPE* value, /// value to s
              TYPE siz); /// s size


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

// all the pointer are device pointer
void gpu_lp_insert(TYPE* key,
                   TYPE* value,
                   TYPE size,
                   int* resize);

void gpu_lp_search(TYPE* key,
                   TYPE* ans,
                   TYPE size);

void gpu_lp_delete(TYPE* key,
                   TYPE* ans,
                   TYPE size);

void gpu_lp_set_table(cuckoo* h_table);


#endif //LINEAR_LINEAR_PROBE_H
