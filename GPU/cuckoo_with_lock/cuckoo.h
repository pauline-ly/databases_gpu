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
              TTT* resize, /// insert error?
              cuckoo* table, /// hash table
              TTT table_size);

__global__ void
cuckoo_kernel(TTT* key, /// key to insert
              TTT* value, /// value to insert
              TTT size, /// insert size
              TTT* resize, /// insert error?
              cuckoo* table, /// hash table
              TTT table_size,
              TTT *op);


__global__ void
cuckoo_search(TTT* key, /// key to s
              TTT* value, /// value to s
              TTT size, /// s size
              cuckoo* table /// hash table
              );

void gpu_rehash(TTT old_size,TTT new_table_size);
// pubilc api
// all the vertor are device pointer
void gpu_lp_insert(TTT* key,TTT* value,TTT size,TTT* resize,cuckoo *table,TTT &table_size);
//void gpu_lp_delete();
void gpu_lp_search(TTT* key,TTT* ans,TTT size,cuckoo* table);

void gpu_lp_kernel(TTT* key,TTT* value,TTT size,TTT* resize,cuckoo *table,TTT &table_size,TTT *op);



#endif //LINEAR_LINEAR_PROBE_H
