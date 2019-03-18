//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_API_H
#define LINEAR_API_H

#include "dy_hash.h"
#include "libgpuhash.h"
#include "gpu_hash.h"

#define get_table_bucket_size(num) ((h_table->Lsize[num_table_to_resize])>>5)

class hashAPI{

    /// hash table info
    char* d_hash_table;

    TYPE table_size; /// all key it can contain
    TYPE num_size;   /// size it has


    /// device memory manager
    cnmemDevice_t device;

    /// batch input
    int batch_num_to_op;
    int batch_del_num_to_op;
    ielem_t *d_elem_pool;
    ielem_t *d_elem_batch[BATCH_NUM];
    TYPE *d_check; /// for delete search

    void api_resize(TYPE after_insert_size);

public:
    explicit hashAPI(TYPE size);
    ~hashAPI();

    void hash_insert(TYPE *k, TYPE *v,TYPE size);
    void hash_search(TYPE *k, TYPE *ans,TYPE size);
    void hash_delete(TYPE *k,TYPE *v,TYPE size);

    bool set_data_to_gpu(TYPE *key,TYPE* value,TYPE size);
    void hash_insert_batch();
    void hash_search_batch();
    void hash_delete_batch();

    /// public for test
    void resize(TYPE);
};


#endif //LINEAR_API_H
