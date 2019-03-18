//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_API_H
#define LINEAR_API_H

#include "dy_hash.h"
#include "cuckoo.h"

#define get_table_bucket_size(num) ((h_table->Lsize[num_table_to_resize])>>5)

class hashAPI{

    /// hash table info
    cuckoo* h_table;
    TYPE table_size; /// all it can contain
    TYPE num_size;   /// size it has

    /// rehash flag ,check long chain in inserting
    int *rehash;
    /// device memory manager
    cnmemDevice_t device;

    /// batch input
    int batch_num_to_op;
    int batch_del_num_to_op;
    int batch_search_num_to_op;
    TYPE *d_key_pool,*d_value_pool;
    TYPE *d_key_batch[BATCH_NUM],*d_value_batch[BATCH_NUM];
    TYPE *d_check; /// for delete search

    void api_resize(TYPE after_insert_size);

public:
    explicit hashAPI(TYPE size);
    ~hashAPI();

    void hash_insert(TYPE *k, TYPE *v,TYPE size);
    void hash_search(TYPE *k, TYPE *ans,TYPE size);
    void hash_delete(TYPE *k,TYPE *ans,TYPE size);

    bool set_data_to_gpu(TYPE *key,TYPE* value,TYPE size);
    void hash_insert_batch();
    void hash_search_batch();
    void hash_delete_batch();

    /// public for test
    void resize_low();
    void resize_up();

};


#endif //LINEAR_API_H
