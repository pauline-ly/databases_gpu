//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_API_H
#define LINEAR_API_H

#include "dy_hash.h"
#include "cuckoo.h"



class hashAPI{
    //device pointer
    cuckoo* h_table;
    //cuckoo* hash_table;
    TTT *rehash;
    TTT table_size; /// all it can contain
    TTT num_size;   /// size it has
    void api_resize(TTT after_insert_size,cudaStream_t stream=0);
    cnmemDevice_t device;

public:
    void resize_low(cudaStream_t stream=0);
    void resize_up(cudaStream_t stream=0);
    explicit hashAPI(TTT size);
    hashAPI(TTT size,cudaStream_t a[],int NUM_STREAM);
    ~hashAPI();
    void hash_insert(TTT *k, TTT *v,TTT size,cudaStream_t stream=0);
    void hash_search(TTT *k, TTT *v,TTT size,cudaStream_t stream=0);
    //void hash_delete(TTT *k,TTT *ans,TTT size);
};


#endif //LINEAR_API_H
