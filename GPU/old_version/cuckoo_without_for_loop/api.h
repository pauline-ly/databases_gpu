//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_API_H
#define LINEAR_API_H

#include "dy_hash.h"
#include "cuckoo.h"



class hashAPI{
    //device pointer
    cuckoo* hash_table;
    TTT *rehash;
    int table_size;
    int num_size;
public:
    explicit hashAPI(int size);

    ~hashAPI();

    void hash_insert(int *k, int *v,int size);
    
    void hash_kernel(int *k, int *v,int size,int *op);

    void hash_search(int *k, int *v,int size);

    void hash_delete(int *k,int *ans,int size);


};


#endif //LINEAR_API_H
