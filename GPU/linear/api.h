//
// Created by jing on 2018/6/8.
//

#ifndef LINEAR_API_H
#define LINEAR_API_H

#include "dy_hash.h"
#include "linear_probe.h"
#define TTT int


class hashAPI{
    //device pointer
    TTT *dkey;
    TTT *dvalue;
    TTT *rehash;
    int table_size;
public:
    explicit hashAPI(int size);

    ~hashAPI();

    void hash_kernel(int *k, int *v,int size,int *op);

    void hash_getnum();


};


#endif //LINEAR_API_H
