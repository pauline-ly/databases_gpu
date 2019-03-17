//
// Created by jing on 2019-03-15.
//

#ifndef LINEAR_CPU_API_H
#define LINEAR_CPU_API_H

#include "include/dy_hash.h"

typedef struct SlabNode{

}SNode;

template< class T>
class Cpu_api {
public:
    Cpu_api();

    void hash_insert(T* k,T* v,size_t);
    void hash_search(T* k,T* v,size_t);
    void hash_delete(T* k,T* v,size_t);

private:

    int* data_ptr_; // memptr
    std::vector<SNode*> node_List_;
    SNode** hash_table_ptr_;

    /// batch input
    int batch_num_to_op;
    int batch_del_num_to_op;
    int batch_search_num_to_op;
};




#endif //LINEAR_CPU_API_H
