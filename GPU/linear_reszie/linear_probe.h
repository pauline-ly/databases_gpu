//
// Created by jing on 2018/6/4.
//

#ifndef LINEAR_LINEAR_PROBE_H
#define LINEAR_LINEAR_PROBE_H

#include "dy_hash.h"
#include "linear_probe.h"
#include <helper_functions.h>
#include <helper_cuda.h>




__device__  bool
Add(TTT k,TTT v,TTT size);

//__device__ __host__ bool
// Delete(TTT k,TTT size );

__device__  int
Search(TTT k,TTT size);





// pubilc api
// all the vertor are device pointer
void gpu_lp_insert(TTT* key,TTT* value,TTT size,TTT *rehash,TTT &_size);
//void gpu_lp_delete();
void gpu_lp_search(TTT* key,TTT* ans,TTT size,TTT _size);
void gpu_lp_init(TTT *k,TTT *v);
void gpu_lp_getnum(TTT table_size);


#endif //LINEAR_LINEAR_PROBE_H
