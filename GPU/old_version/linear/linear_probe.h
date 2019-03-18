


#ifndef LINEAR_LINEAR_PROBE_H
#define LINEAR_LINEAR_PROBE_H

#include "dy_hash.h"
#include "linear_probe.h"
#include <helper_functions.h>
#include <helper_cuda.h>




__device__  bool
Add(TTT k,TTT v,TTT size);

__device__  bool
Delete(TTT k,TTT size );

__device__  TTT
Search(TTT k,TTT size);



// pubilc api
void gpu_lp_kernel(TTT* key,TTT* ans,TTT size,TTT _size,TTT *op);
void gpu_lp_init(TTT *k,TTT *v);
void gpu_lp_getnum(TTT table_size);


#endif //LINEAR_LINEAR_PROBE_H
