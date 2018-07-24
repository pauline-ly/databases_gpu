//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_DY_HASH_H
#define LINEAR_DY_HASH_H

#include <host_defines.h>
#include <cuda_runtime.h>

#include "cstdio"

/// type
#define TTT int
#define TYPE int

/// hash function
#define _PRIME 43112609


/// grow para
#define NUM_OVERFLOW_ratio 0.8
#define NUM_grow_ratio 2

/// cuckoo prar
#define BUCKET_SIZE 32
#define MAX_ITERATOR 100
#define TABLE_NUM 5

/// kernel
#define NUM_THREADS 512
#define NUM_BLOCK 64



#endif //LINEAR_DY_HASH_H
