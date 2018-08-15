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
#define MAX_ITERATOR 5
#define TABLE_NUM 4

/// kernel
//#define NUM_THREADS 512
//#define NUM_BLOCK 32
#define NUM_THREADS 512
#define NUM_BLOCK 512


/// for debug
#define insert_debug 0
#define search_debug 0
#define head_info 0
#define debug_big_data 0
#define __show_table 0
#define size_debug 0

#endif //LINEAR_DY_HASH_H
