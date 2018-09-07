//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_DY_HASH_H
#define LINEAR_DY_HASH_H

#include <host_defines.h>
#include <cuda_runtime.h>
#include "gputimer.h"
#include "cnmem.h"

#include "cstdio"
#include "random_numbers.h"

/// type
#define TTT unsigned int
#define TYPE unsigned int

/// hash function
#define _PRIME 4294967291u


/// grow para
#define NUM_OVERFLOW_ratio 0.8
#define NUM_grow_ratio 2

#define SIZE_UP(size) ((size)*0.8)
#define SIZE_LOW(size) ((size)*0.5)

/// cuckoo prar
#define BUCKET_SIZE 32
#define MAX_ITERATOR 7
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
#define down_size_debug 0
#define down_size_casinsert_debug 0

#endif //LINEAR_DY_HASH_H
