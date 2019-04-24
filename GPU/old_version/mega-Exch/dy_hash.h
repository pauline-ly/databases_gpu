//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_DY_HASH_H
#define LINEAR_DY_HASH_H

#include <host_defines.h>
#include <cuda_runtime.h>
#include "gputimer.h"
#include "cnmem.h"
#include "gtest/gtest.h"
#include <helper_functions.h>
#include <helper_cuda.h>

#include "cstdio"
#include "random_numbers.h"

/// type
#define TYPE unsigned int


/// hash function
#define PRIME_uint 4294967291u


/// grow para
#define NUM_downsize_ratio 0.5
#define NUM_upsize_ratio  2

/// fill ratio control
#define Upper_Bound(size) ((size)*0.8)
#define Lower_Bound(size) ((size)*0.5)

/// batch test
#define SIZE_KV (sizeof(TYPE)*3)  /// size :k v hash
#define max_input_size ((1024*1024*1024)/SIZE_KV)
#define BATCH_NUM 1000
#define BATCH_SIZE 100000

/// cuckoo prar
#define BUCKET_SIZE_BIT 5
/// bucket size cannot change in there directly
#define BUCKET_SIZE (1<<BUCKET_SIZE_BIT)
#define MAX_ITERATOR 15
#define TABLE_NUM 3
/// 当evict 多次插入不进去时是否立即停止本次插入，等resize
#define MAX_ITERATOR_over_to_break_insert 0

#define RESIZE_UP_RATE 2
#define RESIZE_DOWN_RATE 0.5


/// kernel
//#define THREAD_NUM 512
//#define BLOCK_NUM 32
#define THREAD_NUM 512
#define BLOCK_NUM 512


/// for debug

#define size_info 1
#define down_size_debug 0
#define down_size_cas_insert_debug 0
#define api_show_message 1
#define cuckoo_cu_speed 1
#define insert_debug 0


#endif //LINEAR_DY_HASH_H
