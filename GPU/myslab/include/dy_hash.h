//
// Created by jing on 2018/7/1.
//

#ifndef LINEAR_DY_HASH_H
#define LINEAR_DY_HASH_H

#include <host_defines.h>
#include <cuda_runtime.h>

//#include "gtest/gtest.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include "cstdio"
#include <vector>

/// type
#define TYPE unsigned int


/// hash function
#define PRIME_uint 4294967291u
#define BUCKET_NUM  32
#define BUCKET_SIZE_BIT 5
#define BUCKET_SIZE (1<<BUCKET_SIZE_BIT)
//#define MASK_CHECK_key 0xfffffffc
//#define MASK_CHECK_kv  0x2aaaaaaa
/// k loc: 0 2 4 6 v loc: 1 3 5 7
#define MASK_CHECK_kv  0x15555555

/// seting of test
#define TABLE_NUM 1000001
#define NUM_DATA  10000000

#define THREAD_NUM 512
#define BLOCK_NUM 128

//#define MEM_POOL_NUM (THREAD_NUM* BLOCK_NUM /32)
#define MEM_POOL_NUM BLOCK_NUM
// zuida  4*1024*1024*1024 / 32*4*BLOCK_NUM =32*1024*1024/BLOCK_NUM
#define NODE_OF_POOL 128*1024


/// batch test
//#define SIZE_KV (sizeof(TYPE)*3)  /// size :k v hash
//#define max_input_size ((2UL*1024*1024*1024)/SIZE_KV)

/// seting of hash
#define CAS1 1
#define ADD0 0
#define using_CAS_OR_ADD CAS1
#define using_block_

/// for debug
#define size_info 0
#define cuckoo_cu_speed 1
#define insert_debug 0
#define CAS_debug 0
#define kv_in_Node 0
#define show_list_in_kernel 0
#define show_node_alloc 0
#define show_memory 0
#define check_memory_alloc 0
#define api_info 0
#define show_table_flag 0
#define separate_time 1
#define api_compute 0
#define head_info 0





#endif //LINEAR_DY_HASH_H
