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
#define MASK_CHECK_key 0xfffffffc
#define MASK_CHECK_kv  0x2aaaaaaa
#define TABLE_NUM 10



/// batch test
//#define SIZE_KV (sizeof(TYPE)*3)  /// size :k v hash
//#define max_input_size ((2UL*1024*1024*1024)/SIZE_KV)


/// gpu kernel
#define THREAD_NUM 512
#define BLOCK_NUM 512
//#define DELAY_DOWN_SIZE_TIME 20
// 每个warp一个池子
#define MEM_POOL_NUM (THREAD_NUM* BLOCK_NUM /32)
// 一个池子16个Node
#define NODE_OF_POOL 32



/// for debug

#define size_info 0
#define cuckoo_cu_speed 1
#define insert_debug 0
#define CAS_debug 0
#define kv_in_Node 1
#define show_memory 0
#define check_memory_alloc 0
#define api_info 0
#define show_table_flag 0
#define separate_time 1
#define api_compute 0
#define head_info 0


#define  check_search_result(key,check,value, size) { \
    TYPE tmp = 0;                                    \
    for (TYPE i = 0; i < size; i++) {                \
        if (check[i] !=value[i]) {                  \
           if (tmp < 0)                               \
                printf("check i:%d error k:%d search:%d except:%d \n", i, key[i], check[i], value[i]);\
            tmp++;                              \
        }                           \
    }                                       \
    if(tmp!=0)printf("\t%d/%d not pass:abort %.2f\n", tmp, size, tmp * 1.0 / size);         \
    memset(check,0,sizeof(TYPE)*size);              \
}


#endif //LINEAR_DY_HASH_H
