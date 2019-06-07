#ifndef LINEAR_DY_HASH_H
#define LINEAR_DY_HASH_H

#include <host_defines.h>
#include <cuda_runtime.h>
#include "gputimer.h"
#include "cnmem.h"
#include "gtest/gtest.h"

#include "cstdio"
#include "random_numbers.h"
#include "../test_uilt.h"
#include <stdint.h>

/// type
#define TYPE unsigned int
/// kernel
//#define THREAD_NUM_k 512
//#define BLOCK_NUM_k 32
#define THREAD_NUM_k 512
#define BLOCK_NUM_k 512

/// hash function
#define PRIME_uint 294967291u


/// grow para
#define NUM_OVERFLOW_ratio 0.8
#define NUM_grow_ratio 2

/// cuckoo prar
#define TABLE_NUM 4
#define MAX_ITER  15
#define ELEM_NUM_P_gpu 4

#define BUCKET_SIZE_P ELEM_NUM_P_gpu
#define BUCKET_SIZE (1<<BUCKET_SIZE_P)

/// 当evict 多次插入不进去时是否立即停止本次插入，等resize
#define MAX_ITERATOR_overflow_to_break_insert 0
#define INSERT_AFTER_DEL 0
#define update_on 0

/// api control
// 1: open rehash
#define no_rehash 1
// 1: open n choose 2
#define n_choose_2 1
// 1: open random insert when no better pos
#define cu_random 0
// 1: have a small slot
#define small_slot 0

/// do not fix
/// slot 101 * 32 kv
#define SLOT_SIZE 101
/// batch test
#define SIZE_KV (sizeof(TYPE)*2)
#define max_input_size ((1024*1024*1024)/SIZE_KV)


/// for debug
#define insert_debug 0
#define record_cannot_insert_num 0
#define search_debug 0
#define head_info_debug 0
#define show_table_debug 0
#define size_info_of_resize_debug 0
#define down_size_debug 0
#define down_size_cas_insert_debug 0
#define ITERAOTOR_COUNT_FLAG 1
#define api_info 0
#define cu_info 1




#endif //LINEAR_DY_HASH_H
