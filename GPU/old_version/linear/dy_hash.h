//
// Created by jing on 2018/6/4.
//

#ifndef LINEAR_DY_HASH_H
#define LINEAR_DY_HASH_H

#include <host_defines.h>
#include <cuda_runtime.h>

#include "cstdio"

// some common var...
#define TTT int
#define _PRIME 43112609
#define MAX_ITERATOR 100

//parameters
#define jump_prime 41
#define hp 4294967291  //max int32 prime
#define SLOTEMPTY ((Entry)0xffffffff00000000)
#define LINEAR 1
#define QUADRATIC 2
#define method 3  //differenf kind of probe 3:double hashing
#define kMaxProbes 3000
#define ha 2654435769
#define hb 11

#define NUM_THREADS 512
#define NUM_BLOCK 32



#endif //LINEAR_DY_HASH_H
