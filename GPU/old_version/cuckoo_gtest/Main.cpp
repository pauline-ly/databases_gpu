//
// Created by jing on 2018/7/1.
//

#include "cuckoo.h"
#include "api.h"
#include "mt19937ar.h"
#include "gtest/gtest.h"

extern float mytime;


using namespace std;


int main(int argc, char** argv) {


    init_genrand(10327u);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();


}




//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////        simple stream test                //////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////



bool  stream_test_size_mix(TTT size,TTT init_size,int STREAM_NUM) {
    printf("\tsingle input size:%d , times:%d\n", size,STREAM_NUM);

    /// set a bigger pool
    size *= STREAM_NUM;

    /// alloc
    cudaDeviceReset();
    auto *key = new TTT[size];
    auto *value = new TTT[size];
    auto *chck = new TTT[size];
    ///init data
    GenerateUniqueRandomNumbers(key, size);
    for (TTT i = 0; i < size; i++) {
        //key[i]=2*i+3;
        key[i] = key[i] >> 1;
        value[i] = 3 * i + 3 + 1;
        chck[i] = 0;
    }

    /// set small input
    size /= STREAM_NUM;


    /// set stream data pointer
    TTT* key_s[STREAM_NUM];
    TTT* value_s[STREAM_NUM];
    TTT* chck_s[STREAM_NUM];
    for (int j = 0; j < STREAM_NUM; ++j) {
        key_s[j]=&key[j*size];
        value_s[j]=&value[j*size];
        chck_s[j]=&chck[j*size];
    }

    /// stream init
    cudaStream_t stream[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamCreate(&stream[i]);
    }

    /// hash table init
    hashAPI h(init_size, stream, STREAM_NUM);


    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = (size_t)1024*1024*1024; /// 1G
    device.numStreams=STREAM_NUM;
    device.streams=stream;
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);

    TTT* rehash;
    cnmemMalloc((void **) &rehash, sizeof(TTT),0);

    TTT* d_keys[STREAM_NUM];
    TTT* d_value[STREAM_NUM];

    checkCudaErrors(cudaGetLastError());
    GpuTimer alltimer;
    alltimer.Start();
    /// i use stream[i]
    for(int i=0;i<STREAM_NUM;i++){

        checkCudaErrors(cudaGetLastError());
        cnmemMalloc((void**)&d_keys[i], sizeof(TTT)*size,stream[i]);
        checkCudaErrors(cudaGetLastError());
        cudaMemcpyAsync(d_keys[i], key_s[i], sizeof(TTT)*size, cudaMemcpyHostToDevice,stream[i]);
        checkCudaErrors(cudaGetLastError());

        cnmemMalloc((void**)&d_value[i], sizeof(TTT)*size,stream[i]);
        cudaMemcpyAsync(d_value[i], value_s[i], sizeof(TTT)*size, cudaMemcpyHostToDevice,stream[i]);
        checkCudaErrors(cudaGetLastError());
        // does size need be copy first
        gpu_lp_insert(d_keys[i],d_value[i],size,rehash,stream[i]);


    }


    alltimer.Stop();

    double alldiff = alltimer.Elapsed() * 1000000;
    printf("<<<>>>ALL-insert-search-stream：the time is %.2lf us ( %.2f Mops)\n",
           (double) alldiff, (double) (size*STREAM_NUM) / alldiff);




    printf("¶¶¶test %d complete\n\n\n", size);
    /// free
    delete[] key;
    delete[] value;
    //delete[] search;
    delete[] chck;
    checkCudaErrors(cudaGetLastError());
    return true;
}


TEST(HashStreamTEST, Four){
    EXPECT_TRUE(stream_test_size_mix(1000000,1000000*5,4));
}