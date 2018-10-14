//
// Created by jing on 2018/7/1.
//

#include "api.h"
#include "mt19937ar.h"
#include "libgpuhash.h"

using namespace std;
TYPE *key;

void check_search_result(TYPE* key,TYPE* check,TYPE* value,TYPE size) {
    TYPE tmp = 0;
    for (TYPE i = 0; i < size; i++) {
        if (check[i] !=1) {
            /// more information
//            if (tmp < 20)
//                printf("check i:%d error k:%d search:%d except:%d \n", i, key[i], check[i], value[i]);
            tmp++;
        }
    }
    EXPECT_LE(tmp*100.0/size,2)<<"size:%d",size;
    printf("\t%d/%d not pass:abort %.2f\n", tmp, size, tmp * 1.0 / size);
    memset(check,0,sizeof(TYPE)*size);
}



bool  test_size(TYPE size,double usage) {

    printf("\n\n\n\tsize:%d usage:%lf\n", size,usage);

    /// set a bigger pool
    size *= 2;

    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());

    auto *value = new TYPE[size];
    auto *search = new TYPE[size];
    auto *chck = new TYPE[size];
    /// genrand_int32() may turn key<0 , fix to genrand_int31()
    for (TYPE i = 0; i < size; i++) {
        value[i] = i*3+2;
        search[i] = 2 * i + 3 + 1;
        chck[i] = 0;
    }
    size /= 2;

    /// hash table init
    hashAPI h(size/usage);

    h.hash_insert(key, value, size);

    h.hash_search(key, chck, size);
    check_search_result(key,chck,value,size);

//    h.resize(size*usage*1.2);
//
//    h.hash_search(key,chck,size);
//
//
//    check_search_result(key,chck,value,size);
//
//    h.hash_delete(key,value,size);

    delete[] value;
    delete[] search;
    delete[] chck;
    return true;
}

bool  test_size_new_api() {
    int run_time=44;
    int batch_num_every_run_time=3;
    int pool_size=BATCH_SIZE*run_time*batch_num_every_run_time;
    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());
    auto *key = new TYPE[pool_size];
    auto *value = new TYPE[pool_size];


    /// genrand_int32() may turn key<0 , fix to genrand_int31()
    GenerateUniqueRandomNumbers(key, pool_size);
    for (TYPE i = 0; i < pool_size; i++) {
        value[i] = 3 * i + 3 + 1;
    }




    /// hash table init
    /// every table 1M
    hashAPI h(TABLE_NUM*1024*1024);
    /// pool init
    h.set_data_to_gpu(key,value,pool_size);


    for(int i=0;i<run_time;i++) {


        h.hash_insert_batch();


/// search

        h.hash_search_batch();

//
///// delete
//        timer.Start();
//        h.hash_delete_batch();
//        cudaDeviceSynchronize();
//        timer.Stop();
//        diff = timer.Elapsed() * 1000000;
//        printf("delete：the time is %.2lf us ( %.2f Mops)\n",
//               (double) diff, (double) (BATCH_SIZE) / diff);
//


    }

    printf("del\n\n");
    for(int i=0;i<run_time/4;i++) {
        h.hash_delete_batch();

    }


    checkCudaErrors(cudaGetLastError());
    delete[] key;
    delete[] value;
    return true;
}



#define  max_size 10000000
int main(int argc, char** argv) {
    init_genrand(10327u);

    key = new TYPE[max_size];
    GenerateUniqueRandomNumbers(key, max_size);

    /// set same input


//    double usage[]={0.5};
    double usage[]={0.95,0.9,0.85,0.8,0.75,0.7,0.5};
    for(double us:usage){
        for (TYPE size = 100000; size <=max_size ; size*=10) {
            test_size(size,us);
        }
    }

//    test_size_new_api();


//    testing::InitGoogleTest(&argc, argv);
//    return RUN_ALL_TESTS();
    return 0;
}

