//
// Created by jing on 2018/7/1.
//

#include "cuckoo.h"
#include "api.h"
#include "mt19937ar.h"
#include <stdlib.h>
#include <cuda_profiler_api.h>

using namespace std;
#define NUM_DATA 100000000
void check_search_result(TYPE *key, TYPE *check, TYPE *value, TYPE size) {
    TYPE tmp = 0;
    for (TYPE i = 0; i < size; i++) {
        if(key[i] == 0) continue;
        if (check[i] == 0) {
            /// more information
            if (tmp < 10)
//                 printf("check i:%d error k:%d search:%d except:%d \n", i, key[i], check[i], value[i]);
            tmp++;
        }
    }
//    EXPECT_LE(tmp * 100.0 / size, 2) << "size:%d", size;
    printf("\t%d/%d not pass:abort %.2f\n", tmp, size, tmp * 1.0 / size);
    memset(check, 0, sizeof(TYPE) * size);
}

unsigned int *DataRead_CMode(char *filename) {
//     printf("info:filename:%s\n",filename);
    int size=NUM_DATA;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/twitter-unique.dat")==0)
        size=25297548;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/tpch-unique.dat")==0)
        size=90319761;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/ali-unique.dat")==0)
        size=4583941;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/reddit-unique.dat")==0)
        size=90319761;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/twitter.dat")==0)
        size=size/2;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/tpc-h.dat")==0)
        size=size/2;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/l32.dat")==0)
        size=size/10;
    FILE *fid;
    fid = fopen(filename, "rb");
    unsigned int *pos;
    pos = (unsigned int *) malloc(sizeof(unsigned int) * size);//申请内存空间，大小为n个int长度
    int i;
    if (fid == NULL) {
        printf("the data file is unavailable.\n");
        exit(1);
        return pos;
    }
    fread(pos, sizeof(unsigned int),size, fid);
    fclose(fid);
    return pos;
}


bool test_size_new_api(char *filename, int r) {
    // int run_time=1;
    // int batch_num_every_run_time=1;
    // int pool_size=BATCH_SIZE*run_time*batch_num_every_run_time;
//     int pool_size = 100000000;
    int pool_size=NUM_DATA;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/twitter-unique.dat")==0)
        pool_size=25297548;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/tpch-unique.dat")==0)
        pool_size=90319761;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/ali-unique.dat")==0)
        pool_size=4583941;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/reddit-unique.dat")==0)
        pool_size=90319761;
     if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/twitter.dat")==0)
       pool_size=pool_size/2;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/tpc-h.dat")==0)
        pool_size=pool_size/2;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/l32.dat")==0)
        pool_size=pool_size/10;
    int num_more = 10;
    int batch_num = pool_size / BATCH_SIZE;
    int batch_every_time = batch_num / num_more;
    int num_search = 2;

    int num_less = r;
//     printf("batch_num = %d\n",batch_num);
//     printf("batch_every_time = %d\n",batch_every_time);

    double usage[] = { 0.95,0.9,0.85, 0.8, 0.75, 0.7,0.65,0.6,0.55,0.5};
    int num_usage = 10;
//     double usage[] = {0.55,0.85};
//     int num_usage = 2;
//    int num_usage = 1;
//    double usage[] = {0.85};

    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());
    TYPE *key = nullptr;// = new TYPE[pool_size];
    auto *value = new TYPE[pool_size];
    auto *chck = new TYPE[pool_size];
    TYPE *sea = new TYPE[pool_size];



    key = DataRead_CMode(filename);

//     GenerateUniqueRandomNumbers(sea, pool_size);
//    char* search_name;
//        search_name="/home/udms/ly/GPU_Hash/finally-test/data/search-unique.dat";
//     sea=DataRead_CMode(search_name);

    for (TYPE i = 0; i < pool_size; i++) {
//         key[i]=100;
        value[i] =i+1;
       chck[i]  = 0;
    }
    int up_time = 0;
    int down_time = 0;
    double resize_load = 1 + (down_time - up_time) * 1.0 / TABLE_NUM;

//    for (int j = num_usage-1; j >-1; j--) {
//        //         printf("--------------------------------\n");
//        printf("usage = %lf   ", usage[j]);
#if 1
    for (int j = num_usage-1; j >-1; j--) {
        //         printf("--------------------------------\n");
        printf("================== usage = %lf ==================\n", usage[j]);
        int first_insert = pool_size * 0.8;
        int second_insert = pool_size * 0.2;
        hashAPI h(first_insert * resize_load / usage[j]);
        h.hash_insert(key, value, first_insert);
        h.hash_search(sea, chck, first_insert);
        gpu_computer_table();
        check_search_result(key, chck, value, first_insert);

        printf("\tsecond inserting\n", usage[j]);
        h.resize_up();
        h.hash_insert(key+first_insert, value+first_insert, second_insert);
        h.hash_search(sea+first_insert, chck+first_insert, second_insert);
        gpu_computer_table();
        check_search_result(key + first_insert, chck + first_insert, value + first_insert, second_insert);
    }
#endif
#if 0
        gpu_computer_table();
        for (int ut = 0; ut < up_time; ut++) {
            h.resize_up();
            h.hash_search(key, value, pool_size);
            check_search_result(key, value, key, pool_size);
            gpu_computer_table();
        }

        for (int ut = 0; ut < down_time; ut++) {
            h.resize_low();
            h.hash_search(key, value, pool_size);
            check_search_result(key, value, key, pool_size);
            gpu_computer_table();
        }
#endif


#if 0
        hashAPI h(TABLE_NUM*1024*1024);

        /// pool init
        h.set_data_to_gpu(key,value,pool_size);


    //     printf("......\n");
        int i;
//         int cou=0;
       GpuTimer time;
       time.Start();
        for(i=0;i<batch_every_time/2+1;i++)
        {
            for(int j=0;j<num_more;j++)
            {
               
                  h.hash_insert_batch();
//                 
            }
//             printf("insert end..\n");
            for(int j=0;j<num_search;j++)
            {
              
                   h.hash_search_batch();
            }
//             printf("search end..\n");
            for(int j=0;j<num_less;j++)
            {
    
                   h.hash_delete_batch();
//                  
            }
        }
//         printf("end...%d\n",i);

        h.batch_reset();
        for(i=0;i<batch_every_time/2-1;i++)
        {
            
//             printf("i = %d\n",i);
            for(int j=0;j<num_less;j++)
            {
                
                  h.hash_insert_batch();
//                 
            }
//             printf("insert end..\n");
            for(int j=0;j<num_search;j++)
            {
              
                   h.hash_search_batch();
//                 
            }
//             printf("search end..\n");
            for(int j=0;j<num_more;j++)
            {
                
                   h.hash_delete_batch();
//                  

            }
//             printf("delete end..\n");
        }
     time.Stop();
    double diff = time.Elapsed() * 1000000;
    printf("%d %.2lf %.2f ",
           r,(double)diff, (double)(BATCH_SIZE*(num_more+num_search+num_less)*batch_every_time) / diff);
#endif

    check_search_result(key, chck, value, pool_size);

    checkCudaErrors(cudaGetLastError());

}


//     delete[] key;
//     delete[] value;
//     return true;
// }


int main(int argc, char **argv) {

//     printf("process started\n");
    /// set same input
    // init_genrand(10327u);
    // testing::InitGoogleTest(&argc, argv);
    // return RUN_ALL_TESTS();
    if(TABLE_NUM > 4)
        printf("!!!\n!!!\n!!! cuckoo hash funtion need fix\n");
    char *filename;
    int r;
    if (argc > 1) {
//         num_more=atoi(argv[1]);
//         no_data=atoi(argv[2]);
        r=atoi(argv[1]);
        filename = argv[2];
    } else {
        printf("run using : ./linear file_neme");
    }


    test_size_new_api(filename,r);

    return 0;
}



