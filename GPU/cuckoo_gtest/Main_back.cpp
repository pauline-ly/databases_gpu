//
// Created by jing on 2018/7/1.
//

#include "cuckoo.h"
#include "api.h"
#include "mt19937ar.h"
#include "gtest/gtest.h"

extern float mytime;


using namespace std;

void check_search_result(TTT* key,TTT* check,TTT* value,TTT size) {
    TTT tmp = 0;
    for (TTT i = 0; i < size; i++) {
        if (check[i] != value[i]) {
            /// more information
//            if (tmp < 20)
//                printf("check i:%d error k:%d search:%d except:%d \n", i, key[i], check[i], value[i]);
            tmp++;
        }
    }
    EXPECT_LE(tmp*100.0/size,2)<<"size:%d",size;
    printf("\t%d/%d not pass:abort %.2f\n", tmp, size, tmp * 1.0 / size);
    memset(check,0,sizeof(TTT)*size);
}




bool  test_size(TTT size) {
    mytime=0;

    printf("\tsize:%d\n", size);


    /// set a bigger pool
    size *= 2;
    /// stream init



    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());
    auto *key = new TTT[size];
    auto *value = new TTT[size];
    auto *search = new TTT[size];
    auto *chck = new TTT[size];

    GenerateUniqueRandomNumbers(key, size);

    for (TTT i = 0; i < size; i++) {
        //key[i]=2*i+3;
        key[i] = key[i] >> 1;
        value[i] = 3 * i + 3 + 1;
        search[i] = 2 * i + 3 + 1;
        chck[i] = 0;
    }

    size /= 2;

    GpuTimer timer;


    hashAPI h(size * 1.5);

    /// hash table init




    timer.Start();

    /// <<<insert>>> after insert ,the size 100/200
    h.hash_insert(key, value, size);


    //cudaDeviceSynchronize();
    timer.Stop();

    /// sec
    double diff = timer.Elapsed()*1000000;
    printf("insert：the time is %.2lf us, ( %.2f Mops)\n",
           (double) diff, (double) (size) / diff);


    //printf("\nfirst search,only insert\n");
    timer.Start();

    /// <<<search>>>
    h.hash_search(key, chck, size);

    cudaDeviceSynchronize();
    timer.Stop();

    diff = timer.Elapsed()*1000000;
    printf("search：the time is %.2lf us ( %.2f Mops)\n",
           (double) diff, (double) (size) / diff);
    check_search_result(key,chck,value,size);

    checkCudaErrors(cudaGetLastError());

#if __show_table
    GPU_show_table();
#endif

//    timer.Start();
//    /// after resize 100/175;
////    h.resize_low();
//    //h.resize_low();
//    cudaDeviceSynchronize();
//    timer.Stop();
//
//    diff = timer.Elapsed()*1000000;
//    printf("resize：the time is %.2lf us, speed is %.2f Mops\n",
//           (double) diff, (double) (size) / diff);


//    mytime += diff;
//    printf("main mytime:%lf\n",mytime);

//    printf("\nsecond search, after resize \n");
    h.hash_search(key, chck, size);
    check_search_result(key,chck,value,size);
#if __show_table
    GPU_show_table();
#endif
    cudaDeviceSynchronize();
    /// set next insert data
    TTT* a=&key[size];
    TTT* b=&value[size];

    checkCudaErrors(cudaGetLastError());

    /// after insert 125/175
    h.hash_insert(a,b,size/4);
    checkCudaErrors(cudaGetLastError());

//    printf("\nthird search, after insert again\n");
    timer.Start();

    h.hash_search(key, chck, size);

    cudaDeviceSynchronize();
    timer.Stop();
    diff = timer.Elapsed()*1000000;
    printf("search：the time is %.2lf us, speed is %.2f Mops\n",
           (double) diff, (double) (size) / diff);
    check_search_result(key,chck,value,size);


    printf("¶¶¶test %d complete\n\n\n", size);
    checkCudaErrors(cudaGetLastError());
    delete[] key;
    delete[] value;
    delete[] search;
    delete[] chck;
    return true;
}


//
//
//int main(){
//
////    assert(test_size(128));
////    assert(test_size(1<<6));
////    assert(test_size(1<<8));
////    assert(test_size(1<<10));
////    assert(test_size(1<<12));
////    assert(test_size(1<<14));
////    assert(test_size(1<<16));
////    assert(test_size(1<<18));
////    assert(test_size(1<<20));
////
//
//
//
//}



int main(int argc, char** argv) {


    init_genrand(10327u);
   // printf("rand1 :%d",genrand_int32());

    TTT size=1024;

    if (argc >=2 )
    {
        size=(TTT)atoi(argv[1]);
    }

    test_size(size);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();


}




//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////     test different size                  //////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////

//class for index
class HastTest : public::testing::TestWithParam<int> { };

INSTANTIATE_TEST_CASE_P(TestDiffSize, HastTest, testing::Values(100,200,1024,2048,4096,1<<12,1<<16,1<<20,1<<22));

TEST_P(HastTest, TestDiffSize)
{
    int n=GetParam();
    EXPECT_TRUE(test_size(n));
}


//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////     test different input                 //////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////

#define SIZE_TEST 10000
class HashBoundaryTest : public ::testing::Test {
protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};


TEST(HashBoundaryTest, InsertsameKeyValue) {
    auto *key=new TTT[SIZE_TEST];

    auto *value=new TTT[SIZE_TEST];

    auto *search=new TTT[SIZE_TEST];
    auto *chck=new TTT[SIZE_TEST];

    for(int i=0;i<SIZE_TEST;i++){
        key[i]=11;
        value[i]=9;
        search[i]=i;
        chck[i]=0;
    }


    hashAPI h5(SIZE_TEST);
    h5.hash_insert(key,value,SIZE_TEST);
    h5.hash_search(search,chck,SIZE_TEST);

    for(int i=1;i<SIZE_TEST;i++){
        if(i!=11){
            EXPECT_EQ(0,chck[i])<<i<<key[i];
        }else{
            EXPECT_EQ(9,chck[i])<<i;
        }
    }

    memset(chck,0,SIZE_TEST*sizeof(int));
    h5.hash_search(key,chck,SIZE_TEST);
    for(int i=0;i<SIZE_TEST;i++){
        EXPECT_EQ(chck[i],9)<<key[i];
    }

    delete[] key;
    delete[] value;
    delete[] search;
    delete[] chck;

}


TEST(HashBoundaryTest, RandTestKV) {
    auto *key=new TTT[SIZE_TEST];
    auto *value=new TTT[SIZE_TEST];
    auto *search=new TTT[SIZE_TEST];
    auto *chck=new TTT[SIZE_TEST];

    for(int i=0;i<SIZE_TEST;i++){
        search[i]=key[i]=rand()%10000000;
        value[i]=i;
        chck[i]=0;
    }


    hashAPI h(SIZE_TEST);
    h.hash_insert(key,value,SIZE_TEST);
    h.hash_search(search,chck,SIZE_TEST);
    check_search_result(key,chck,value,SIZE_TEST);



    delete[] key;
    delete[] value;
    delete[] search;
    delete[] chck;

}



//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////        simple stream test                //////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////

bool  stream_test_size(TTT size,TTT init_size,int STREAM_NUM) {


    printf("\tsize:%d\n", size);

    /// set a bigger pool
    size *= STREAM_NUM;

    /// alloc
    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());
    auto *key = new TTT[size];
    auto *value = new TTT[size];
//    auto *search = new TTT[size];
    auto *chck = new TTT[size];
    ///init data
    GenerateUniqueRandomNumbers(key, size);
    for (TTT i = 0; i < size; i++) {
        //key[i]=2*i+3;
        key[i] = key[i] >> 1;
        value[i] = 3 * i + 3 + 1;
//        search[i] = 2 * i + 3 + 1;
        chck[i] = 0;
    }




    /// set small input
    size /= STREAM_NUM;

    /// set stream data
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
    GpuTimer alltimer;
    alltimer.Start();


    GpuTimer timer[STREAM_NUM];
    /// hash table init
    hashAPI h(init_size, stream, STREAM_NUM);
    double diff[STREAM_NUM];

    /// i use stream[i]
    for(int i=0;i<STREAM_NUM;i++){


        timer[i].Start(stream[i]);
        /// <<<insert>>> after insert ,the size 100/200
        h.hash_insert(key_s[i], value_s[i], size,stream[i]);

        //// printf
        timer[i].Stop(stream[i]);
        diff[i] = timer[i].Elapsed() * 1000000;
        printf("insert：the time is %.2lf us, ( %.2f Mops)\n",
               (double) diff[i], (double) (size) / diff[i]);



        timer[i].Start(stream[i]);

        /// <<<search>>>
        h.hash_search(key_s[i], chck_s[i], size,stream[i]);

//        cudaDeviceSynchronize();
        timer[i].Stop(stream[i]);

        diff[i] = timer[i].Elapsed() * 1000000;
        printf("search：the time is %.2lf us ( %.2f Mops)\n",
               (double) diff[i], (double) (size) / diff[i]);
        check_search_result(key_s[i], chck_s[i], value_s[i], size);
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



TEST(HashStreamTEST, TwoStreamWithoutRehash){
    EXPECT_TRUE(stream_test_size(1000000,1000000*2.5,2));
}

TEST(HashStreamTEST, FourStreamWithoutRehash){
    EXPECT_TRUE(stream_test_size(1000000,1000000*10,8));
}

TEST(HashStreamTEST, EightStreamWithoutRehash){
    EXPECT_TRUE(stream_test_size(1000000,1000000*10,8));
}


bool  stream_test_size_mix(TTT size,TTT init_size,int STREAM_NUM) {


    printf("\tsize:%d\n", size);

    /// set a bigger pool
    size *= STREAM_NUM;

    /// alloc
    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());
    auto *key = new TTT[size];
    auto *value = new TTT[size];
//    auto *search = new TTT[size];
    auto *chck = new TTT[size];
    ///init data
    GenerateUniqueRandomNumbers(key, size);
    for (TTT i = 0; i < size; i++) {
        //key[i]=2*i+3;
        key[i] = key[i] >> 1;
        value[i] = 3 * i + 3 + 1;
//        search[i] = 2 * i + 3 + 1;
        chck[i] = 0;
    }




    /// set small input
    size /= STREAM_NUM;

    /// set stream data
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


    GpuTimer alltimer;
    alltimer.Start();

//    GpuTimer timer[STREAM_NUM];
    /// hash table init
    hashAPI h(init_size, stream, STREAM_NUM);
//    double diff[STREAM_NUM];

    /// i use stream[i]
    for(int i=0;i<STREAM_NUM;i++){


//        timer[i].Start(stream[i]);
        /// <<<insert>>> after insert ,the size 100/200
        h.hash_insert(key_s[i], value_s[i], size,stream[i]);

        //// printf
//        timer[i].Stop(stream[i]);
//        diff[i] = timer[i].Elapsed() * 1000000;
//        printf("insert：the time is %.2lf us, ( %.2f Mops)\n",
//               (double) diff[i], (double) (size) / diff[i]);

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