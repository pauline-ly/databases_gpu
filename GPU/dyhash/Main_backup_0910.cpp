//
// Created by jing on 2018/7/1.
//

#include "cuckoo.h"
#include "api.h"
#include "mt19937ar.h"


extern float mytime;


using namespace std;

void check_search_result(TYPE* key,TYPE* check,TYPE* value,TYPE size) {
    TYPE tmp = 0;
    for (TYPE i = 0; i < size; i++) {
        if (check[i] != value[i]) {
            /// more information
            if (tmp < 20)
                printf("check i:%d error k:%d search:%d except:%d \n", i, key[i], check[i], value[i]);
            tmp++;
        }
    }
    EXPECT_LE(tmp*100.0/size,2)<<"size:%d",size;
    printf("\t%d/%d not pass:abort %.2f\n", tmp, size, tmp * 1.0 / size);
    memset(check,0,sizeof(TYPE)*size);
}




bool  test_size(TYPE size) {
    mytime=0;

    printf("\tsize:%d\n", size);

    /// set a bigger pool
    size *= 2;

    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());
    auto *key = new TYPE[size];
    auto *value = new TYPE[size];
    auto *search = new TYPE[size];
    auto *chck = new TYPE[size];

    /// genrand_int32() may turn key<0 , fix to genrand_int31()
    GenerateUniqueRandomNumbers(key, size);

    for (TYPE i = 0; i < size; i++) {
        //key[i]=2*i+3;
        value[i] = 3 * i + 3 + 1;
        search[i] = 2 * i + 3 + 1;
        chck[i] = 0;
    }

    size /= 2;

    GpuTimer timer;

    /// hash table init
    hashAPI h(size * 1.1);



    timer.Start();

    /// <<<insert>>> after insert ,the size 100/200
    h.hash_insert(key, value, size);

    cudaDeviceSynchronize();
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

//#if show_table_debug
//    GPU_show_table();
//#endif

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
//    h.hash_search(key, chck, size);
//    check_search_result(key,chck,value,size);
//#if show_table_debug
//    GPU_show_table();
//#endif
//    cudaDeviceSynchronize();
//    /// set next insert data
//    TYPE* a=&key[size];
//    TYPE* b=&value[size];
//
//    checkCudaErrors(cudaGetLastError());
//
//    /// after insert 125/175
//    h.hash_insert(a,b,size/4);
//    checkCudaErrors(cudaGetLastError());
//
////    printf("\nthird search, after insert again\n");
//    timer.Start();
//
//    h.hash_search(key, chck, size);
//
//    cudaDeviceSynchronize();
//    timer.Stop();
//    diff = timer.Elapsed()*1000000;
//    printf("search：the time is %.2lf us, speed is %.2f Mops\n",
//           (double) diff, (double) (size) / diff);
//    check_search_result(key,chck,value,size);
//
//
//    printf("¶¶¶test %d complete\n\n\n", size);
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

    TYPE size=1024;

    if (argc >=2 )
    {
        size=(TYPE)atoi(argv[1]);
    }

    test_size(size);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();


}



//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////        test diff size                    //////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////


//class for index
class HastTest : public::testing::TestWithParam<int> { };

INSTANTIATE_TEST_CASE_P(TrueReturn, HastTest, testing::Values(100,200,1024,2048,4096,1<<12,1<<16,1<<20,1<<22));

TEST_P(HastTest, SizeTEST)
{

    int n=GetParam();
    EXPECT_TRUE(test_size(n));
}

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////        test diff input                   //////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////

class HashBoundaryTest : public ::testing::Test {
protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

#define SIZE_TEST 100
//same input
TEST(HashBoundaryTest, InsertsameKeyValue) {
    auto *key=new TYPE[SIZE_TEST];

    auto *value=new TYPE[SIZE_TEST];

    auto *search=new TYPE[SIZE_TEST];
    auto *chck=new TYPE[SIZE_TEST];

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
            EXPECT_EQ(2,chck[i])<<i<<key[i];
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
    auto *key=new TYPE[SIZE_TEST];
    auto *value=new TYPE[SIZE_TEST];
    auto *search=new TYPE[SIZE_TEST];
    auto *chck=new TYPE[SIZE_TEST];

    for(int i=0;i<SIZE_TEST;i++){
        search[i]=key[i]=rand()%1000000;
        value[i]=i;
        chck[i]=0;

    }


    hashAPI h(SIZE_TEST);
    h.hash_insert(key,value,SIZE_TEST);
    h.hash_search(search,chck,SIZE_TEST);

    for(int i=1;i<SIZE_TEST;i++){

        EXPECT_EQ(chck[i],value[i]);

    }

    memset(chck,0,SIZE_TEST*sizeof(int));
    h.hash_search(key,chck,SIZE_TEST);
    for(int i=0;i<SIZE_TEST;i++){
        EXPECT_EQ(chck[i],value[i])<<i<<" key:"<<key[i]<<"\n";
    }

    delete[] key;
    delete[] value;
    delete[] search;
    delete[] chck;

}

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////        test up down size                 //////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////

bool  test_size_resize(TYPE size) {


    printf("\tsize:%d\n", size);

    /// set a bigger pool
    size *= 2;

    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());
    auto *key = new TYPE[size];
    auto *value = new TYPE[size];
    auto *search = new TYPE[size];
    auto *chck = new TYPE[size];

    /// genrand_int32() may turn key<0 , fix to genrand_int31()
    GenerateUniqueRandomNumbers(key, size);

    for (TYPE i = 0; i < size; i++) {
        /// may cause inset error
//        key[i] = 2 * i + 2;
        /// ok
//        key[i] = i + 1;
        value[i] = 3 * i + 3 + 1;
        search[i] = 2 * i + 3 + 1;
        chck[i] = 0;
    }

    size /= 2;

    /// hash table init
    hashAPI h(size * 2);
    /// <<<insert>>> after insert ,the size 100/200
    h.hash_insert(key, value, size);
    /// <<<search>>>
    h.hash_search(key, chck, size);
    check_search_result(key,chck,value,size);

    checkCudaErrors(cudaGetLastError());

#if show_table_debug
    GPU_show_table();
#endif
    /// error , size/2!=0 , error occur in resize down : pos tid -> tid- new_size can fill
    /// error , size/2!=0 , error occur in resize down : pos tid -> tid- new_size can fill
    /// error , size/2!=0 , error occur in resize down : pos tid -> tid- new_size can fill
    /// 1. firest ,resize table choose false
    /// set table  & choose num ok
    /// pro: some error oucur when  densty ,  resize tag should be attached to it
    h.resize_low();


    /// after resize 100/175;
    h.hash_search(key, chck, size);
    check_search_result(key,chck,value,size);
#if show_table_debug
    GPU_show_table();
#endif
    h.resize_low();
    /// after resize 100/150;
    h.hash_search(key, chck, size);
    check_search_result(key,chck,value,size);
#if show_table_debug
    GPU_show_table();
#endif

//    /// set next insert data
//    TYPE* a=&key[size];
//    TYPE* b=&value[size];
//    checkCudaErrors(cudaGetLastError());
//
//    /// after insert 125/150
//    h.hash_insert(a,b,size/4);
//    checkCudaErrors(cudaGetLastError());
//    h.hash_search(key, chck, size);
//    check_search_result(key,chck,value,size);



    printf("¶¶¶test %d complete\n\n\n", size);
    checkCudaErrors(cudaGetLastError());
    delete[] key;
    delete[] value;
    delete[] search;
    delete[] chck;
    return true;
}

TEST(HashResizeTest,RandTestWithUpDown){
    EXPECT_TRUE(test_size_resize(1000000));
}

