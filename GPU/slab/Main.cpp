//
// Created by jing on 2018/7/1.
//

#include <cstring>
#include <cstdlib>
//#include "api.h"
#include "include/dy_hash.h"
//#include "libgpuhash.h"
//#include "Cpu_api.cpp"
#include "slab_hash.h"

using namespace std;
TYPE *key;
#define NUM_DATA 100000000




unsigned int* DataRead_CMode(char* filename)
{
//     printf("info:filename:%s\n",filename);
    int size=NUM_DATA;
    if(strcmp(filename,"/home/udms/ly/finally-test/data/twitter.dat")==0)
        size=size/2;
    if(strcmp(filename,"/home/udms/ly/finally-test/data/tpc-h.dat")==0)
        size=size/2;
    if(strcmp(filename,"/home/udms/ly/data/real_2018/l32.dat")==0)
        size=size/10;

    FILE *fid;
    fid = fopen(filename, "rb");
    unsigned int *pos;
    pos = (unsigned int *)malloc(sizeof(unsigned int)*size);//申请内存空间，大小为n个int长度
    int i;
    if (fid == NULL)
    {
        printf("the data file is unavailable.\n");
        exit(1);
        return pos;
    }
    fread(pos, sizeof(unsigned int), size, fid);
    fclose(fid);
    return pos;
}





int main(int argc, char** argv) {

//    char* name;
//    if (argc >=2 )
//    {
////         num_more=atoi(argv[1]);
////         no_data=atoi(argv[2]);
//        name=argv[1];
//
//    }else{
////        printf("run using : ./linear file_neme");
//    }
//    //test_size_new_api();
    printf("main \n");
    simple_gpu_test();

    return 0;
}



