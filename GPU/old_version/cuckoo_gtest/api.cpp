//
// Created by jing on 2018/7/1.
//

#include "api.h"
#include "mt19937ar.h"
double mytime=0;

bool isPrime(int num)
{
    if (num == 2 || num == 3) return true;
    if (num % 6 != 1 && num % 6 != 5) return false;
    for (int i = 5; i*i <= num; i += 6)
        if (num % i == 0 || num % (i+2) == 0) return false;
    return true;
}

TTT nextPrime(TTT n)
{
    bool state=isPrime(n);
    while(!state)
        state=isPrime(++n);
#if size_debug
    printf("find prime %d\n",n);
#endif
    return n;
}


/// size : the key  in the table
hashAPI::hashAPI(TTT size) {
    num_size=0;
    ///cnmem init
//    memset(&device, 0, sizeof(device));
//    device.size = (size_t)4*1024*1024*1024; /// more =(size_t) (0.95*props.totalGlobalMem);
//    //device.numStreams=4;
//    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);



    checkCudaErrors(cudaGetLastError());

    /// not ned=ed init, rehash will be init by every insert function
    cnmemMalloc((void **) &rehash, sizeof(TTT),0);



    /// adjust size to prime
    TTT s = size / (TABLE_NUM * BUCKET_SIZE);

    //s = nextPrime(s > 3 ? s : 3);
    s = nextPrime(s+1);

//    TTT s_bucket  =  s;
    TTT s_bucket = (s & 1) ? s + 1 : s; // 当需要down size 的时候,如果s为奇数则加以1
    TTT s_size = s_bucket * BUCKET_SIZE;

    table_size = s_size * TABLE_NUM;

    /// malloc table
    checkCudaErrors(cudaGetLastError());
    h_table = (cuckoo *) malloc(sizeof(cuckoo));
    printf(">>>size: %d tablesize:%d s_bucket:%d \n>>>block:%d  thread:%d\n", size, table_size, s_bucket, NUM_THREADS,
           NUM_BLOCK);
    for (TTT i = 0; i < TABLE_NUM; i++) {
        /// alloc table
        cnmemMalloc((void **) &h_table->table[i], sizeof(TYPE) * s_size * 2,0);
        cudaMemset(h_table->table[i], 0, sizeof(TYPE) * s_size * 2);
        h_table->Lsize[i] = s_size;

        /// set hash fun
        h_table->hash_fun[i].x=(TTT)(genrand_int32()%_PRIME);
        h_table->hash_fun[i].y=(TTT)(genrand_int32()%_PRIME);
    }
    checkCudaErrors(cudaGetLastError());
    /// malloc lock   TODO： size of lock , for reisze it is 2 times than all_bucket
    cnmemMalloc((void **) &h_table->Lock, sizeof(TTT) * table_size * 2 / 32,0);
    cudaMemset(h_table->Lock, 0, sizeof(TTT) * table_size / 32);

    /// copy to gpu table
    //printf("starting set table\n");
    gpu_lp_set_table(h_table);


//    cnmemMalloc((void**)&hash_table,sizeof(cuckoo),0);
//    cudaMemcpy(hash_table,h_table,sizeof(cuckoo),cudaMemcpyHostToDevice);

    checkCudaErrors(cudaGetLastError());
    //printf("init ok\n");
}

/// size : the key  in the table
hashAPI::hashAPI(TTT size,cudaStream_t streams[],int NUM_STREAM) {
    num_size=0;
    ///cnmem init
    memset(&device, 0, sizeof(device));
    device.size = (size_t)4*1024*1024*1024; /// more =(size_t) (0.95*props.totalGlobalMem);
    device.numStreams=NUM_STREAM;
    device.streams=streams;
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);



    checkCudaErrors(cudaGetLastError());

    /// not ned=ed init, rehash will be init by every insert function
    cnmemMalloc((void **) &rehash, sizeof(TTT),0);
    checkCudaErrors(cudaGetLastError());


    /// adjust size to prime
    TTT s = size / (TABLE_NUM * BUCKET_SIZE);

    //s = nextPrime(s > 3 ? s : 3);
    s = nextPrime(s+1);

//    TTT s_bucket  =  s;
    TTT s_bucket = (s & 1) ? s + 1 : s; // 当需要down size 的时候,如果s为奇数则加以1
    TTT s_size = s_bucket * BUCKET_SIZE;

    table_size = s_size * TABLE_NUM;

    /// malloc table
    checkCudaErrors(cudaGetLastError());
    h_table = (cuckoo *) malloc(sizeof(cuckoo));
    printf(">>>size: %d tablesize:%d s_bucket:%d \n>>>block:%d  thread:%d\n", size, table_size, s_bucket, NUM_THREADS,
           NUM_BLOCK);
    for (TTT i = 0; i < TABLE_NUM; i++) {
        /// alloc table
        cnmemMalloc((void **) &h_table->table[i], sizeof(TYPE) * s_size * 2,0);
        checkCudaErrors(cudaGetLastError());
        cudaMemset(h_table->table[i], 0, sizeof(TYPE) * s_size * 2);
        h_table->Lsize[i] = s_size;

        /// set hash fun
        h_table->hash_fun[i].x=(TTT)(genrand_int32()%_PRIME);
        h_table->hash_fun[i].y=(TTT)(genrand_int32()%_PRIME);
    }
    checkCudaErrors(cudaGetLastError());
    /// malloc lock   TODO： size of lock , for reisze it is 2 times than all_bucket
    cnmemMalloc((void **) &h_table->Lock, sizeof(TTT) * table_size * 2 / 32,0);
    cudaMemset(h_table->Lock, 0, sizeof(TTT) * table_size / 32);

    /// copy to gpu table
    //printf("starting set table\n");
    gpu_lp_set_table(h_table);


//    cnmemMalloc((void**)&hash_table,sizeof(cuckoo),0);
//    cudaMemcpy(hash_table,h_table,sizeof(cuckoo),cudaMemcpyHostToDevice);

    checkCudaErrors(cudaGetLastError());
    //printf("init ok\n");
}


void hashAPI::resize_up(cudaStream_t stream){
    /// choose lowest

    TTT num_table_to_resize=0;
    for(TTT i=0;i<TABLE_NUM-1;i++){
        if(h_table->Lsize[i]==2*(h_table->Lsize[i+1])){
            num_table_to_resize=i+1;
            break;
        }
    }
    table_size+=h_table->Lsize[num_table_to_resize];

    printf("up_size:%u/%u --> +%u\n",num_size,table_size,h_table->Lsize[3]);


    int new_size=(h_table->Lsize[num_table_to_resize])*2;
    /// alloc

    bucket* new_table;
    cnmemMalloc((void**)&new_table,sizeof(TTT)*new_size*2,stream);
//    checkCudaErrors(cudaGetLastError());
    cudaMemsetAsync((void*)new_table,0,sizeof(TTT)*new_size*2,stream);
//    checkCudaErrors(cudaGetLastError());


    /// resize
    GPU_cuckoo_resize_up(num_table_to_resize,h_table->Lsize[num_table_to_resize],new_table,h_table,stream);

}

void hashAPI::resize_low(cudaStream_t stream){
    printf("need decrease size\n ");
    // exit(1);
    /// choose biggest
    TTT num_table_to_resize=TABLE_NUM-1;
    for(TTT i=num_table_to_resize-1;i>0;i--){
        if(h_table->Lsize[i]*2==h_table->Lsize[i-1]){
            num_table_to_resize=i;
            break;
        }
    }

    printf("lowsize:%u/%u --> -%u\n",num_size,table_size,h_table->Lsize[3]/2);

    int new_size=(h_table->Lsize[num_table_to_resize])/2;
    bucket* new_table;
    cnmemMalloc((void**)&new_table,sizeof(TTT)*new_size*2,stream);

    cudaMemsetAsync((void*)new_table,0,sizeof(TTT)*new_size*2,stream);

    /// resize
    GPU_cuckoo_resize_down(num_table_to_resize,h_table->Lsize[num_table_to_resize],new_table,h_table,stream);
}

void hashAPI::api_resize(TTT after_insert_size,cudaStream_t stream){
    /// TODO: contain mutly resize if insert is much
    if(after_insert_size>SIZE_UP(table_size)){
        resize_up(stream);
    }else if(after_insert_size <SIZE_LOW(table_size)){
        resize_low(stream);
    }
}

void hashAPI::hash_insert(TTT *key, TTT *value,TTT size,cudaStream_t stream) {

    checkCudaErrors(cudaGetLastError());

    TTT after_insert_size=num_size+size;
//
////    /// need resize?
////    if(after_insert_size>SIZE_UP(table_size) || after_insert_size < SIZE_LOW(table_size)){
////        printf("resize\n");
////        api_resize(after_insert_size);
////        /// recall to resize multiple times
////        hash_insert(key,value,size);
////        return;
////    }
    /// need resize?
    if(after_insert_size>SIZE_UP(table_size) ){
        printf("resize:%u/%u -(%u)-> +%u\n",num_size,table_size,size,h_table->Lsize[3]);
        api_resize(after_insert_size,stream);
        /// recall to resize multiple times
        hash_insert(key,value,size,stream);
        return;
    }

    num_size+=size;




}



void hashAPI::hash_search(TTT *key, TTT *value,TTT size,cudaStream_t stream){
//    checkCudaErrors(cudaGetLastError());
    TTT* d_keys;
    cnmemMalloc((void**)&d_keys, sizeof(TTT)*size,stream);
    cudaMemcpyAsync(d_keys, key, sizeof(TTT)*size, cudaMemcpyHostToDevice,stream);

    TTT* d_value;
    cnmemMalloc((void**)&d_value, sizeof(TTT)*size,stream);
    cudaMemcpyAsync(d_value, value, sizeof(TTT)*size, cudaMemcpyHostToDevice,stream);

    //kernel
    gpu_lp_search(d_keys,d_value,size,stream);
    
    
    cudaMemcpyAsync(value, d_value, sizeof(TTT)*size, cudaMemcpyDeviceToHost,stream);
//    checkCudaErrors(cudaGetLastError());
}

hashAPI::~hashAPI() {

    cnmemFree(rehash,0);
    cnmemFree(h_table->Lock,0);
    for (int i = 0; i <4 ; ++i) {
        cnmemFree(h_table->table[i],0);
    }
    free(h_table);

    /// need?
    cnmemRelease();
}



//void hashAPI::hash_delete(TTT *key,TTT *ans,TTT size) {
//    for(TTT i=0;i<size;i++){
//        ans[i]=table->remove(key[i]);
//    }
//}

