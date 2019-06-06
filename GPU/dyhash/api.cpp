#include "api.h"
#include "mt19937ar.h"

bool isPrime(int num)
{
    if (num == 2 || num == 3) return true;
    if (num % 6 != 1 && num % 6 != 5) return false;
    for (int i = 5; i*i <= num; i += 6)
        if (num % i == 0 || num % (i+2) == 0) return false;
    return true;
}

TYPE nextPrime(TYPE n)
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
hashAPI::hashAPI(TYPE size) {
    batch_del_num_to_op=0;
    batch_search_num_to_op=0;
    num_size=0;
    d_key_pool=NULL;
    d_value_pool=NULL;
    d_check=NULL;
    batch_num_to_op=0;
    res=0;
    d_batch_iter_count=NULL;

    ///cnmem init
    memset(&device, 0, sizeof(device));
    device.size = (size_t)4*1024*1024*1024; /// more =(size_t) (0.95*props.totalGlobalMem);
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
    checkCudaErrors(cudaGetLastError());

    /// not ned=ed init, rehash will be init by every insert function
    cnmemMalloc((void **) &rehash, sizeof(int),0);

    /// adjust size to prime
    TYPE s = size / (TABLE_NUM * BUCKET_SIZE);
    s = nextPrime(s);

    /// 单表bucket数目
    TYPE s_bucket = (s & 1) ? s + 1 : s;
//    TYPE s_bucket = s;
    /// 单表可容纳元素数目
    TYPE s_size = s_bucket * BUCKET_SIZE;
    /// 总共可容纳元素数目
    table_size = s_size * TABLE_NUM;

    /// malloc tables
    h_table = (cuckoo *) malloc(sizeof(cuckoo));
#if api_info
    printf(">>>size: %d tablesize:%d s_bucket:%d \n"
           ">>>max thread info: block:%d  thread:%d\n",
           size, table_size, s_bucket, THREAD_NUM_k, BLOCK_NUM_k);
#endif
    for (TYPE i = 0; i < TABLE_NUM; i++) {
        /// alloc table
        cnmemMalloc((void **) &h_table->table[i], sizeof(TYPE) * s_size * 2,0);
        cudaMemset(h_table->table[i], 0, sizeof(TYPE) * s_size * 2);
        h_table->Lsize[i] = s_size;

        /// set hash fun
        h_table->hash_fun[i].x=(TYPE)(genrand_int32()%PRIME_uint);
        h_table->hash_fun[i].y=(TYPE)(genrand_int32()%PRIME_uint);
    }
    checkCudaErrors(cudaGetLastError());

    /// malloc lock   ： size of lock , for reisze it is 2 times than all_bucket
    cnmemMalloc((void **) &h_table->Lock, sizeof(TYPE) * table_size * 2 / 32,0);
    EXPECT_TRUE((h_table->Lock)!=NULL)<<"malloc error";
    cudaMemset(h_table->Lock, 0, sizeof(TYPE) * table_size / 32);
#if small_slot
    /// malloc small table
    cnmemMalloc((void **) &h_table->slot, sizeof(TYPE) * ((SLOT_SIZE) * BUCKET_SIZE * 2 ),0);
    EXPECT_TRUE((h_table->slot)!=NULL)<<"malloc error";
    cudaMemset(h_table->slot, 0, sizeof(TYPE) * ((SLOT_SIZE) * BUCKET_SIZE * 2 ));
#endif

    /// copy to gpu table
    gpu_lp_set_table(h_table);
    checkCudaErrors(cudaGetLastError());
}

hashAPI::~hashAPI() {
    cnmemFree(rehash,0);
    cnmemFree(h_table->Lock,0);
    for (int i = 0; i <4 ; ++i) {
        cnmemFree(h_table->table[i],0);
    }
    free(h_table);

    /// batch
    cnmemFree(d_key_pool,0);
    cnmemFree(d_value_pool,0);
    cnmemFree(d_check,0);
    cnmemFree(d_batch_iter_count,0);

    /// need?
    cnmemRelease();
}


void hashAPI::resize_up() {

//     printf("upsizeing\n");
    /// choose lowest
    TYPE num_table_to_resize = 0;
    for (TYPE i = 0; i < TABLE_NUM - 1; i++) {
        if (h_table->Lsize[i] == 2 * (h_table->Lsize[i + 1])) {
            num_table_to_resize = i + 1;
            break;
        }
    }



    #if size_info_of_resize_debug
        printf("up_size:%u/%u --> +%u\n",num_size,table_size,h_table->Lsize[3]);
    #endif
    table_size += h_table->Lsize[num_table_to_resize];
    int new_size = (h_table->Lsize[num_table_to_resize]) * 2;

    /// alloc
    bucket *new_table;
    cnmemMalloc((void **) &new_table, sizeof(TYPE) * new_size * 2, 0);
    checkCudaErrors(cudaGetLastError());
    cudaMemset((void *) new_table, 0, sizeof(TYPE) * new_size * 2);
    checkCudaErrors(cudaGetLastError());

    /// resize
    GPU_cuckoo_resize_up(num_table_to_resize, h_table->Lsize[num_table_to_resize], new_table, h_table);
}

void hashAPI::resize_low() {

//     printf("downsizeing\n");
    /// choose biggest
    TYPE num_table_to_resize = TABLE_NUM - 1;
    for (TYPE i = 0; i < TABLE_NUM - 1; i++) {
        if (h_table->Lsize[i] > h_table->Lsize[i + 1]) {
            num_table_to_resize = i;
            break;
        }
    }

    int new_size = ((get_table_bucket_size(num_table_to_resize) + 1) / 2) <<BUCKET_SIZE_P ;

    #if size_info_of_resize_debug
        printf("lowsize:%u/%u --> -%u\n",num_size,table_size,new_size);
    #endif
    table_size-=new_size;
    /// bucket to size : << BUCKET_SIZE_P
//    int new_size = ((get_table_bucket_size(num_table_to_resize) + 1) / 2) << BUCKET_SIZE_P;
    bucket *new_table;

    /// alloc
    cnmemMalloc((void **) &new_table, sizeof(TYPE) * new_size * 2, 0);
    checkCudaErrors(cudaGetLastError());
    cudaMemset((void *) new_table, 0, sizeof(TYPE) * new_size * 2);
    checkCudaErrors(cudaGetLastError());

    /// resize
    GPU_cuckoo_resize_down(num_table_to_resize, h_table->Lsize[num_table_to_resize], new_table, h_table);
}

void hashAPI::api_resize(TYPE after_insert_size){
    if(after_insert_size>Upper_Bound(table_size)){
        resize_up();
    }else if(after_insert_size <Lower_Bound(table_size)){
        resize_low();
    }
}


void hashAPI::hash_insert(TYPE *key, TYPE *value,TYPE size){
#ifndef no_rehash
    /// check size , need resize?
    TYPE after_insert_size=num_size+size;
    if(after_insert_size>Upper_Bound(table_size)){
        api_resize(after_insert_size);
        /// recall to resize multiple times
        hash_search_batch();
        return;
    }
#endif
    int* d_iter_count;

#if ITERAOTOR_COUNT_FLAG
    const int ssize=size;
    cnmemMalloc((void**)&d_iter_count, sizeof(int)*ssize,0);
    cudaMemset(d_iter_count,0, sizeof(int)*ssize);
#endif

    /// alloc
    TYPE* d_keys;
    cnmemMalloc((void**)&d_keys, sizeof(TYPE)*size,0);
    cudaMemcpy(d_keys, key, sizeof(TYPE)*size, cudaMemcpyHostToDevice);
    TYPE* d_value;
    cnmemMalloc((void**)&d_value, sizeof(TYPE)*size,0);
    cudaMemcpy(d_value, value, sizeof(TYPE)*size, cudaMemcpyHostToDevice);
    EXPECT_TRUE(d_keys!=NULL)<<"malloc error";
    EXPECT_TRUE(d_value!=NULL)<<"malloc error";
    num_size+=size;

    /// insert
    gpu_lp_insert(d_keys,d_value,size,rehash,d_iter_count);

//    gpu_computer_table();

#if ITERAOTOR_COUNT_FLAG

    int *iter_count=new int[ssize];
    cudaMemcpy(iter_count, d_iter_count, sizeof(int)*ssize, cudaMemcpyDeviceToHost);
    FILE *fid;
    char* const filename="./ans_confi.dat";
    if((fid=fopen(filename,"wb+"))==NULL)
    {
        printf("Can not open output file.\n");
        exit(1);
    }

    fwrite(iter_count,sizeof(int),ssize,fid);
    int maxi=0,sumi=0;
    int up_10=0;
    for(int i=0;i<ssize;i++){
        sumi+=iter_count[i];
        maxi=maxi>iter_count[i]?maxi:iter_count[i];
        if(iter_count[i] > 10 ) up_10 ++;
//        if(i<100) printf("iter%d:%d ",i,iter_count[i]);
    }
    printf("iterator counting :datasize %d, max %d, everage:%lf, >10: %d \n",ssize,maxi,sumi*1.0/ssize,up_10);
    delete[] iter_count;
    cnmemFree(d_iter_count,0);
#endif
    /// check rehash flag
    int *a=(int*)malloc(sizeof(int));
    *a=2;
    cudaMemcpy(a, rehash, sizeof(int), cudaMemcpyDeviceToHost);
#ifndef no_rehash
    printf("check insert error,reszie:%d\n",*a);



    /// reinsert
    if(*a != 0){
        printf("insert error,reszie\n");
        *a=0;
        resize_up();
        gpu_lp_insert(d_keys,d_value,size,rehash);
        cudaMemcpy(a, rehash, sizeof(int), cudaMemcpyDeviceToHost);

    }
    cudaDeviceSynchronize();
#endif
    checkCudaErrors(cudaGetLastError());
}

void hashAPI::hash_search(TYPE *key, TYPE *ans,TYPE size){

    /// alloc
    TYPE* d_keys;
    cnmemMalloc((void**)&d_keys, sizeof(TYPE)*size,0);
    cudaMemcpy(d_keys, key, sizeof(TYPE)*size, cudaMemcpyHostToDevice);
    TYPE* d_value;
    cnmemMalloc((void**)&d_value, sizeof(TYPE)*size,0);
    cudaMemset(d_value,0, sizeof(TYPE)*size);
    checkCudaErrors(cudaGetLastError());
    EXPECT_TRUE(d_keys!=NULL)<<"malloc error";
    EXPECT_TRUE(d_value!=NULL)<<"malloc error";

    //kernel
    gpu_lp_search(d_keys,d_value,size);

    cudaMemcpy(ans, d_value, sizeof(TYPE)*size, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}



void hashAPI::hash_delete(TYPE *key, TYPE *ans,TYPE size) {
    /// alloc
    TYPE *d_keys;
    cnmemMalloc((void **) &d_keys, sizeof(TYPE) * size, 0);
    cudaMemcpy(d_keys, key, sizeof(TYPE) * size, cudaMemcpyHostToDevice);
    TYPE *d_value;
    cnmemMalloc((void **) &d_value, sizeof(TYPE) * size, 0);
    cudaMemset(d_value, 0, sizeof(TYPE) * size);

    //kernel
    gpu_lp_delete(d_keys, d_value, size);

    cudaMemcpy(ans, d_value, sizeof(TYPE) * size, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}





bool hashAPI::set_data_to_gpu(TYPE *key,TYPE* value,TYPE size){
    EXPECT_LE(size, max_input_size) << "set data error size of kv out of 2G";

    /// malloc kv pool
    cnmemMalloc((void **)&d_key_pool, size * sizeof(TYPE), 0);
    checkCudaErrors(cudaGetLastError());
    cnmemMalloc((void **)&d_value_pool, size * sizeof(TYPE), 0);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    EXPECT_TRUE(d_key_pool!=NULL)<<"malloc error";
    EXPECT_TRUE(d_value_pool!=NULL)<<"malloc error";

    /// copy data to pool
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(d_key_pool, key, size * sizeof(TYPE), cudaMemcpyHostToDevice));
    cudaMemcpy(d_value_pool, value, size * sizeof(TYPE), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
    /// set batch pointer
    for (int i = 0; i * BATCH_SIZE < size && i < BATCH_NUM; i++) {
        d_key_batch[i] = d_key_pool + BATCH_SIZE * i;
        d_value_batch[i] = d_value_pool + BATCH_SIZE * i;
    }

    #if ITERAOTOR_COUNT_FLAG
        const int ssize=BATCH_SIZE;
        cnmemMalloc((void**)&d_batch_iter_count, sizeof(int)*ssize,0);
        cudaMemset(d_batch_iter_count,0, sizeof(int)*ssize);
    #endif

    /// small check array for insert del
    cnmemMalloc((void **)&d_check, BATCH_SIZE * sizeof(TYPE), 0);
    cudaMemset(d_check,0,BATCH_SIZE * sizeof(TYPE));
    EXPECT_TRUE(d_check!=NULL)<<"malloc error";

    return true;
}

void hashAPI::batch_reset(){
    batch_num_to_op=0;
    batch_del_num_to_op=0;
    batch_search_num_to_op=0;
    res=1;
}

void hashAPI::hash_insert_batch() {
    if(batch_num_to_op >= BATCH_NUM ){
        EXPECT_LT(batch_num_to_op,BATCH_NUM);
        return ;
    }

    /// check size ,need resize ?
    TYPE after_insert_size=num_size+BATCH_SIZE;
    if(after_insert_size>Upper_Bound(table_size) ){
    #if size_info_of_resize_debug
            printf("resize:%u/%u -(%u)-> +%u\n",num_size,table_size,BUCKET_SIZE,h_table->Lsize[3]);
    #endif
        api_resize(after_insert_size);
        /// recall to resize multiple times
        hash_insert_batch();
        return;
    }
//     printf("%d ",batch_num_to_op);
    num_size+=BATCH_SIZE;


/// inserting
#if ITERAOTOR_COUNT_FLAG
    const int ssize=BATCH_SIZE;
    cudaMemset(d_batch_iter_count,0, sizeof(int)*ssize);
    gpu_lp_insert(d_key_batch[batch_num_to_op],
                  d_value_batch[batch_num_to_op],
                  BATCH_SIZE,
                  rehash,
                  d_batch_iter_count);

    int *iter_count=new int[ssize];
    cudaMemcpy(iter_count, d_batch_iter_count, sizeof(int)*ssize, cudaMemcpyDeviceToHost);
//    FILE *fid;
//    char* const filename="./ans_confi.dat";
//    if((fid=fopen(filename,"wb+"))==NULL)
//    {
//        printf("Can not open output file.\n");
//        exit(1);
//    }
//
//    fwrite(iter_count,sizeof(int),ssize,fid);
    int maxi=0,sumi=0;
    for(int i=0;i<ssize;i++){
        sumi+=iter_count[i];
        maxi=maxi>iter_count[i]?maxi:iter_count[i];
    }
    printf("iterator counting :in total %d data inserting, max %d,everage:%lf\n", BATCH_SIZE, maxi, sumi*1.0/ssize);
    delete[] iter_count;
#else
    gpu_lp_insert(d_key_batch[batch_num_to_op],
                  d_value_batch[batch_num_to_op],
                  BATCH_SIZE,
                  rehash);
#endif

    cudaDeviceSynchronize();
//     if(res==1)
    gpu_computer_table();
    cudaDeviceSynchronize();
//     printf("\n%d/%d\n",num_size,table_size);
//
//    /// check
//    int *a=(int*)malloc(sizeof(int));
//    *a=2;
//    cudaMemcpy(a, rehash, sizeof(int), cudaMemcpyDeviceToHost);
//    /// reinsert
//    if(*a != 0){
//        printf("batch insert error,reszie\n");
//        *a=0;
//        resize_up();
//        gpu_lp_insert(d_key_batch[batch_num_to_op],
//                      d_value_batch[batch_num_to_op],
//                      BATCH_SIZE,
//                      rehash);
//    }

    batch_num_to_op++;

}


void hashAPI::hash_search_batch() {
    if(batch_search_num_to_op >= BATCH_NUM ){
        EXPECT_LT(batch_search_num_to_op,BATCH_NUM);
        return ;
    }
//     printf("%d ",batch_search_num_to_op);

//    GpuTimer timer;
//    timer.Start();

    gpu_lp_search(d_key_batch[batch_search_num_to_op],
                  d_check,
                  BATCH_SIZE);
    gpu_computer_table();

//    timer.Stop();
//    double diff = timer.Elapsed() * 1000000;
//    printf("api<<<search>>>：the time is %.2lf us, ( %.2f Mops)s\n",
//           (double)diff, (double)(BATCH_SIZE) / diff);

    /// 默认不copy back
    ///cudaMemcpy(???,d_check,sizeof(TYPE)*BATCH_SIZE,cudaMemcpyDeviceToHost);

    batch_search_num_to_op  ++;
}


void hashAPI::hash_delete_batch() {
    if(batch_del_num_to_op >= BATCH_NUM ){
        EXPECT_LT(batch_del_num_to_op,BATCH_NUM);
        return ;
    }
//     printf("%d ",batch_del_num_to_op);

//    GpuTimer timer;
//    timer.Start();

    gpu_lp_delete(d_key_batch[batch_del_num_to_op],
                  d_check,
                  BATCH_SIZE);
    cudaDeviceSynchronize();
//     if(res==1)
    gpu_computer_table();

    cudaDeviceSynchronize();
//     printf("\n%d/%d\n",num_size,table_size);

//    timer.Stop();
//    double diff = timer.Elapsed() * 1000000;
//    printf("api<<<delete>>>：the time is %.2lf us, ( %.2f Mops)s\n",
//           (double)diff, (double)(BATCH_SIZE) / diff);

    /// 默认不copy back
    ///cudaMemcpy(???,d_check,sizeof(TYPE)*BATCH_SIZE,cudaMemcpyDeviceToHost);

    num_size -= BATCH_SIZE;

    batch_del_num_to_op++;

    /// check size ,need resize ?

    if(num_size<Lower_Bound(table_size) && table_size>(TABLE_NUM*1024*1024*1.2)){

#if size_info_of_resize_debug
        printf("resize:%u/%u -(%u)-> -%u\n",num_size,table_size,BATCH_SIZE,h_table->Lsize[3]);
#endif

//        if(res==1) GPU_show_table();
        api_resize(num_size);
        return;
    }
}


