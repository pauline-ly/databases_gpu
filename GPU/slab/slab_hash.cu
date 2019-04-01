//
// Created by jing on 2019-03-17.
//

//#include <cstdint>
#include "cuda_runtime.h"
#include "slab_hash.h"
#include "include/dy_hash.h"
#include "gputimer.h"
//#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#define ulong unsigned long long int
#define make_ulong(k,v) ((((ulong)v)<<32)+k)
#define getk_from_ulong(kv) ((unsigned)((kv)&0xffffffff))
#define getv_from_ulong(kv) ((unsigned)((kv)>>32))
#define gtid (blockIdx.x * blockDim.x + threadIdx.x)

#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

//#define NUM_DATA 100000000
//#define NUM_DATA 500
//#define NUM_DATA 500

#define DE_ACTIVE 0
#define ACTIVE 1
#define TYPE unsigned int




class Node{
public:
    unsigned _entry[32];

    //__device__ Node* next(){ return (Node*)&_entry[30]; }
    __device__ ulong getkv(int pos){
        return ((ulong*)_entry)[pos];
    }
    __device__ ulong getnext(){
        return getkv(15);
    }

    __device__ bool ptrCAS(ulong old_val,ulong node){
        long old=atomicCAS((ulong*)&_entry[30],old_val,node);
        return old==old_val;
    }

    __device__ bool set_kv_on_pos(int k,int v,int pos,ulong oldkv){
        ulong newkv=make_ulong(k,v);
        ulong old=atomicCAS((ulong*)&_entry[pos*2],oldkv,newkv);
#if CAS_debug
        printf(">>>CAS:id:%d,kv %d,%d pos %d \n>>>CAS:oldkv %lld,%lld CASold: %lld,%lld kv: %lld, %lld pos: %d,%d\n",
                threadIdx.x,k,v,pos,
                oldkv>>32,oldkv& 0xffffffff,old>>32,old& 0xffffffff,newkv>>32,newkv& 0xffffffff,
                _entry[pos*2],_entry[pos*2+1]);
        show("in CAS",k);
#endif

        return old==oldkv;
    }

    __device__ bool set_ptr(Node* ptr){
        return ptrCAS(getnext(),(ulong)ptr);
    }

    __device__ void show(const char* src,int pos=0){
        printf("%s: val:%d tid:%d loc:%lx",src,pos,gtid,&_entry[0]);
#if kv_in_Node
        printf("\n");
        for(int i=0;i<30;i++){
            printf("%d ",_entry[i]);
        }
#endif
        printf("next: %lx\n",getnext());
    }
};

class MemPool{
public:
    Node* data[MEM_POOL_NUM];
    unsigned int pos[MEM_POOL_NUM];
    MemPool(){}
    MemPool(int s){
        printf("memory pool %d initing\n",s);
        Node* ptr;
        cudaMalloc((void**)&ptr,sizeof(Node)*MEM_POOL_NUM*NODE_OF_POOL);
        cudaMemset(ptr,0, sizeof(Node)*MEM_POOL_NUM*NODE_OF_POOL);
        for(int i=0;i<MEM_POOL_NUM;i++){
            data[i]=ptr+i*NODE_OF_POOL;
        }
        memset(pos,0, sizeof(int)*MEM_POOL_NUM);
        printf("memory pool %d inited\n",s);
    }

    __device__ void show_pool(){
        for(int i=0;i<NODE_OF_POOL;i++){
            printf("%d %d %llx\n",0,i,&data[0][i]);
            data[0][i].show("mempool show",i);
        }
    }

    __device__ Node* get_Node(){
#if show_memory
        show_pool();
#endif
        int tid = gtid>>5;
        int old=atomicInc(&pos[tid],NODE_OF_POOL+10);
        if(old>=NODE_OF_POOL)
            return NULL;
#if check_memory_alloc
        printf("get one:%d loc:%llx\n",old,&data[tid][old]);
#endif
        return &data[tid][old];
    }
    __device__ void free_Node(Node* n){
        int tid = gtid>>5;
        int unuse_pos=n-data[tid];
        assert(unuse_pos>=0);
        atomicCAS(&pos[tid],unuse_pos+1,unuse_pos);
    }
};

__device__ MemPool* mem_pool;

void printinit(){
    static int a=1;
    if(a==1){
        printf("slab list inited\n");
        a=0;
    }
}

class SlabList{
public:
    Node* first;
    static int inited;

    SlabList(){
        printinit();
        cudaMalloc((void**)&first,sizeof(Node));
        cudaMemset(first,0, sizeof(Node));
    }

    __device__ bool alloc_new_node(Node* tmp) {

        int id_in_warp = threadIdx.x & 0x1f;
        if (id_in_warp == 0) {
            Node *nxt = mem_pool->get_Node();
            if (nxt == NULL) {
                printf("!!!! error , pool used over\n");
                printf("!!!! error , pool used over tid:%d ,nxt:%lld \n", gtid,nxt);
                printf("!!!! error , pool used over\n");
                return false;
            }
            if (!tmp->set_ptr(nxt)) {
                mem_pool->free_Node(nxt);
            }

        }
        return true;
    }

    __device__ bool check_target_and_CAS(int k,int v,int &is_active,int target,Node* tmp){
        int id_in_warp=threadIdx.x & 0x1f;
        int ballot=__ballot(tmp->_entry[id_in_warp]==target);
        ballot&=MASK_CHECK_kv;
        //  只要还发现 target
        while(ballot!=0){
            int chosen_simd=__ffs(ballot)-1;
            ulong oldkv=tmp->getkv(chosen_simd/2);
            if(getk_from_ulong(oldkv)==target) {
                if(id_in_warp==chosen_simd) {
                    if (tmp->set_kv_on_pos(k, v, chosen_simd / 2, oldkv)) {
                        is_active = DE_ACTIVE;
                    }
                }
                #if insert_debug
                    if(id_in_warp==0){
                        printf("kv %d,%d target %d isa:%d,ballot:%llx\n",k,v,target,is_active,ballot);
                        tmp->show("check in insert",k);
                    }
                #endif
                is_active = __shfl(is_active, chosen_simd, BUCKET_SIZE);
                if (is_active == DE_ACTIVE) return true;
            }
            ballot=__ballot(tmp->_entry[id_in_warp]==target);
            ballot&=MASK_CHECK_kv;
        }
        return false;
    }

    // after ballot , warp insert
    __device__ bool warp_insert(int k,int v){
        int id_in_warp=threadIdx.x & 0x1f;
        Node* tmp=first;
        assert(tmp!=NULL);

        int is_active=ACTIVE;

        while(tmp!=NULL){
#if insert_debug
            if(id_in_warp==0)  tmp->show("check in main while in insert",k);
#endif

            // check exist
//            if( check_target_and_CAS(k,v,is_active,k,tmp) )
//                return true;

            // check null
            if( check_target_and_CAS(k,v,is_active,0,tmp) )
                return true;

            if(NULL == (Node*)(tmp->getnext())){
                alloc_new_node(tmp);
            }
            tmp=(Node*)(tmp->getnext());
        }

        if(id_in_warp==0)
            printf("need resize\n");
        return false;
    }

    __device__ bool check_target_and_return_value(int k,Node* tmp,int write_pos,TYPE* value){
        int id_in_warp=threadIdx.x & 0x1f;
        int ballot=__ballot(tmp->_entry[id_in_warp]==k);
        ballot&=MASK_CHECK_kv;
        if(ballot!=0){
            int chosen_simd=__ffs(ballot)-1;
            ulong oldkv=tmp->getkv(chosen_simd/2);
            if(id_in_warp==chosen_simd) {
                value[write_pos]=getv_from_ulong(oldkv);
            }
            return true;
        }
        return false;
    }


    __device__ bool warp_search(int k,int write_pos,TYPE* value){
        Node* tmp=first;
        assert(tmp!=NULL);
        while(tmp!=NULL) {
            if(check_target_and_return_value(k,tmp,write_pos,value))
                return true;
            tmp=(Node*)(tmp->getnext());
        }
        return false;
    }

    __device__ void show_list(int n){
        Node* tmp=first;
        for(int i=0;i<n && tmp!=NULL;i++){
            tmp->show("list show",i);
            tmp=(Node*)(tmp->getnext());
        }
    }
}; // class slablist


struct SlabHash{
public:
    SlabList slist[TABLE_NUM];
};


__global__ void
kernel_find_test(SlabHash* ht,TYPE *key,TYPE* value,int size){
    int tid = gtid;
    int id_of_warp = (tid) >> 5;
    int step = (gridDim.x * blockDim.x) >> 5;

    for (; id_of_warp < size; id_of_warp += step) {
        int k=key[id_of_warp];
        ht->slist[k % TABLE_NUM].warp_search(k,id_of_warp, value );
    }
#if show_list_in_kernel
    __syncthreads();
    if (gtid == 0) {
        for (int i = 0; i < TABLE_NUM; i++) {
            printf("listnum:%d\n", i);
            ht->slist[i].show_list(300);
        }
    }
#endif
}

__global__ void
kernel_test(SlabHash* ht,TYPE *key,TYPE* value,int size) {
    int tid = gtid;
    int id_of_warp = (tid) >> 5;
    int step = (gridDim.x * blockDim.x) >> 5;

    for (; id_of_warp < size; id_of_warp += step) {
        int k=key[id_of_warp];
        int v=value[id_of_warp];
        ht->slist[k % TABLE_NUM].warp_insert(k, v);
    }
#if show_list_in_kernel
    __syncthreads();
    if (gtid == 0) {
        for (int i = 0; i < TABLE_NUM; i++) {
            printf("listnum:%d\n", i);
            ht->slist[i].show_list(300);
        }
    }
#endif
}

__global__ void
set_mempool(MemPool* pool){
    mem_pool=pool;
}

using namespace std;


unsigned int* Read_Data(char* filename)
{
//     printf("info:filename:%s\n",filename);
    int size=NUM_DATA;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/twitter.dat")==0)
        size=size/2;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/tpc-h.dat")==0)
        size=size/2;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/real_2018/l32.dat")==0)
        size=size/10;

    FILE *fid;
    fid = fopen(filename, "rb");
    unsigned int *pos;
    pos = (unsigned int *)malloc(sizeof(unsigned int)*size);//申请内存空间，大小为n个int长度

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


bool init_kv(TYPE *key,TYPE *value,char *filename,int size){

//    TYPE *k;
//    k=Read_Data(filename);

//     GenerateUniqueRandomNumbers(key, pool_size);

    for (int i = 0; i < size; i++) {
//        key[i]=k[i];
        key[i] =(TYPE) i;
//        value[i] =(TYPE) 3 * i + 3 + 1;
        value[i] =(TYPE) i;
//        chck[i]  = 0;
    }
//    for (int i = 0; i < size; i++) {
//        printf("k:%d   v:%d\n",key[i],value[i]);
////        chck[i]  = 0;
//    }
    return true;
}

using namespace thrust;
void check_result(TYPE* key,TYPE* value,TYPE* check,int size){
    int tmp=0;
    for(int i=0;i<size;i++){
        if(value[i]!=check[i]){
            if(tmp++<20) printf("false: %d :k:%d v:%d find:%d\n",i,key[i],value[i],check[i]);
        }
    }
    printf("check result:%d insert ,find %d ,not find %d (%d)\n",size,size-tmp,tmp,tmp*1.0/size);
}

void simple_gpu_test(char *filename){

    int size=NUM_DATA;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/twitter.dat")==0)
        size=size/2;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/tpc-h.dat")==0)
        size=size/2;
    if(strcmp(filename,"/home/udms/ly/GPU_Hash/finally-test/data/real_2018/l32.dat")==0)
        size=size/10;


    // alloc data
    thrust::host_vector<TYPE> key(size+1);
    thrust::host_vector<TYPE> value(size+1);


    //init data
    init_kv(RAW_PTR(key),RAW_PTR(value),filename,size);
    printf("=========init over==========\n");

    // copy to  gpu
    thrust::device_vector<TYPE> dkey(key);
    thrust::device_vector<TYPE> dvalue(value);
    thrust::device_vector<TYPE> dcheck(size+1);

    // 初始化hash表
    SlabHash hash;
    SlabHash* dhash;
    cudaMalloc((void**)&dhash,sizeof(SlabHash));
    cudaMemcpy(dhash,&hash, sizeof(SlabHash),cudaMemcpyHostToDevice);
    // 初始化 pool
    MemPool h_pool(0);
    MemPool *d_pool;
    cudaMalloc((void**)&d_pool,sizeof(MemPool));
    cudaMemcpy(d_pool,&h_pool, sizeof(MemPool),cudaMemcpyHostToDevice);
    set_mempool<<<1,1>>>(d_pool);


    // 插入数据
    GpuTimer timer;
    timer.Start();
    kernel_test<<<512,512>>>(dhash,RAW_PTR(dkey),RAW_PTR(dvalue),size);
    timer.Stop();
    double  diff = timer.Elapsed()*1000000;
    printf("<<<time>>> %.2lf ( %.2f)\n",
           (double) diff, (double) (size) / diff);
    cudaDeviceSynchronize();
    cudaGetLastError();


    // find 数据
    timer.Start();
    kernel_find_test<<<512,512>>>(dhash,RAW_PTR(dkey),RAW_PTR(dcheck),size);
    timer.Stop();
    diff = timer.Elapsed()*1000000;
    printf("<<<time>>> %.2lf ( %.2f)\n",
           (double) diff, (double) (size) / diff);
    cudaDeviceSynchronize();
    cudaGetLastError();

    thrust::host_vector<TYPE> check(dcheck);
    check_search_result(RAW_PTR(key),RAW_PTR(value),RAW_PTR(check),size);


}