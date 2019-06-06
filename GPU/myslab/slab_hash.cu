//
// Created by jing on 2019-03-17.
//


//#include <cstdint>
#include "cuda_runtime.h"
#include "slab_hash.h"
#include "include/dy_hash.h"
#include "include/gputimer.h"

#include <helper_functions.h>
#include <helper_cuda.h>

#define ulong unsigned long long int
///
#define make_ulong(k, v) ((((ulong)v)<<32)+k)
#define getk_from_ulong(kv) ((unsigned)((kv)&0xffffffff))
#define getv_from_ulong(kv) ((unsigned)((kv)>>32))
#define gtid (blockIdx.x * blockDim.x + threadIdx.x)

#define DE_ACTIVE 0
#define ACTIVE 1

class Node {
public:
    unsigned _entry[32];

    //__device__ Node* next(){ return (Node*)&_entry[30]; }
    __device__ ulong getkv(int pos) {
        return ((ulong *) _entry)[pos];
    }

    __device__ ulong getnext() {
        return getkv(15);
    }

    __device__ bool ptrCAS(ulong old_val, ulong node) {
        long old = atomicCAS((ulong *) &_entry[30], old_val, node);
        return old == old_val;
    }

    __device__ bool set_kv_on_pos(int k, int v, int pos, ulong oldkv) {
        ulong newkv = make_ulong(k, v);
        ulong old = atomicCAS((ulong *) &_entry[pos * 2], oldkv, newkv);
#if CAS_debug
        printf(">>>CAS:id:%d,kv %d,%d pos %d \n>>>CAS:oldkv %lld,%lld CASold: %lld,%lld kv: %lld, %lld pos: %d,%d\n",
                threadIdx.x,k,v,pos,
                oldkv>>32,oldkv& 0xffffffff,old>>32,old& 0xffffffff,newkv>>32,newkv& 0xffffffff,
                _entry[pos*2],_entry[pos*2+1]);
        show("in CAS",k);
#endif

        return old == oldkv;
    }

    __device__ bool set_ptr(Node *ptr) {
        return ptrCAS(getnext(), (ulong) ptr);
    }

    __device__ void show(const char *src, int pos = 0) {
        printf("%s: val:%d tid:%d loc:%lx ", src, pos, gtid, &_entry[0]);
#if kv_in_Node
        printf("\n");
        for(int i=0;i<30;i++){
            printf("%d ",_entry[i]);
        }
#endif
        printf("next: %lx\n", getnext());
    }
};

class MemPool {
public:
    Node *data[MEM_POOL_NUM];
    unsigned int pos[MEM_POOL_NUM];

    MemPool() {}

    MemPool(int s) {
        printf("memory pool %d initing\n", s);
        Node *ptr;
        cudaMalloc((void **) &ptr, sizeof(Node) * MEM_POOL_NUM * NODE_OF_POOL);
        cudaMemset(ptr, 0, sizeof(Node) * MEM_POOL_NUM * NODE_OF_POOL);
        for (int i = 0; i < MEM_POOL_NUM; i++) {
            data[i] = ptr + i * NODE_OF_POOL;
        }
        memset(pos, 0, sizeof(int) * MEM_POOL_NUM);
        printf("memory pool %d inited\n", s);
    }

    __device__ void show_pool() {
        for (int i = 0; i < NODE_OF_POOL; i++) {
            printf("%d %d %llx\n", 0, i, &data[0][i]);
            data[0][i].show("mempool show", i);
        }
    }

    __device__ Node *get_Node() {

//        int tid = gtid >> 5;
        int tid=blockIdx.x;
#if using_CAS_OR_ADD
        int old=atomicAdd(&pos[tid],1);
#else
        int old = pos[tid] + 1;
#endif
#if show_node_alloc
        printf("adding new node,nodeid %d, pool id %d, gtid:%d\n",old,tid,gtid);
#endif
        if (old >= NODE_OF_POOL)
            return NULL;

        return &data[tid][old];
    }

    __device__ void free_Node(Node *n) {
        /// we do not free it now
        return ;
//        int tid = gtid >> 5;
//        int unuse_pos = n - data[tid];
//        assert(unuse_pos >= 0);
//#if using_CAS_OR_ADD
//        atomicCAS(&pos[tid],unuse_pos+1,unuse_pos);
//#else
//        pos[tid]--;
//#endif
    }
};



__device__ MemPool *mem_pool;

void __inline__ printinit() {
    static int a = 1;
    if (a == 1) {
        printf("slab list inited\n");
        a = 0;
    }
}


__device__ bool
alloc_one_new_node(Node *tmp) {
    int id_in_warp = threadIdx.x & 0x1f;
    if (id_in_warp == 0) {
        /// TODO (zj)  add block shared pool strategy
        Node *nxt = mem_pool->get_Node();
        if (nxt == NULL) {
            printf("!!!! error , pool used over tid:%d ,nxt:%lld \n", gtid, nxt);
            return false;
        }
        /// set ptr or free to pool
        if (tmp->set_ptr(nxt) == false) {
            mem_pool->free_Node(nxt);
        }
    }
    return true;
}

/// check target exist ,if exist CAS until there is no target
__device__ bool __forceinline__
check_target_and_CAS(int k, int v, int &is_active, int target, Node *tmp) {
    int id_in_warp = threadIdx.x & 0x1f;
    int ballot = __ballot(tmp->_entry[id_in_warp] == target);
    ballot &= MASK_CHECK_kv;

    ///  if there is  target
    while (ballot != 0) {
        int chosen_simd = __ffs(ballot) - 1;
        ulong oldkv = tmp->getkv(chosen_simd / 2);  /// for kv pair
        if (getk_from_ulong(oldkv) == target) {
            if (id_in_warp == chosen_simd) {
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
        ballot = __ballot(tmp->_entry[id_in_warp] == target);
        ballot &= MASK_CHECK_kv;
    }
    return false;
}

// after ballot , warp insert
__device__ bool __forceinline__
warp_insert(Node *ht, int k, int v) {

    int id_in_warp = threadIdx.x & 0x1f;
    Node *tmp = ht;
    assert(tmp != NULL);

    int is_active = ACTIVE;

    while (tmp != NULL) {
#if insert_debug
        if(id_in_warp==0)  tmp->show("check in main while in insert ",k);
#endif

        // check exist
//            if( check_target_and_CAS(k,v,is_active,k,tmp) )
//                return true;

        /// check null
        if (check_target_and_CAS(k, v, is_active, 0, tmp))
            return true;

        if (NULL == (Node *) (tmp->getnext()))
            alloc_one_new_node(tmp);

        tmp = (Node *) (tmp->getnext());
    }

    if (id_in_warp == 0)
        printf("need resize\n");
    return false;
}

__device__ bool __forceinline__
check_target_and_return_value(int k, Node *tmp, int write_pos, TYPE *value) {
    int id_in_warp = threadIdx.x & 0x1f;
    int ballot = __ballot(tmp->_entry[id_in_warp] == k);
//    if(gtid==0) printf("find k:%d ,ballot:%llx\n",k,ballot),tmp->show("check in searching ballot ",k);;
    ballot &= MASK_CHECK_kv;
//    if(gtid==0) printf("find k:%d ,ballot:%lx\n",k,ballot);
    if (ballot != 0) {
        int chosen_simd = __ffs(ballot) - 1;
        ulong oldkv = tmp->getkv(chosen_simd / 2);
//        if(gtid==0) printf("find k:%d ,ballot:%llx,bucket:%d %d\n",k,ballot,getk_from_ulong(oldkv),getv_from_ulong(oldkv));
        if (id_in_warp == chosen_simd) {
            value[write_pos] = getv_from_ulong(oldkv);
        }
        return true;
    }
    return false;
}

__device__ bool __forceinline__
check_target_and_return_value_no_ballot(int k, Node *tmp, int write_pos, TYPE *value) {
    int id_in_warp = threadIdx.x & 0x1f;
    if(tmp->_entry[id_in_warp] == k){
        value[write_pos] = tmp->_entry[id_in_warp+1];
    }
}


__device__ bool __forceinline__
warp_search(Node *ht, int k, int write_pos, TYPE *value) {
    Node *tmp = ht;
    while (tmp != NULL) {
        check_target_and_return_value(k, tmp, write_pos, value);
        tmp = (Node *) (tmp->getnext());
    }
    return false;
}

__device__ __forceinline__ void
show_list(int n, Node *ht) {
    Node *tmp = ht;
    for (int i = 0; i < n && tmp != NULL; i++) {
        tmp->show("list show", i);
        tmp = (Node *) (tmp->getnext());
    }
}


__device__ TYPE __forceinline__
hash_function(TYPE k) {
    return ((k ^ 59064253) + 72355969) % PRIME_uint;
}


__global__ void
kernel_find_test(Node *ht, TYPE *key, TYPE *value, int size, int table_size) {
    int tid = gtid;
    int id_of_warp = (tid) >> 5;
    int step = (gridDim.x * blockDim.x) >> 5;

    for (; id_of_warp < size; id_of_warp += step) {
        TYPE k = key[id_of_warp];
        warp_search(ht + hash_function(k) % table_size, k, id_of_warp, value);
//        ht[k % TABLE_NUM].warp_search(k,id_of_warp, value );
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
kernel_test(Node *ht, TYPE *key, TYPE *value, int size, int table_size) {
    int tid = gtid;
    int id_of_warp = (tid) >> 5;
    int step = (gridDim.x * blockDim.x) >> 5;

    for (; id_of_warp < size; id_of_warp += step) {
        int k = key[id_of_warp];
        int v = value[id_of_warp];
//        if(gtid==0) printf("kv %d %d\n",k,v);
        warp_insert(ht + hash_function(k) % table_size, k, v);
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
cal_avg_len(Node *ht, int table_size) {

    int maxl=0,sumlen=0;
    for (int i=0; i < table_size; i += 1) {
        Node* tmp=ht+i;
        int tmplen=0;
        while(tmp!=NULL){
            tmplen++;
            tmp = (Node *) (tmp->getnext());
        }
        sumlen+=tmplen;
        maxl=tmplen>maxl?tmplen:maxl;
    }
    printf("suml:%d average:%f,max:%d\n",sumlen,sumlen*1.0/table_size,maxl);
}

__global__ void
set_mempool(MemPool *pool) {
    mem_pool = pool;
}

using namespace std;


unsigned int *Read_Data(char *filename,int size) {
    if (strcmp(filename, "/home/udms/ly/GPU_Hash/finally-test/data/twitter.dat") == 0)
        size = size / 2;
    if (strcmp(filename, "/home/udms/ly/GPU_Hash/finally-test/data/tpc-h.dat") == 0)
        size = size / 2;
    if (strcmp(filename, "/home/udms/ly/GPU_Hash/finally-test/data/real_2018/l32.dat") == 0)
        size = size / 10;

    FILE *fid;
    fid = fopen(filename, "rb");
    unsigned int *pos;
    pos = (unsigned int *) malloc(sizeof(unsigned int) * size);//申请内存空间，大小为n个int长度

    if (fid == NULL) {
        printf("the data file is unavailable.\n");
        exit(1);
        return pos;
    }
    fread(pos, sizeof(unsigned int), size, fid);
    fclose(fid);
    return pos;
}


bool init_kv(TYPE *key, TYPE *value, char *filename, int size) {
//     GenerateUniqueRandomNumbers(key);
//    TYPE* k=Read_Data(filename,size);
    for (int i = 0; i < size; i++) {
        key[i] = (TYPE) rand();
//        key[i] = i+2;
//        key[i]=k[i];
        value[i] = (TYPE) i+1;

    }
//    free(k);

    return true;
}


void check_result(TYPE *key, TYPE *value, TYPE *check, int size) {
    int tmp = 0;
    for (int i = 0; i < size; i++) {
        if (value[i] != check[i]) {
            if (tmp++ < 20) printf("false: %d :k:%d v:%d find:%d\n", i, key[i], value[i], check[i]);
        }
    }
    printf("check result:%d insert ,find %d ,not find %d (%d)\n", size, size - tmp, tmp, tmp * 1.0 / size);
}


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

using namespace thrust;
#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())
void cheching_result(const host_vector<unsigned int> &key, const host_vector<unsigned int> &value,
                     const host_vector<unsigned int> &check, int size);

/// hold hash table , pointer in host
static Node *d_hash_table;
static int table_size = TABLE_NUM;
static MemPool *d_pool;


void simple_gpu_test(char *filename) {

    int size = NUM_DATA;
    if (strcmp(filename, "/home/udms/ly/GPU_Hash/finally-test/data/twitter.dat") == 0)
        size = size / 2;
    if (strcmp(filename, "/home/udms/ly/GPU_Hash/finally-test/data/tpc-h.dat") == 0)
        size = size / 2;
    if (strcmp(filename, "/home/udms/ly/GPU_Hash/finally-test/data/real_2018/l32.dat") == 0)
        size = size / 10;


    // alloc data
    thrust::host_vector<TYPE> key(size + 1);
    thrust::host_vector<TYPE> value(size + 1);


    //init data
    init_kv(RAW_PTR(key), RAW_PTR(value), filename, size);
    printf("=========init over==========\n");

    // copy to  gpu
    thrust::device_vector<TYPE> dkey(key);
    thrust::device_vector<TYPE> dvalue(value);
    thrust::device_vector<TYPE> dcheck(size + 1);

    // 初始化hash表
    cudaMalloc((void **) &d_hash_table, TABLE_NUM * sizeof(Node));
    cudaMemset(d_hash_table, 0, TABLE_NUM * sizeof(Node));


    // 初始化 pool
    MemPool h_pool(0);

    cudaMalloc((void **) &d_pool, sizeof(MemPool));
    cudaMemcpy(d_pool, &h_pool, sizeof(MemPool), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaGetLastError();

    /// set pool
    set_mempool <<< 1, 1 >>> (d_pool);


    printf("=======starting insert=======\n");
    // 插入数据
    GpuTimer timer;
    timer.Start();
    kernel_test <<< BLOCK_NUM, THREAD_NUM >>> (d_hash_table, RAW_PTR(dkey), RAW_PTR(dvalue), size, table_size);
    timer.Stop();
    double diff = timer.Elapsed() * 1000000;
    printf("<<<time>>> %.2lf ( %.2f)\n",
           diff, (double) (size) / diff);
    cudaDeviceSynchronize();
    cudaGetLastError();


    // find 数据
    timer.Start();
    kernel_find_test <<< BLOCK_NUM, THREAD_NUM >>> (d_hash_table, RAW_PTR(dkey), RAW_PTR(dcheck), size, table_size);
    timer.Stop();
    diff = timer.Elapsed() * 1000000;
    printf("<<<time>>> %.2lf ( %.2f)\n",
           diff, (double) (size) / diff);
    cudaDeviceSynchronize();
    cudaGetLastError();

    cal_avg_len<<<1,1>>>(d_hash_table,table_size);

    /// copy back
    thrust::host_vector<TYPE> check(dcheck);




    cheching_result(key, value, check, size);
}

void cheching_result(const host_vector<unsigned int> &key, const host_vector<unsigned int> &value,
                     const host_vector<unsigned int> &check, int size)
{ // check function
    unsigned int tmp = 0;
    for (unsigned int i = 0; i < size; i++) {
        if (raw_pointer_cast((value).data())[i] ==0){ //!= raw_pointer_cast((check).data())[i]) {
            if (tmp < 20)
                printf("check i:%d error k:%d search:%d except:%d \n", i, raw_pointer_cast((key).data())[i],
                       raw_pointer_cast((value).data())[i], raw_pointer_cast((check).data())[i]);
            tmp++;
        }
    }

    if (tmp != 0)
        printf("\t%d/%d not pass:abort %.2f\n", tmp, size, tmp * 1.0 / size);
    else
        printf("checking pass\n");

}

