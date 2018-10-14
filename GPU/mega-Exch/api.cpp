//
// Created by jing on 2018/7/1.
//

#include "api.h"
#include "mt19937ar.h"
#include "dy_hash.h"


/// size : the key  in the table
hashAPI::hashAPI(TYPE size) {
    /// common var
    batch_del_num_to_op=0;
    num_size=0;
    d_elem_pool=NULL;
    d_check=NULL;
    batch_num_to_op=0;

    ///cnmem init
    memset(&device, 0, sizeof(device));
    device.size = (size_t)4*1024*1024*1024; /// more =(size_t) (0.95*props.totalGlobalMem);
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
    checkCudaErrors(cudaGetLastError());

    /// calcul size
    int s_table=(size);
    s_table= (s_table/(ELEM_NUM))*(ELEM_NUM); //保证整数倍
    if(s_table==0) s_table+=ELEM_NUM*2*2;
    table_size=(unsigned)s_table;
#if size_info
    printf("init_table_size:%d\n",s_table);
#endif


    /// malloc gpu tables
    cnmemMalloc((void**)&d_hash_table, table_size * 2 * sizeof(uint32_t), 0); /// hold kv
    cudaMemset(d_hash_table, 0, table_size * 2 * sizeof(uint32_t));
    EXPECT_TRUE(d_hash_table);

    checkCudaErrors(cudaGetLastError());
}

hashAPI::~hashAPI() {
    cnmemFree(d_hash_table,0);
    cnmemFree(d_elem_pool,0);
    cnmemFree(d_check,0);
    /// need?
    cnmemRelease();
}

void hashAPI::resize(TYPE new_size) {



    int old_size=table_size;
    table_size =new_size;
#if size_info
    printf("api-resize:%d -> %u\n",old_size,table_size);
#endif
    char* old_table=d_hash_table;
    /// alloc old table

    d_hash_table=NULL;
    cnmemMalloc((void**)&d_hash_table, table_size * 2 * sizeof(uint32_t), 0);
    cudaMemset(d_hash_table, 0, table_size * 2 * sizeof(uint32_t));
    EXPECT_TRUE(d_hash_table);

    /// call resize kernel
    gpu_hash_resize(table_size,(bucket_t*)d_hash_table,old_size,(bucket_t*)old_table);

    checkCudaErrors(cudaGetLastError());
    ///free old
    cnmemFree(old_table,0);

}

void hashAPI::hash_insert(TYPE *key, TYPE *value,TYPE size){

    /// check resize
//    if(num_size+size >= Upper_Bound(table_size)){
//        resize(table_size*NUM_upsize_ratio);
//        hash_insert(key,value,size);
//        return ;
//    }

    /// malloc operator kv
    ielem_t *h_insert=(ielem_t*)malloc(size*sizeof(ielem_t));
    for (int i = 0; i <size ; ++i) {
        h_insert[i].sig=key[i];
        h_insert[i].loc=value[i];
        h_insert[i].hash=0;// hash will be calcul in kernel
    }
    ielem_t *d_insert;
    cnmemMalloc((void**)&d_insert,size*sizeof(ielem_t),0);
    cudaMemcpy(d_insert,h_insert,size*sizeof(ielem_t),cudaMemcpyHostToDevice);


    num_size+=size;
    /// insert
    gpu_hash_insert((bucket_t*)d_hash_table,table_size,d_insert,size);

    if(size<=100)gpu_show_table((bucket_t*)d_hash_table,table_size);

    free(h_insert);
    cnmemFree(d_insert,0);
    checkCudaErrors(cudaGetLastError());
}

void hashAPI::hash_search(TYPE *key, TYPE *ans,TYPE size){

    /// alloc
    ielem_t *h_elem=(ielem_t*)malloc(size*sizeof(ielem_t));
    for (int i = 0; i <size ; ++i) {
        h_elem[i].sig=key[i];
        h_elem[i].loc=0;
        h_elem[i].hash=0;// hash will be calcul in kernel
    }
    ielem_t *d_elem;
    cnmemMalloc((void**)&d_elem,size*sizeof(ielem_t),0);
    cudaMemcpy(d_elem,h_elem,size*sizeof(ielem_t),cudaMemcpyHostToDevice);
    TYPE* d_value;
    cnmemMalloc((void**)&d_value, sizeof(TYPE)*size,0);
    cudaMemset(d_value,0, sizeof(TYPE)*size);

    //kernel
    gpu_hash_search(d_elem,d_value,(bucket_t*)d_hash_table,table_size,size);

    cudaMemcpy(ans, d_value, sizeof(TYPE)*size, cudaMemcpyDeviceToHost);

    free(h_elem);
    cnmemFree(d_elem,0);
    cnmemFree(d_value,0);
    checkCudaErrors(cudaGetLastError());

    /// free
}

/// mega kv do now return del value
void hashAPI::hash_delete(TYPE *key,TYPE* value,TYPE size) {

    /// malloc operator kv
    ielem_t *h_del=(delem_t*)malloc(size*sizeof(delem_t));
    for (int i = 0; i <size ; ++i) {
        h_del[i].sig=key[i];
        h_del[i].loc=value[i];
        h_del[i].hash=0;// hash will be calcul in kernel
    }
    delem_t *d_del;
    cnmemMalloc((void**)&d_del,size*sizeof(delem_t),0);
    cudaMemcpy(d_del,h_del,size*sizeof(delem_t),cudaMemcpyHostToDevice);

    num_size-=size;
    //kernel
    gpu_hash_delete(d_del, (bucket_t*)d_hash_table, table_size,size);


    free(h_del);
    cnmemFree(d_del,0);

    if(num_size+size <= Lower_Bound(table_size)){
        resize((TYPE)(table_size*NUM_downsize_ratio));
        return ;
    }

}


bool hashAPI::set_data_to_gpu(TYPE *key,TYPE* value,TYPE size){
    EXPECT_LE(size, max_input_size) << "set data error size of kv out of 2G";

    /// malloc operator kv
    ielem_t *h_elem=(ielem_t*)malloc(size*sizeof(ielem_t));
    for (int i = 0; i <size ; ++i) {
        h_elem[i].sig=key[i];
        h_elem[i].loc=value[i];
        h_elem[i].hash=0;// hash will be calcul in kernel
    }
    cnmemMalloc((void **)&d_elem_pool, size * sizeof(ielem_t), 0);
    checkCudaErrors(cudaGetLastError());
    EXPECT_TRUE(d_elem_pool!=NULL)<<"malloc error";

    checkCudaErrors(cudaMemcpy(d_elem_pool, h_elem, size * sizeof(TYPE), cudaMemcpyHostToDevice));
    /// set batch pointer
    for (int i = 0; i * BATCH_SIZE < size && i < BATCH_NUM; i++) {
        d_elem_batch[i] = d_elem_pool + BATCH_SIZE * i;
    }

    cnmemMalloc((void **)&d_check, BATCH_SIZE * sizeof(TYPE), 0);
    cudaMemset(d_check,0,BATCH_SIZE * sizeof(TYPE));
    EXPECT_TRUE(d_check!=NULL)<<"malloc error";

    free(h_elem);

    return true;
}

void hashAPI::hash_insert_batch() {
    if(batch_num_to_op >= BATCH_NUM ){
        EXPECT_LT(batch_num_to_op,BATCH_NUM);
        return ;
    }

    /// check resize
    if(num_size+BUCKET_SIZE >= Upper_Bound(table_size)){
        resize(table_size*NUM_upsize_ratio);
        hash_insert_batch();
        return ;
    }

    num_size+=BATCH_SIZE;

    gpu_hash_insert(
            (bucket_t*)d_hash_table,
            table_size,
            d_elem_batch[batch_del_num_to_op],
            BATCH_SIZE);


    batch_num_to_op++;
}


void hashAPI::hash_search_batch() {
    if(batch_num_to_op >= BATCH_NUM ){
        EXPECT_LT(batch_num_to_op,BATCH_NUM);
        return ;
    }

    gpu_hash_search(
            d_elem_batch[batch_num_to_op],
            d_check,
            (bucket_t*)d_hash_table,
            table_size,
            BATCH_SIZE);

    batch_num_to_op++;
    /// copy back?
}


void hashAPI::hash_delete_batch() {

    if(batch_del_num_to_op >= BATCH_NUM ){
        EXPECT_LT(batch_del_num_to_op,BATCH_NUM);
        return ;
    }

    gpu_hash_delete(d_elem_batch[batch_del_num_to_op], (bucket_t*)d_hash_table, table_size,BATCH_SIZE);


    batch_del_num_to_op++;

    /// check size ,need resize ?
    if(num_size<Lower_Bound(table_size) && table_size>(TABLE_NUM*1024*1024*1.2)){
        resize((TYPE)(table_size*NUM_downsize_ratio));
        return;
    }

}





//void hashAPI::hash_delete(TYPE *key,TYPE *ans,TYPE size) {
//    for(TYPE i=0;i<size;i++){
//        ans[i]=table->remove(key[i]);
//    }
//}

