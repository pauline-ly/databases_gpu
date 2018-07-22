#include"cuda_runtime.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
typedef unsigned long long LL;


//9    100000007
//8       43112609
//7-digit 1111151
//6-d     100003
#define NUM_BUCKETS 100003
#define NUM_ITEMS   1000000
#define NUM_THREADS 1024
#define NUM_BLOCK 16384
#define KEYS 0x0080000000000000

// Definition of generic node class
//class __attribute__((aligned (16))) Node
class Node
{
public:
    LL key;
    LL value;
    Node* next;

    Node()//init
    {
        next=NULL;
    }
};


// class Hash_Table
// {
// public:
//     Node *head;
//     Node *pool;
// };

__device__ Node* table;



__device__ bool
Search(Node &head,LL key)
{
    Node* curr = head.next;
    while(true){
        if(curr->key >= key)
        {
            return (curr->key == key);
        }
        curr=curr->next;
    }
}



__device__ bool
Add(Node &table,Node* pointer)
{

    //printf("111:%llu 2:%llu 3:%llu\n",(LL)(table.next), NULL, (LL)pointer);
    if(NULL==atomicCAS((LL*)&(table.next), NULL, (LL)pointer))
        return true;
    //printf("121:%llu 2:%llu 3:%llu\n",(LL)(table.next), NULL, (LL)pointer);
    Node* curr=&table;

    while(true)
    {
        if(curr->key==pointer->key)  //find the same key
        {
            return true;
        }
        
        if(curr->next!=NULL)//如果curr不是最后一个节点
        {
            curr=curr->next;
            continue;
        }
        //printf("21 curr:%llu   1:%llu 2:%llu 3:%llu\n",(LL)curr,(LL)(curr->next), NULL, (LL)pointer);   
        if(NULL==atomicCAS((LL*)&(curr->next), NULL, (LL)pointer))
        {
         //printf("22 curr:%llu   1:%llu 2:%llu 3:%llu\n",(LL)curr,(LL)(curr->next), NULL, (LL)pointer);
            return true;
        }
    }
}

__device__ __inline__
LL Hash(LL x)
{
    return x%NUM_BUCKETS;
}


__global__
void kernel_creat(LL* key, Node* n,Node* h)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
//     if(tid==0){
//         tables.pool=n;//set  pre-allocated
//         tables.head=h;
//     }
    
    
    table=h;
    //NUM_ITEMS :need be inserted items number
    while(tid<NUM_ITEMS){
        // Grab the operation and the associated key and execute
        LL k = key[tid];
        //printf("%d\t",k);
        LL bkt = Hash(k);
        Node* p = &n[tid];    //get node for pool
        p->key=k;
        p->value=tid;
        p->next=NULL;
//         Add((tables.head[bkt])->next,p);
        Add(h[bkt],p);
        
        tid+=NUM_BLOCK*NUM_THREADS;
    } 
}


__global__
void kernel_find(LL* key)
{  
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    
    //NUM_ITEMS :need be inserted items number
    while(tid<NUM_ITEMS){
        LL k = key[tid];
        LL bkt = Hash(k);
        Search(table[bkt],k);
        tid+=NUM_BLOCK*NUM_THREADS;
    } 
    
}





int main(int argc, char** argv)
{
    //printf("%llu",TAIL);
    //printf("\n%llu\n",HEAD);
    //init v
    //alloc node list
    //init list
    //init data
    //run kernel
    //free
   
    checkCudaErrors(cudaGetLastError());

    LL* h_items=(LL*)malloc(sizeof(LL)*NUM_ITEMS);		// Array of keys
    // value is tid
    srand(0);
    // NUM_ITEMS is the total number of operations to execute
    for(int i=0;i<NUM_ITEMS;i++){
        h_items[i]=10+rand()%KEYS;	// Keys
    }
    
    //copy to device
    LL* d_items;
    checkCudaErrors(cudaMalloc((void**)&d_items, sizeof(LL)*NUM_ITEMS));
    checkCudaErrors(cudaMemcpy(d_items, h_items, sizeof(LL)*NUM_ITEMS,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetLastError());  

    // Allocate hash table
    Node* h_buckets=new Node[NUM_BUCKETS];
 
    //copy to device
    Node* d_buckets;
    checkCudaErrors(cudaMalloc((void**)&d_buckets, sizeof(Node)*NUM_BUCKETS));
    checkCudaErrors(cudaMemcpy(d_buckets, h_buckets, sizeof(Node)*NUM_BUCKETS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetLastError());

    Node* h_pointers=(Node*)malloc(sizeof(Node)*NUM_ITEMS);   //pool
    Node* d_pointers;
    checkCudaErrors(cudaMalloc((void**)&d_pointers,sizeof(Node)*NUM_ITEMS));


    checkCudaErrors(cudaMemcpy(d_pointers, h_pointers, sizeof(Node)*NUM_ITEMS,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetLastError());
    int b=(NUM_ITEMS+NUM_THREADS-1)/NUM_THREADS;
    dim3 block=b<NUM_BLOCK? b : NUM_BLOCK;

    //create event
    cudaEvent_t ts,tn;
    cudaEventCreate(&ts);
    cudaEventCreate(&tn);
    //record stard
    cudaEventRecord(ts,0);

    kernel_creat<<<block,NUM_THREADS>>>(d_items, d_pointers,d_buckets);
    checkCudaErrors(cudaGetLastError());
    //record end
    cudaEventRecord(tn, 0);
    cudaEventSynchronize(tn);

    //printf elapsedtime
    float te;
    cudaEventElapsedTime(&te, ts, tn);
    printf("creat time (ms): %lf\n",te);

     
    cudaEvent_t ts1,tn1;
    cudaEventCreate(&ts1);
    cudaEventCreate(&tn1);

    //record stard
    cudaEventRecord(ts1,0);

    kernel_find<<<block,NUM_THREADS>>>(d_items);
    

    //record end
    cudaEventRecord(tn1, 0);
    cudaEventSynchronize(tn1);

    //printf elapsedtime
    float te1;
    cudaEventElapsedTime(&te1, ts1, tn1);
    printf("find time (ms): %lf\n",te1);
    
   

    return 0;
}