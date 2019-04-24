/*
 * Copyright (c) 2015 Kai Zhang (kay21s@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include "gpu_hash.h"

#define bucketk(b,i) ((uint32_t)((b->sigloc[i])>>32))
#define bucketv(b,i) ((uint32_t)((b->sigloc[i])&0xffffffff))
#define bucket_all(b,i) (b->sigloc[i])


__device__ __forceinline__ uint32_t
get_hash(uint32_t sig){
    return ((sig ^ 6409721) + 7837638) % PRIME_uint;
}

__device__ __forceinline__ uint32_t
get_hash2(uint32_t sig){
    return ((sig ^ 28264960) + 91843754) % PRIME_uint;
}

__device__ __forceinline__ uint32_t
get_hash3(uint32_t sig){
    return ((sig ^ 59064253) + 72355969) % PRIME_uint;
}

#define hash_fun_num 4


__device__ __forceinline__ uint32_t
hash_function(uint32_t sig, int num){
//    fill 1.05 25->20
    if(num%hash_fun_num==0) return ((sig ^ 7837638) + 6409721) % PRIME_uint;
    if(num%hash_fun_num==1) return get_hash2(sig);
    if(num%hash_fun_num==2) return get_hash3(sig);
    if(num%hash_fun_num==3) return get_hash(sig);
}



__global__ void hash_search(
        ielem_t			*in,
        loc_t			*out,
        int				size,
        bucket_t		*hash_table,
        TYPE            bucket_num_of_single_table)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int id = 0;
	// (1 << ELEM_NUM_P) threads to cooperate for one element
	int step = (gridDim.x * blockDim.x) >> ELEM_NUM_P;
	int ballot;
	
	int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
	int elem_id = idx >> ELEM_NUM_P;

	int hash;
	int bit_move;
	bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);

    selem_t elem1;

	for (id = elem_id; id < size; id += step) {
        selem_t *elem=&elem1;
		elem->sig = in[id].sig;
        for (int i_num = 0; i_num <hash_fun_num ; ++i_num) {
            hash=hash_function(elem->sig,i_num);
            bucket_t *b = &(hash_table[hash % bucket_num_of_single_table]);
            if (bucketk(b,simd_lane) == elem->sig) {
                out[id ] = 1;
            }
        }
	}

	return;
}



__device__ __forceinline__ Entry
read_input( ielem_t *input, int &lan_num_in_all,Entry entry) {
    ielem_t *elem = &input[lan_num_in_all];
//    hash = get_hash(elem->sig);
    entry=makeEntry(elem->sig,elem->loc);
    return entry;
}

__device__ __forceinline__ bool
exist_insert(
        volatile Entry *shared_entry,
        int simd_lane_in_block,
        bucket_t *b,
        int simd_lane,
        int bit_move,
        unsigned long long int entry) {


    int ballot = __ballot(bucketk(b, simd_lane) == getk(entry));//find ok
    ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);// whether find
    int chosen_simd;
    /// exist the same key?
    if (ballot != 0) { //try updata loc
        Entry old;
        chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
#if insert_debug
if(simd_lane==0) printf("try :%d %d to %d %d\n",
        getk(entry),getv(entry),bucketk(b,chosen_simd),bucketv(b,chosen_simd));
#endif
        if (simd_lane == chosen_simd) {
            printf("exist:%d-%d:%d->%d\n", getk(entry), bucketk(b,simd_lane), bucketv(b,simd_lane), getv(entry));
            old=atomicExch(&bucket_all(b,simd_lane),entry);
            shared_entry[simd_lane_in_block]=old;
        }
        old=shared_entry[chosen_simd];
        if(getk(old)==getk(entry)) return true;
    }
    return entry == makeEntry(0, 0);
}

__device__ __forceinline__ bool
exch_of_choose_simd(
        volatile Entry *shared_entry,
        bucket_t *b,
        int chosen_simd,
        int simd_lane_in_block,
        int simd_lane,
        Entry &entry)
{
#if insert_debug
    if(blockIdx.x * blockDim.x + threadIdx.x==0 ) printf("try :%d %d to %d %d\n",
                            getk(entry),getv(entry),bucketk(b,chosen_simd),bucketv(b,chosen_simd));
#endif

    if (simd_lane == chosen_simd) {
        Entry tmpCAS = atomicExch(&bucket_all(b,simd_lane), entry);
        shared_entry[simd_lane_in_block]=tmpCAS;
    }
    entry=shared_entry[simd_lane_in_block];
    return entry == makeEntry(0, 0);
}


__global__ void
hash_insert_cuckoo(
		bucket_t		*hash_table,
		TYPE            bucket_num_of_single_table,
		ielem_t			*input,
		int				elem_num) {

    int idx = threadIdx.x;//mmm id
    int lan_num_in_all = (blockIdx.x * blockDim.x + threadIdx.x) >> ELEM_NUM_P;
    volatile __shared__ uint64_t shared_entry[(THREAD_NUM) >> ELEM_NUM_P];
    int step = (gridDim.x * blockDim.x) >> ELEM_NUM_P;

    bucket_t *b;
    int chosen_simd;
    uint32_t hash;

    int simd_lane_in_block=idx>>ELEM_NUM_P;
    int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
    int bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);

#if insert_debug
    if (lan_num_in_all == 0 && simd_lane==0) {
        printf("elem:%d step %d \n", elem_num, (gridDim.x * blockDim.x) >> ELEM_NUM_P);
    }
#endif
    Entry entry;

    for (; lan_num_in_all <elem_num ; lan_num_in_all+=step) {

        entry= read_input(input,lan_num_in_all,entry);
#if insert_debug
        if (simd_lane == 0)
            printf(" kv  %d,%d\n", getk(entry), getv(entry));
#endif
        for (int evict_num = 0; evict_num <MAX_CUCKOO_NUM ; ++evict_num) {
            /// get insert bucket
            hash= hash_function(getk(entry), evict_num);
            hash %= bucket_num_of_single_table;
            b = &(hash_table[hash]);// bucket num
            /// try update
            bool ans=exist_insert(shared_entry,simd_lane_in_block,b, simd_lane, bit_move, entry);
            if(ans) break;

            int ballot = __ballot(bucketk(b,simd_lane) == 0);
            ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
            if (ballot != 0) {
                chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
            }else{
                chosen_simd = getk(entry) & ((1 << ELEM_NUM_P) - 1);
            }
#if insert_debug
            if(simd_lane==0&&lan_num_in_all%step==0){
                printf(ballot==0?"evict":"tey");
                printf("hash:%d kv%d %d",hash%bucket_num_of_single_table,getk(entry),getv(entry));
            }
#endif
            /// try insert
            ans=exch_of_choose_simd(shared_entry, b, chosen_simd, simd_lane_in_block, simd_lane, entry);
            if(ans) break;

        }
#if insert_debug
        if(getk(entry)!=0&& simd_lane==0)printf("give up :%d %d",getk(entry),getv(entry));
#endif
    }
    return;
}




__global__ void hash_delete(
		delem_t			*in,
		int             total_elem_num,
		bucket_t		*hash_table,
        TYPE            bucket_num_of_single_table){}
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int id = 0;
//	// 16 threads to cooperate for one element
//	int step = (gridDim.x * blockDim.x) >> ELEM_NUM_P;
//	int ballot;
//
//	int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
//	int elem_id = idx >> ELEM_NUM_P;
//	bucket_t *b;
//
//	int bit_move;
//	bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);
//
//
//
//    for (id = elem_id; id < total_elem_num; id += step) {
//        ielem_t *elem=&in[id];
//        elem->hash = get_hash(elem->sig);
//
//		b = &(hash_table[(elem->hash) % bucket_num_of_single_table]);
//		/* first perform ballot */
//		ballot = __ballot(bucketk(b,simd_lane) == elem->sig && bucketv(b,simd_lane) == elem->loc);
//
//		if (bucketk(b,simd_lane) == elem->sig && bucketv(b,simd_lane) == elem->loc) {
//			bucketk(b,simd_lane) = 0;
//		}
//
//
//		//b = &(hash_table[(elem->hash ^ elem->sig) & HASH_MASK]);
//		int hash = (((elem->hash ^ elem->sig) & BLOCK_HASH_MASK)
//				| (elem->hash & ~BLOCK_HASH_MASK)) % bucket_num_of_single_table;
//		b = &(hash_table[hash]);
//		if (bucketk(b,simd_lane) == elem->sig && bucketv(b,simd_lane) == elem->loc) {
//			bucketk(b,simd_lane) = 0;
//		}
//	}
//
//	return;
//}

__global__ void
hash_reisze_cuckoo(
        bucket_t		*hash_table,
        TYPE            bucket_num_of_single_table,
        bucket_t		*old_table,
        int				old_bucket_num_of_single_table){}
//{
//    int idx = threadIdx.x;//mmm id
//    int id=(blockIdx.x * blockDim.x + threadIdx.x)>>ELEM_NUM_P;
//    int step=((gridDim.x * blockDim.x)>>ELEM_NUM_P);
//
//    hash_t hash, second_hash;
//    loc_t loc, new_loc;
//    sign_t sig, new_sig;
//
//    int cuckoo_num;
//    bucket_t *b;
//    int chosen_simd;
//    int ballot, ml_mask;
//
//    int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
//    int bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);
//
//
//
//    for(;id< old_bucket_num_of_single_table*ELEM_NUM ;id+=step) {
//
//        bucket_t* b1=&old_table[id >> ELEM_NUM_P];
//        sig=b1->sig[id % ELEM_NUM];
//        loc=b1->loc[id % ELEM_NUM];
//        hash=get_hash(sig);
//
//
//        if (sig == 0 && loc == 0) {
////            printf("error, all is zero\n");
//            continue;
//        }
//
//
//        b = &(hash_table[hash % bucket_num_of_single_table]);// bucket num
//
//        /*=====================================================================
//         * The double __syncthreads() seems useless in else, this is to match the two in
//         * if (chosen_simd == simd_lane). As is stated in the paper <Demystifying GPU
//         * Microarchitecture through Microbenchmarking>, the __syncthreads() will not go
//         * wrong if not all threads in one wrap reach it. However, the wraps in the same
//         * block need to reach a __syncthreads(), even if they are not on the same line */
//        /* Check for same signatures in two bucket */
//        ballot = __ballot(bucketk(b,simd_lane) == sig);//find ok
//        /* first half warp(0~15 threads), bit_move = 0
//         * for second half warp(16~31 threads), bit_move = 16 */
//        ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);// whether find
//        if (ballot != 0) { //try updata loc
//            chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
//            if (simd_lane == chosen_simd) {
//                bucketv(b,simd_lane) = loc;
//            }
//            continue;// updata then next
//        }
//
//        /*=====================================================================*/
//        /* Next we try to insert, the while loop breaks if ballot == 0, and the
//         * __syncthreads() in the two loops match if the code path divergent between
//         * the warps in a block. Or some will terminate, or process the next element.
//         * FIXME: if some wrap go to process next element, some stays here, will this
//         * lead to mismatch in __syncthreads()? If it does, we should launch one thread
//         * for each element. God knows what nVidia GPU will behave. FIXME;
//         * Here we write b->loc, and the above code also write b->loc. This will not
//         * lead to conflicts, because here all the signatures are 0, while the aboves
//         * are all non-zero */
//
//        /* Major Location : use last 4 bits of signature */
//        ml_mask = (1 << (sig & ((1 << ELEM_NUM_P) - 1))) - 1;
//        /* find the empty slot for insertion */
//        while (1) {
//            ballot = __ballot(bucketk(b,simd_lane) == 0);
//            ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
//            /* 1010|0011 => 0000 0011 1010 0000, 16 bits to 32 bits*/
//            ballot = ((ballot & ml_mask) << 16) | ((ballot & ~(ml_mask)));
//            if (ballot != 0) {
//                chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
//                if (simd_lane == chosen_simd) {
//                    bucketk(b,simd_lane) = sig;
//                }
//            }
//
//            __syncthreads();
//
//            if (ballot != 0) {
//                if (b->sig[chosen_simd] == sig) {
//                    if (simd_lane == chosen_simd) {
//                        bucketv(b,simd_lane) = loc;
//                    }
//                    goto finish;
//                }
//            } else {
//                break;
//            }
//        }
//        // 以上，先访问位置、判断是否找到、尝试插入到空位置
//
//        /* ==== try next bucket ==== */
//        cuckoo_num = 0;
//
//        cuckoo_evict:
//        second_hash = (((hash ^ sig) & BLOCK_HASH_MASK)
//                       | (hash & ~BLOCK_HASH_MASK)) % bucket_num_of_single_table;
//        b = &(hash_table[second_hash]);
//        /*=====================================================================*/
//        /* Check for same signatures in two bucket */
//        ballot = __ballot(bucketk(b,simd_lane) == sig);
//        /* first half warp(0~15 threads), bit_move = 0
//         * for second half warp(16~31 threads), bit_move = 16 */
//        ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
//        if (0 != ballot) {
//            chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
//            if (simd_lane == chosen_simd) {
//                bucketv(b,simd_lane) = loc;
//            }
//            continue;
//        }
//        // 以上、先判断第二个桶是否存在
//
//        while (1) {
//            ballot = __ballot(bucketk(b,simd_lane) == 0);
//            ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
//            ballot = ((ballot & ml_mask) << 16) | ((ballot & ~(ml_mask)));
//            if (ballot != 0) {// 如果找到空位
//                chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
//            } else { // 没找到
//                /* No available slot.
//                 * Get a Major location between 0 and 15 for insertion */
//                chosen_simd = sig & ((1 << ELEM_NUM_P) - 1);
//                if (cuckoo_num < MAX_CUCKOO_NUM) {
//                    /* record the signature to be evicted */
//                    new_sig = b->sig[chosen_simd];
//                    new_loc = b->loc[chosen_simd];
//                }
//            }
//
//            /* synchronize before the signature is written by others */
//            __syncthreads();
//
//            if (ballot != 0) {
//                if (simd_lane == chosen_simd) {
//                    bucketk(b,simd_lane) = sig;
//                }
//            } else {
//                /* two situations to handle: 1) cuckoo_num < MAX_CUCKOO_NUM,
//                 * replace one element, and reinsert it into its alternative bucket.
//                 * 2) cuckoo_num >= MAX_CUCKOO_NUM.
//                 * The cuckoo evict exceed the maximum insert time, replace the element.
//                 * In each case, we write the signature first.*/
//                if (simd_lane == chosen_simd) {
//                    bucketk(b,simd_lane) = sig;
//                }
//            }
//
//            __syncthreads();
//
//            if (ballot != 0) {
//                /* write the empty slot or try again when conflict */
//                if (b->sig[chosen_simd] == sig) {
//                    if (simd_lane == chosen_simd) {
//                        bucketv(b,simd_lane) = loc;
//                    }
//                    goto finish;
//                }
//            } else {
//                if (cuckoo_num < MAX_CUCKOO_NUM) {
//                    cuckoo_num ++;
//                    if (b->sig[chosen_simd] == sig) {//已经修改 sig 修改loc再驱逐就好了
//                        if (simd_lane == chosen_simd) {
//                            bucketv(b,simd_lane) = loc;
//                        }
//                        sig = new_sig;
//                        loc = new_loc;
//                        goto cuckoo_evict;
//                    } else {//mmm other insert  then give up
//                        /* if there is conflict when writing the signature,
//                         * it has been replaced by another one. Reinserting
//                         * the element is meaningless, because it will evict
//                         * the one that is just inserted. Only one will survive,
//                         * we just give up the failed one */
//                        goto finish;
//                    }
//                } else {
//                    /* exceed the maximum insert time, evict one */
//                    if (b->sig[chosen_simd] == sig) {
//                        if (simd_lane == chosen_simd) {
//                            bucketv(b,simd_lane) = loc;
//                        }
//                    }
//                    /* whether or not succesfully inserted, finish */
//                    goto finish;
//                }
//            }
//        }
//
//        finish:
//        ;
//        //now we get to the next element
//    }
//
//    return;
//}

__global__ void show_table(
        bucket_t		*hash_table,
        TYPE            bucket_num_of_single_table)
{
        for (int num_bucket = 0; num_bucket <bucket_num_of_single_table ; ++num_bucket) {
            printf("\nbucket:%d",num_bucket);
            for (int i = 0; i < ELEM_NUM ; ++i) {
                if(i==0) printf("\n\t\t");
                printf("%d,%d ",getk(hash_table[num_bucket].sigloc[i]),getv(hash_table[num_bucket].sigloc[i]));
            }
        }
}

extern "C" void gpu_show_table(
        bucket_t	*hash_table,
        TYPE        table_size)
{
    int bucket_num_of_single_table=table_size/(ELEM_NUM);
    show_table<<<1,1>>>(hash_table,bucket_num_of_single_table);
}


extern "C" void gpu_hash_search(
		ielem_t 	    *in,
		loc_t		*out,
		bucket_t	*hash_table,
		TYPE        table_size,
		int			num_elem)
{
	int num_blks = 512;
	int num_thread=512;
    int bucket_num_of_single_table=table_size/(ELEM_NUM);

    #if   cuckoo_cu_speed
        GpuTimer timer;
        timer.Start();
    #endif

	hash_search<<<num_blks, num_thread>>>(
			in, out,num_elem, hash_table,bucket_num_of_single_table);

    #if   cuckoo_cu_speed
        timer.Stop();
        double diff = timer.Elapsed()*1000000;
        printf("kernel<<<search>>>the time is %.2lf us ( %.2f Mops)\n",
               (double) diff, (double) (num_elem) / diff);
    #endif

    checkCudaErrors(cudaGetLastError());
	return;
}

/* num_blks is the array size of blk_input and blk_output */
extern "C" void gpu_hash_insert(
		bucket_t	*hash_table,
		TYPE        table_size,
		ielem_t		*input,
		int			elem_num)
{

	int num_thread = THREAD_NUM;
	int num_blks=512;
    int bucket_num_of_single_table=table_size/(ELEM_NUM);

    #if   cuckoo_cu_speed
        GpuTimer timer;
        timer.Start();
    #endif

	hash_insert_cuckoo<<<num_blks, num_thread>>>(
			hash_table,bucket_num_of_single_table, input, elem_num);

    #if   cuckoo_cu_speed
        timer.Stop();
        double  diff = timer.Elapsed()*1000000;
        printf("kernel<<<insert>>>the time is %.2lf us ( %.2f Mops)\n",
               (double) diff, (double) (elem_num) / diff);
    #endif
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
	return;
}

extern "C" void gpu_hash_delete(
		delem_t 	*in,
		bucket_t	*hash_table,
        TYPE        table_size,
		int			num_elem)
{

	int	num_thread=THREAD_NUM;
	int num_blks = 512;
    int bucket_num_of_single_table=table_size/(ELEM_NUM);

    #if   cuckoo_cu_speed
        GpuTimer timer;
        timer.Start();
    #endif

    hash_delete<<<num_blks, num_thread>>>(
			in,num_elem, hash_table,bucket_num_of_single_table );

    #if   cuckoo_cu_speed
        timer.Stop();
        double  diff = timer.Elapsed()*1000000;
        printf("kernel<<<delete>>>the time of %d is %.2lf us ( %.2lf Mops)\n",
               num_elem,(double) diff, (double) (num_elem) / diff);
    #endif

    checkCudaErrors(cudaGetLastError());
    return;
}






extern "C" void gpu_hash_resize(
        int new_size,
        bucket_t* new_table,
        int old_size,
        bucket_t* old_table)
{
    int num_thread = THREAD_NUM;
    int num_blks=512;
    int bucket_num_of_single_table=new_size/(ELEM_NUM);
    int old_bucket_num_of_single_table=old_size/(ELEM_NUM);

    #if   cuckoo_cu_speed
        GpuTimer timer;
        timer.Start();
    #endif

    hash_reisze_cuckoo<<<num_blks, num_thread>>>(
            new_table,bucket_num_of_single_table, old_table, old_bucket_num_of_single_table);

    #if   cuckoo_cu_speed
        timer.Stop();
        double  diff = timer.Elapsed()*1000000;
        printf("kernel<<<rehash>>>the time is %.2lf us ( %.2f Mops)\n",
               (double) diff, (double) (old_size) / diff);
    #endif

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    return;
}

