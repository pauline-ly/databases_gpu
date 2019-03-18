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

#ifndef _LIBGPUHASH_H_
#define _LIBGPUHASH_H_

#include <cuda_runtime.h>
#include "gpu_hash.h"

extern "C" void gpu_hash_search(
        ielem_t 	    *in,
        loc_t		*out,
        bucket_t	*hash_table,
        TYPE        table_size,
        int			num_elem);

extern "C" void gpu_hash_insert(
        bucket_t	*hash_table,
        TYPE table_size,
        ielem_t		*input,
        int			elem_num);

extern "C" void gpu_hash_delete(
        delem_t 	*in,
        bucket_t	*hash_table,
        TYPE        table_size,
        int			num_elem);

extern "C" void gpu_hash_resize(
		int new_size,
		bucket_t* new_table,
		int old_size,
		bucket_t* old_table);

extern "C" void gpu_show_table(
        bucket_t	*hash_table,
        TYPE        table_size);

#define CUDA_SAFE_CALL(call) do {                                            \
	cudaError_t err = call;                                                    \
	if(cudaSuccess != err) {                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
				 __FILE__, __LINE__, cudaGetErrorString( err) );             \
		exit(EXIT_FAILURE);                                                  \
	} } while (0)

#define CUDA_SAFE_CALL_SYNC(call) do {                                       \
	CUDA_SAFE_CALL_NO_SYNC(call);                                            \
	cudaError_t err |= cudaDeviceSynchronize();                                \
	if(cudaSuccess != err) {                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
				__FILE__, __LINE__, cudaGetErrorString( err) );              \
		exit(EXIT_FAILURE);                                                  \
	} } while (0)
#endif
