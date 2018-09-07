#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

#include <string>
#include <iostream>

class GpuTimer {
	cudaEvent_t start;
	cudaEvent_t stop;

public:

	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start(cudaStream_t stream=0) {
		cudaEventRecord(start, stream);
	}

	void Stop(cudaStream_t stream=0) {
		cudaEventRecord(stop, stream);
	}

	float Elapsed() {
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		/// return second
		return elapsed / 1000;
	}

};

#endif  /* __GPU_TIMER_H__ */
