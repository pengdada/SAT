
#include "cudaLib.cuh"
#include <stdio.h>
#include <vector>
#include <memory>
#include "integralImage.h"

namespace IncrementScanY {
	static const int WARP_SIZE = 32;
	static const int BLOCK_SIZE = WARP_SIZE;
	template<typename T, uint BLOCK_SIZE, uint BLOCK_DIM_X, uint BLOCK_DIM_Y>
	__global__ void ScanY(const T* __restrict dataIn, T* dataOut, uint width, uint widthStride, uint height, uint heightStride) {
		__shared__ T smem[BLOCK_SIZE][WARP_SIZE];
		__shared__ T smemSum[BLOCK_SIZE];
		//auto smem = _smem[0];

		uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
		uint tidy = (blockIdx.y * blockDim.y + threadIdx.y)*BLOCK_SIZE;
		uint warpId = threadIdx.y;
		uint laneId = threadIdx.x & 31;
		const uint warpCount = BLOCK_DIM_Y;
		const uint BLOCK_PROCESS_COUNT = BLOCK_DIM_Y*BLOCK_SIZE;

		T data[BLOCK_SIZE];

		//for (uint y = tidy*BLOCK_SIZE; y < height; y += gridDim.y*BLOCK_SIZE)
		{
			const uint x = tidx;
			if (warpId == 0) {
				smemSum[laneId] = 0;
			}
			__syncthreads();
			for (uint y = tidy, cnt = 0; y < height; y += BLOCK_PROCESS_COUNT, cnt++) {
				{
					uint offset = y*widthStride + x;
					uint yy = y;
#pragma unroll
					for (uint s = 0; s < BLOCK_SIZE; s++) {
						if (yy < height) {
							data[s] = ldg(&dataIn[offset]);
							offset += widthStride;
							yy++;
						}
					}
				}/*
				 //rotate
				 #pragma unroll
				 for (int k = 0; k < warpCount; k += SMEM_COUNT) {
				 if (warpId >= k && warpId < k + SMEM_COUNT) {
				 auto csMem = _smem[warpId - k];
				 assert(warpId >= k);
				 #pragma unroll
				 for (uint s = 0; s < BLOCK_SIZE; s++) {
				 csMem[s][laneId] = data[s];
				 }
				 #pragma unroll
				 for (uint s = 0; s < BLOCK_SIZE; s++) {
				 data[s] = csMem[laneId][s];
				 }
				 }
				 __syncthreads();
				 }*/
				{
					#pragma unroll
					for (uint s = 1; s < BLOCK_SIZE; s++) {
						data[s] += data[s - 1];
					}
					//__syncthreads();
				}
				smem[warpId][laneId] = data[BLOCK_SIZE - 1];
				__syncthreads();

				if (warpId == 0) {
					#pragma unroll
					for (uint s = 1; s < BLOCK_SIZE; s++) {
						smem[s][laneId] += smem[s - 1][laneId];
					}
				}
				__syncthreads();
				if (warpId > 0) {
					T sum = smem[warpId - 1][laneId];
#pragma unroll
					for (uint s = 0; s < BLOCK_SIZE; s++) {
						data[s] += sum;
					}
				}
				//__syncthreads();
				if (cnt > 0) {
					T sum = smemSum[laneId];
#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						data[s] += sum;
					}
				}
				//__syncthreads();

				if (warpId == WARP_SIZE - 1) {
					smemSum[laneId] = data[BLOCK_SIZE - 1];
				}
				//__syncthreads();
#if 0
				uint _x = y & (~uint(31));
				uint _y = x & (~uint(31));
				offset = _y*heightStride + _x;
#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					dataOut[offset + laneId] = data[s];
					offset += heightStride;
				}
#else
				{
					uint offset = y*widthStride + x;
					uint yy = y;
#pragma unroll
					for (uint s = 0; s < BLOCK_SIZE; s++) {
						if (yy < height) {
							dataOut[offset] = data[s];
							offset += widthStride;
							yy++;
						}
					}
				}
#endif
				__syncthreads();
			}
		}
	}
	void TestY(int width, int height) {
		DISPLAY_FUNCTION;
		std::cout << "begin : IncrementScanY::TestY" << std::endl;
		float inc = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		typedef float DataType;

		const uint BLOCK_SIZE = 32;
		const uint BLOCK_DIM_X = 32;
		const uint BLOCK_DIM_Y = 32;
		//int width = 1024 * 2;
		//int height = 1024 * 2;
		int size = width*height;
		std::vector<DataType> vecA(size), vecB(size);
		//for (int i = 0; i < height-16; i += 32) std::fill(vecA.begin()+i*width, vecA.begin() + (i+16)*width, 1);

		std::fill(vecA.begin(), vecA.end(), 1);


		DevData<DataType> devA(width, height), devB(width, height), devTmp(height, width);
		devA.CopyFromHost(&vecA[0], width, width, height);

		DevStream SM;
		const dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y);
		dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE*block_size.y));
		dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
		float tm = 0;
		tm = timeGetTime();
		cudaEventRecord(start, 0);
		IncrementScanY::ScanY<DataType, BLOCK_SIZE, BLOCK_DIM_X, BLOCK_DIM_Y> << <grid_size1, block_size>> > (devA.GetData(), devB.GetData(), width, width, height, height);

		//IncrementScanY::ScanY<DataType, BLOCK_SIZE, BLOCK_DIM_X, BLOCK_DIM_Y> << <grid_size1, block_size>> > (devA.GetData(), devTmp.GetData(), width, width, height, height);
		//IncrementScanY::ScanY<DataType, BLOCK_SIZE, BLOCK_DIM_X, BLOCK_DIM_Y> << <grid_size1, block_size>> > (devTmp.GetData(), devB.GetData(), height, height, width, width);
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		CUDA_CHECK_ERROR;


		tm = timeGetTime() - tm;

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&inc, start, stop);

		devB.CopyToHost(&vecB[0], width, width, height);
		printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);
		//cudaSyncDevice();
		std::cout << "end : TestSerielScan" << std::endl;
		//FILE* fp = fopen("d:/ints.raw", "wb");
		//if (fp) {
		//	fwrite(&vecB[0], sizeof(vecB[0]), width*height, fp);
		//	fclose(fp);
		//}
		//FILE* flog = fopen("d:/log.csv", "wt");
		//std::vector<char> buf;
		//if (flog) {
		//	for (int i = 0; i < vecB.size(); i++) {
		//		DataType* p = &vecB[0];
		//		fprintf(flog, "%.2f ", p[i]);
		//		if (i % width == (width - 1))
		//			fprintf(flog, "\n");
		//		fflush(flog);
		//	}
		//	fclose(flog);
		//}
	}
};

void TestIncrementScanY() {
	std::cout << "------------------------------------------------------" << std::endl;
	for (int i = 0; i < 1; i++)
		IncrementScanY::TestY(1024 * 4, 1024 * 4);
	return;

	std::cout << "------------------------------------------------------" << std::endl;;
}