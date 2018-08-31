
#include "cudaLib.cuh"
#include <stdio.h>
#include <vector>
#include <memory>
#include "integralImage.h"

namespace IncrementScanX {

	static const int WARP_SIZE = 32;
	static const int BLOCK_SIZE = WARP_SIZE;
	template<typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y, int COUNT_PER_THREAD>
	__global__ void ScanX(const T* __restrict dataIn, T* dataOut, uint width, uint widthStride, uint height) {
		T data[COUNT_PER_THREAD];
		uint warpIdX = threadIdx.x / WARP_SIZE;
		uint warpIdY = threadIdx.y;
		uint laneId = threadIdx.x & (WARP_SIZE - 1);
		const uint WARP_COUNT_X = BLOCK_DIM_X / WARP_SIZE;
		const uint WARP_COUNT_Y = BLOCK_DIM_Y;
		const uint WARP_PROCESS_COUNT_X = WARP_SIZE*COUNT_PER_THREAD;
		const uint BLOCK_PROCWSS_COUNT_X = COUNT_PER_THREAD*BLOCK_DIM_X;
		uint tidx = WARP_PROCESS_COUNT_X*warpIdX + laneId;
		uint tidy = (blockIdx.y*blockDim.y + threadIdx.y);
		__shared__ T sMem[WARP_COUNT_Y][WARP_COUNT_X + 1];

		for (uint x = tidx; x < width; x += BLOCK_PROCWSS_COUNT_X) {
			if (x != tidx) {
				if (threadIdx.x == 0)
					sMem[warpIdY][0] = sMem[warpIdY][WARP_COUNT_X];				
				__syncthreads();
			}
			uint index = tidy*widthStride + x;
			{
				//1, load data
				uint xx = x;
				uint idx = index;
#pragma unroll
				for (int i = 0; i < COUNT_PER_THREAD; i++) {
					if (xx < width) {
						data[i] = dataIn[idx];
						idx += WARP_SIZE;
						xx += WARP_SIZE;
					}
				}
			}
			{
				//2, scan x
#pragma unroll
				for (int j = 0; j < COUNT_PER_THREAD; j++) {
					if (j > 0) {
						const T sum = __shfl(data[j - 1], WARP_SIZE - 1);
						if (laneId == 0) {
							data[j] += sum;
						}
					}
#pragma unroll
					for (int i = 1; i <= 32; i <<= 1) {
						/*the first row of the matrix*/
						const T sum = __shfl_up(data[j], i);
						if (laneId >= i) {
							data[j] += sum;
						}
					}
					if (laneId == WARP_SIZE - 1) {
						sMem[warpIdY][warpIdX + 1] = data[COUNT_PER_THREAD - 1];
					}
				}
				__syncthreads();
			}
			{
				//scan partial sum
				if (warpIdX == 0) {
					T s = 0;
					if (laneId < WARP_COUNT_X)
						s = sMem[warpIdY][laneId + 1];
					if (x != tidx && laneId == 0)
						s += sMem[warpIdY][0];
#pragma unroll
					for (int i = 1; i <= 32; i <<= 1) {
						/*the first row of the matrix*/
						const T sum = __shfl_up(s, i);
						if (laneId >= i) {
							s += sum;
						}
					}
					if (laneId < WARP_COUNT_X)
						sMem[warpIdY][laneId + 1] = s;
				}
				__syncthreads();
			}
			{
				if (x >= WARP_PROCESS_COUNT_X) {
					T sum = sMem[warpIdY][warpIdX];
					for (int i = 0; i < COUNT_PER_THREAD; i++) {
						data[i] += sum;
					}
				}
			}
			{
				//save data
				uint xx = x;
				uint idx = index;
#pragma unroll
				for (int i = 0; i < COUNT_PER_THREAD; i++) {
					if (xx < width) {
						dataOut[idx] = data[i];
						idx += WARP_SIZE;
						xx += WARP_SIZE;
					}
				}
			}
		}
	}

	void TestX(int width, int height) {
		DISPLAY_FUNCTION;
		std::cout << __FUNCTION__ << std::endl;
		std::cout << "begin : TestIncrementScan" << std::endl;
		float inc = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		typedef float DataType;

		const uint THREAD_COUNT_PER_BLOCK = 1024;
		const int BLOCK_DIM_X = 32;
		const int BLOCK_DIM_Y = THREAD_COUNT_PER_BLOCK / BLOCK_DIM_X;
		const int COUNT_PER_THREAD = 32;

		//const uint BLOCK_SIZE = 32;

		//const uint BLOCK_DIM_X = 256 * 4;
		//int width = 1024 * 2;
		//int height = 1024 * 2;
		int size = width*height;
		std::vector<DataType> vecA(size), vecB(size);
		//for (int i = 0; i < height-16; i += 32) std::fill(vecA.begin()+i*width, vecA.begin() + (i+16)*width, 1);

		std::fill(vecA.begin(), vecA.end(), 1);


		DevData<DataType> devA(width, height), devB(width, height), devTmp(height, width);
		devA.CopyFromHost(&vecA[0], width, width, height);

		DevStream SM;
		//const int PROCESS_COUNT_PER_THREAD_Y = 32;
		//const int WARP_COUNT = THREAD_COUNT_PER_BLOCK / WARP_SIZE;
		const dim3 block_sizeX(BLOCK_DIM_X, BLOCK_DIM_Y);
		dim3 grid_sizeX(1, UpDivide(height, block_sizeX.y));

		//dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE));
		//dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
		float tm = 0;
		//tm = timeGetTime();
		cudaEventRecord(start, 0);
		IncrementScanX::ScanX<DataType, BLOCK_DIM_X, BLOCK_DIM_Y, COUNT_PER_THREAD> << <grid_sizeX, block_sizeX >> > (devA.GetData(), devB.GetData(), width, devA.DataPitch(), height);
		//IncrementScan::IncrementScan<DataType, BLOCK_SIZE, 4 * sizeof(uint) / sizeof(DataType), BLOCK_DIM_X> << <grid_size1, block_size, 0, SM.stream >> > (devA.GetData(), devTmp.GetData(), width, width, height, height);
		//IncrementScan::IncrementScan<DataType, BLOCK_SIZE, 4 * sizeof(uint) / sizeof(DataType), BLOCK_DIM_X> << <grid_size2, block_size, 0, SM.stream >> > (devTmp.GetData(), devB.GetData(), height, height, width, width);
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		//CUDA_CHECK_ERROR;


		//tm = timeGetTime() - tm;

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&inc, start, stop);

		devB.CopyToHost(&vecB[0], width, width, height);
		printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);
		//cudaSyncDevice();
		std::cout << "end : TestSerielScan" << std::endl;
#if 0
		FILE* fp = fopen("d:/ints.raw", "wb");
		if (fp) {
			fwrite(&vecB[0], sizeof(vecB[0]), width*height, fp);
			fclose(fp);
		}
#endif
		FILE* flog = fopen("d:/log.csv", "wt");
		if (flog) {
			for (int i = 0; i < vecB.size(); i++) {
				DataType* p = &vecB[0];
				fprintf(flog, "%.2f ", p[i]);
				if (i % width == (width - 1))
					fprintf(flog, "\n");
				fflush(flog);
			}
			fclose(flog);
		}
	}
};



void TestIncreamentScanX() {
	std::cout << "------------------------------------------------------" << std::endl;
	for (int i = 0; i < 1; i++)
		IncrementScanX::TestX(1024 * 1, 1024 * 1);
	return;
	std::cout << "------------------------------------------------------" << std::endl;;
}