
#include "cudaLib.cuh"
#include <stdio.h>
#include <vector>
#include <memory>
#include "integralImage.h"

namespace IncrementScan {

	static const int WARP_SIZE = 32;
	static const int BLOCK_SIZE = WARP_SIZE;

	/************************************************************
	ScanColumn algorithm
	************************************************************/
	template<typename TSrc, typename TDst, uint SCAN_TYPE, uint COUNT_PER_THREAD, uint WARP_COUNT>
	__global__ void ScanY(const TSrc* __restrict dataIn, TDst* dataOut, uint width, uint widthStride, uint height)
	{
		__shared__ TDst sMem[WARP_COUNT + 1][WARP_SIZE];
		uint warpId = threadIdx.y;
		uint laneId = threadIdx.x;
		uint tidx = blockIdx.x*blockDim.x + threadIdx.x;
		uint tidy = (blockIdx.y*blockDim.y + threadIdx.y)*COUNT_PER_THREAD;
		uint PROCESS_COUNT_Y = COUNT_PER_THREAD*blockDim.y;

		if (tidx >= width)
			return;

		TDst data[COUNT_PER_THREAD];
		for (uint y = tidy; y < height; y += PROCESS_COUNT_Y) {
			if (y != tidy) {
				if (warpId == WARP_COUNT - 1)
					sMem[0][laneId] = data[COUNT_PER_THREAD - 1];
				__syncthreads();
			}
			uint index = y*widthStride + tidx;
			{
				//1,load data
				uint yy = y;
				uint idx = index;
				#pragma unroll
				for (int i = 0; i < COUNT_PER_THREAD; i++) {
					if (yy < height) {
						data[i] = ldg(&dataIn[idx]);
						idx += widthStride;
						yy++;
					}
				}
			}
			{
				//2, increament prefix sum
				#pragma unroll
				for (int i = 1; i < COUNT_PER_THREAD; i++) {
					data[i] += data[i - 1];
				}
				sMem[warpId + 1][laneId] = data[COUNT_PER_THREAD - 1];
				__syncthreads();
			}
			{
#if 1
				//can be improved
				if (warpId == 0) {
#if 1
					TDst s = 0;
					if (y != 0) {
						s = sMem[0][laneId];
					}
					#pragma unroll
					for (int i = 1; i < WARP_COUNT + 1; i++) {
						s += sMem[i][laneId];
						sMem[i][laneId] = s;
					}
#else
					if (y != 0) {
						sMem[1][laneId] += sMem[0][laneId];
					}
					//#pragma unroll
					for (int i = 2; i < WARP_COUNT + 1; i++) {
						sMem[i][laneId] += sMem[i - 1][laneId];
					}
#endif
				}
#else
				for (uint wid = warpId; wid < WARP_SIZE; wid += WARP_COUNT) {
					TDst s = 0;
					if (laneId < WARP_COUNT)
						s = sMem[laneId + 1][wid];
					if (y > tidy && laneId == 0) {
						s += sMem[0][wid];
					}

					if (SCAN_TYPE == KoggeStone_SCAN) {
						#pragma unroll
						for (int i = 1; i <= 32; i <<= 1) {
							/*the first row of the matrix*/
							const TDst val = __shfl_up(s, i);
							if (laneId >= i) {
								s += val;
							}
						}
					}else if (SCAN_TYPE == LF_SCAN) {
						#pragma unroll
						for (int i = 1; i <= 32; i <<= 1) {
							const TDst val = __shfl(s, i - 1, i << 1);
							if ((laneId & ((i << 1) - 1)) >= i) {
								s += val;
							}
						}
					}
					if (laneId < WARP_COUNT)
						sMem[laneId + 1][wid] = s;
				}
#endif
				__syncthreads();
			}

			if (y != 0) {
				TDst sum = sMem[warpId][laneId];
				#pragma unroll
				for (int i = 0; i < COUNT_PER_THREAD; i++) {
					data[i] += sum;
				}
			}
			{
				//store
				uint yy = y;
				#pragma unroll
				for (int i = 0; i < COUNT_PER_THREAD; i++) {
					if (yy < height) 
					{
						dataOut[index] = data[i];
						index += widthStride;
						yy++;
					}
				}
			}
			__syncthreads();
		}
	}

	/************************************************************
	ScanRow algorithm
	************************************************************/
	template<typename TSrc, typename TDst, int SCAN_TYPE, int BLOCK_DIM_X, int BLOCK_DIM_Y, int COUNT_PER_THREAD>
	__global__ void ScanX(const TSrc* __restrict dataIn, TDst* dataOut, uint width, uint widthStride, uint height) {
		TDst data[COUNT_PER_THREAD];
		uint warpIdX = threadIdx.x / WARP_SIZE;
		uint warpIdY = threadIdx.y;
		uint laneId = threadIdx.x & (WARP_SIZE - 1);
		const uint WARP_COUNT_X = BLOCK_DIM_X / WARP_SIZE;
		const uint WARP_COUNT_Y = BLOCK_DIM_Y;
		const uint WARP_PROCESS_COUNT_X = WARP_SIZE*COUNT_PER_THREAD;
		const uint BLOCK_PROCWSS_COUNT_X = COUNT_PER_THREAD*BLOCK_DIM_X;
		uint tidx = WARP_PROCESS_COUNT_X*warpIdX + laneId;
		uint tidy = (blockIdx.y*blockDim.y + threadIdx.y);
		__shared__ TDst sMem[WARP_COUNT_Y][WARP_COUNT_X + 1];

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
						const TDst sum = __shfl(data[j - 1], WARP_SIZE - 1);
						if (laneId == 0) {
							data[j] += sum;
						}
					}

					if (SCAN_TYPE == KoggeStone_SCAN) { //KoggeStone_SCAN algorithm
						#pragma unroll
						for (int i = 1; i <= 32; i <<= 1) {
							/*the first row of the matrix*/
							const TDst val = __shfl_up(data[j], i);
							if (laneId >= i) {
								data[j] += val;
							}
						}
					}else if (SCAN_TYPE == LF_SCAN) { //LF_Scan algorithm
						#pragma unroll
						for (int i = 1; i <= 32; i <<= 1) {
							const TDst val = __shfl(data[j], i - 1, i << 1);
							if ((laneId & ((i << 1) - 1)) >= i) {
								data[j] += val;
							}
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
					TDst s = 0;
					if (laneId < WARP_COUNT_X)
						s = sMem[warpIdY][laneId + 1];
					if (x != tidx && laneId == 0)
						s += sMem[warpIdY][0];
					if (SCAN_TYPE == KoggeStone_SCAN) 
					{
						#pragma unroll
						for (int i = 1; i <= 32; i <<= 1) {
							/*the first row of the matrix*/
							const TDst val = __shfl_up(s, i);
							if (laneId >= i) {
								s += val;
							}
						}
					}else if (SCAN_TYPE == LF_SCAN) {
						#pragma unroll
						for (int i = 1; i <= 32; i <<= 1) {
							const TDst val = __shfl(s, i - 1, i << 1);
							if ((laneId & ((i << 1) - 1)) >= i) {
								s += val;
							}
						}
					}
					if (laneId < WARP_COUNT_X)
						sMem[warpIdY][laneId + 1] = s;
				}
				__syncthreads();
			}
			{
				if (x >= WARP_PROCESS_COUNT_X) {
					TDst sum = sMem[warpIdY][warpIdX];
					#pragma unroll
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

//	void TestX(int width, int height) {
//		std::cout << __FUNCTION__ << std::endl;
//		std::cout << "begin : TestIncrementScan" << std::endl;
//		float inc = 0;
//		cudaEvent_t start, stop;
//		cudaEventCreate(&start);
//		cudaEventCreate(&stop);
//
//		typedef float DataType;
//
//		const uint THREAD_COUNT_PER_BLOCK = 1024;
//		const int BLOCK_DIM_X = 512;
//		const int BLOCK_DIM_Y = THREAD_COUNT_PER_BLOCK / BLOCK_DIM_X;
//		const int COUNT_PER_THREAD = 4;
//
//		//const uint BLOCK_SIZE = 32;
//
//		//const uint BLOCK_DIM_X = 256 * 4;
//		//int width = 1024 * 2;
//		//int height = 1024 * 2;
//		int size = width*height;
//		std::vector<DataType> vecA(size), vecB(size);
//		//for (int i = 0; i < height-16; i += 32) std::fill(vecA.begin()+i*width, vecA.begin() + (i+16)*width, 1);
//
//		std::fill(vecA.begin(), vecA.end(), 1);
//
//
//		DevData<DataType> devA(width, height), devB(width, height), devTmp(height, width);
//		devA.CopyFromHost(&vecA[0], width, width, height);
//
//		DevStream SM;
//		//const int PROCESS_COUNT_PER_THREAD_Y = 32;
//		//const int WARP_COUNT = THREAD_COUNT_PER_BLOCK / WARP_SIZE;
//		const dim3 block_sizeX(BLOCK_DIM_X, BLOCK_DIM_Y);
//		dim3 grid_sizeX(1, UpDivide(height, block_sizeX.y));
//
//		//dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE));
//		//dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
//		float tm = 0;
//		//tm = timeGetTime();
//		cudaEventRecord(start, 0);
//		IncrementScan::ScanX<DataType, BLOCK_DIM_X, BLOCK_DIM_Y, COUNT_PER_THREAD> << <grid_sizeX, block_sizeX >> > (devA.GetData(), devB.GetData(), width, devA.DataPitch(), height);
//		//IncrementScan::IncrementScan<DataType, BLOCK_SIZE, 4 * sizeof(uint) / sizeof(DataType), BLOCK_DIM_X> << <grid_size1, block_size, 0, SM.stream >> > (devA.GetData(), devTmp.GetData(), width, width, height, height);
//		//IncrementScan::IncrementScan<DataType, BLOCK_SIZE, 4 * sizeof(uint) / sizeof(DataType), BLOCK_DIM_X> << <grid_size2, block_size, 0, SM.stream >> > (devTmp.GetData(), devB.GetData(), height, height, width, width);
//		cudaDeviceSynchronize();
//		cudaEventRecord(stop, 0);
//		//CUDA_CHECK_ERROR;
//
//
//		//tm = timeGetTime() - tm;
//
//		cudaEventSynchronize(stop);
//		cudaEventElapsedTime(&inc, start, stop);
//
//		devB.CopyToHost(&vecB[0], width, width, height);
//		printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);
//		//cudaSyncDevice();
//		std::cout << "end : TestSerielScan" << std::endl;
//#if 0
//		FILE* fp = fopen("d:/ints.raw", "wb");
//		if (fp) {
//			fwrite(&vecB[0], sizeof(vecB[0]), width*height, fp);
//			fclose(fp);
//		}
//#endif
//		FILE* flog = fopen("d:/log.csv", "wt");
//		if (flog) {
//			for (int i = 0; i < vecB.size(); i++) {
//				DataType* p = &vecB[0];
//				fprintf(flog, "%.2f ", p[i]);
//				if (i % width == (width - 1))
//					fprintf(flog, "\n");
//				fflush(flog);
//			}
//			fclose(flog);
//		}
//
//
//	}
//	void TestY(int width, int height) {
//		DISPLAY_FUNCTION;
//		std::cout << __FUNCTION__ << std::endl;
//		std::cout << "begin : TestIncrementScan" << std::endl;
//		float inc = 0;
//		cudaEvent_t start, stop;
//		cudaEventCreate(&start);
//		cudaEventCreate(&stop);
//
//		typedef float DataType;
//
//		//const uint BLOCK_SIZE = 32;
//		const uint THREAD_COUNT_PER_BLOCK = 1024;
//		//const uint BLOCK_DIM_X = 256 * 4;
//		//int width = 1024 * 2;
//		//int height = 1024 * 2;
//		int size = width*height;
//		std::vector<DataType> vecA(size), vecB(size);
//		//for (int i = 0; i < height-16; i += 32) std::fill(vecA.begin()+i*width, vecA.begin() + (i+16)*width, 1);
//
//		std::fill(vecA.begin(), vecA.end(), 1);
//
//
//		DevData<DataType> devA(width, height), devB(width, height), devTmp(height, width);
//		devA.CopyFromHost(&vecA[0], width, width, height);
//
//		DevStream SM;
//		const int PROCESS_COUNT_PER_THREAD_Y = 32;
//		const int WARP_COUNT = THREAD_COUNT_PER_BLOCK / WARP_SIZE;
//		const dim3 block_sizeY(WARP_SIZE, WARP_COUNT);
//		dim3 grid_sizeY(UpDivide(width, block_sizeY.x), 1);
//
//		//dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE));
//		//dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
//		float tm = 0;
//		//tm = timeGetTime();
//		cudaEventRecord(start, 0);
//		IncrementScan::ScanY<DataType, PROCESS_COUNT_PER_THREAD_Y, WARP_COUNT> << <grid_sizeY, block_sizeY >> > (devA.GetData(), devB.GetData(), width, devA.DataPitch(), height);
//		//IncrementScan::IncrementScan<DataType, BLOCK_SIZE, 4 * sizeof(uint) / sizeof(DataType), BLOCK_DIM_X> << <grid_size1, block_size, 0, SM.stream >> > (devA.GetData(), devTmp.GetData(), width, width, height, height);
//		//IncrementScan::IncrementScan<DataType, BLOCK_SIZE, 4 * sizeof(uint) / sizeof(DataType), BLOCK_DIM_X> << <grid_size2, block_size, 0, SM.stream >> > (devTmp.GetData(), devB.GetData(), height, height, width, width);
//		cudaDeviceSynchronize();
//		cudaEventRecord(stop, 0);
//		//CUDA_CHECK_ERROR;
//
//
//		//tm = timeGetTime() - tm;
//
//		cudaEventSynchronize(stop);
//		cudaEventElapsedTime(&inc, start, stop);
//
//		devB.CopyToHost(&vecB[0], width, width, height);
//		printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);
//		//cudaSyncDevice();
//		std::cout << "end : TestSerielScan" << std::endl;
//#if 0
//		FILE* fp = fopen("d:/ints.raw", "wb");
//		if (fp) {
//			fwrite(&vecB[0], sizeof(vecB[0]), width*height, fp);
//			fclose(fp);
//		}
//#endif
//#if 1
//		FILE* flog = fopen("d:/log.csv", "wt");
//		if (flog) {
//			for (int i = 0; i < vecB.size(); i++) {
//				DataType* p = &vecB[0];
//				fprintf(flog, "%.2f ", p[i]);
//				if (i % width == (width - 1))
//					fprintf(flog, "\n");
//				fflush(flog);
//			}
//			fclose(flog);
//		}
//#endif
//	}
	template<typename TSrc, typename TDst, int SCAN_TYPE>
	void Test(int width, int height) {
		DISPLAY_FUNCTION;
		std::cout << GetDataType<TSrc>::name() << "-->" << GetDataType<TDst>::name() <<
			", ScanName=" << ScanName(SCAN_TYPE) << std::endl;
		const int REPEAT_COUNT = 1;
		float inc = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//typedef float DataType;

		dim3 block_sizeX, grid_sizeX;
		//X
		const uint THREAD_COUNT_PER_BLOCK = 1024*sizeof(float)/sizeof(TDst);
		const int BLOCK_DIM_X = 32;
		const int BLOCK_DIM_Y = THREAD_COUNT_PER_BLOCK / BLOCK_DIM_X;
		const int COUNT_PER_THREAD = 32;
		block_sizeX = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);
		grid_sizeX = dim3(1, UpDivide(height, block_sizeX.y));

		//Y
		dim3 block_sizeY, grid_sizeY;
		const int PROCESS_COUNT_PER_THREAD_Y = 32*1;
		const int WARP_COUNT = THREAD_COUNT_PER_BLOCK / WARP_SIZE;
		block_sizeY = dim3(WARP_SIZE, WARP_COUNT);
		grid_sizeY = dim3(UpDivide(width, block_sizeY.x), 1);

		int size = width*height;
		std::vector<TSrc> vecA(size);
		std::vector<TDst> vecB(size);
		//for (int i = 0; i < height-16; i += 32) std::fill(vecA.begin()+i*width, vecA.begin() + (i+16)*width, 1);

#if 0
		std::fill(vecA.begin(), vecA.end(), 1);
#else
		for (int i = 0; i < vecA.size(); i++) vecA[i] = (TSrc)(abs(rand()) % 2);
#endif

		DevData<TSrc> devA(width, height);
		DevData<TDst> devB(width, height), devTmp(height, width);
		devA.CopyFromHost(&vecA[0], width, width, height);
		ShowGridBlockDim("ScanX", grid_sizeX, block_sizeX);
		ShowGridBlockDim("ScanY", grid_sizeY, block_sizeY);
		float tm = 0;

		tm = timeGetTime();
		cudaEventRecord(start, 0);
		#pragma unroll
		for (int k=0; k<REPEAT_COUNT; k ++){
				IncrementScan::ScanX<TSrc, TDst, SCAN_TYPE, BLOCK_DIM_X, BLOCK_DIM_Y, COUNT_PER_THREAD> << <grid_sizeX, block_sizeX >> > (devA.GetData(), devTmp.GetData(), width, devA.DataPitch(), height);
				IncrementScan::ScanY<TDst, TDst, SCAN_TYPE, PROCESS_COUNT_PER_THREAD_Y, WARP_COUNT> << <grid_sizeY, block_sizeY >> > (devTmp.GetData(), devB.GetData(), width, devA.DataPitch(), height);
		}
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		CUDA_CHECK_ERROR;

		tm = timeGetTime() - tm;

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&inc, start, stop);

		devB.CopyToHost(&vecB[0], width, width, height);
		printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);
		//cudaSyncDevice();
		{
			std::vector<TDst> vecTmp(size);
			IntegralImageSerial(&vecA[0], &vecTmp[0], width, height);
			bool bCmp = Compare(&vecB[0], &vecTmp[0], width, height);
			printf("compare = %s\n", bCmp ? "successed" : "failed");
		}
		//SaveToRaw(StringFormat("./%d-%d.raw", width, height).c_str(), &vecB[0], width, height);
		//SaveToText("./tmp.txt", &vecTmp[0], devTmp.width, devTmp.height);
		//SaveToText("./vecB.txt", &vecB[0], width, height);
	}
};


//ScanRowColumn
void TestIncreamentScan(int argc, char** argv) {
	std::cout << "------------------------------------------------------" << std::endl;
	int nScanType = SCAN::LF_SCAN;
	int dType0 = DataType::TypeFLOAT32;
	int dType1 = DataType::TypeFLOAT32;
	int repeat = 1;
#define Scan IncrementScan
	GetArgs(argc, argv, nScanType, dType0, dType1, repeat);
#if 0
	//for (int i=0; i<10; i++)
	Scan::Test<uint, double, SCAN::LF_SCAN>(1024 * 1, 1024 * 1);
#else
	for (int i = 1; i <= repeat; i++) {
		int size = i * 1024;
		if (nScanType == SCAN::KoggeStone_SCAN) {
			if (dType0 == DataType::TypeUINT8) {
				if (dType1 == DataType::TypeUINT32) {
					Scan::Test<uchar, uint, SCAN::KoggeStone_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT32) {
					Scan::Test<uchar, float, SCAN::KoggeStone_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<uchar, double, SCAN::KoggeStone_SCAN>(size, size);
				}
			}if (dType0 == DataType::TypeINT8) {
				if (dType1 == DataType::TypeINT32) {
					Scan::Test<char, int, SCAN::KoggeStone_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT32) {
					Scan::Test<char, float, SCAN::KoggeStone_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<char, double, SCAN::KoggeStone_SCAN>(size, size);
				}
			}
			if (dType0 == DataType::TypeINT32) {
				if (dType1 == DataType::TypeINT32) {
					Scan::Test<int, int, SCAN::KoggeStone_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT32) {
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<int, double, SCAN::KoggeStone_SCAN>(size, size);
				}
			}
			else if (dType0 == DataType::TypeFLOAT32) {
				if (dType1 == DataType::TypeUINT32) {
				}
				else if (dType1 == DataType::TypeFLOAT32) {
					Scan::Test<float, float, SCAN::KoggeStone_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<float, double, SCAN::KoggeStone_SCAN>(size, size);
				}
			}
			else if (dType0 == DataType::TypeFLOAT64) {
				if (dType1 == DataType::TypeUINT32) {
				}
				else if (dType1 == DataType::TypeFLOAT32) {
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<double, double, SCAN::KoggeStone_SCAN>(size, size);
				}
			}
		}
		else {
			if (dType0 == DataType::TypeUINT8) {
				if (dType1 == DataType::TypeUINT32) {
					Scan::Test<uchar, uint, SCAN::LF_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT32) {
					Scan::Test<uchar, float, SCAN::LF_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<uchar, double, SCAN::LF_SCAN>(size, size);
				}
			}if (dType0 == DataType::TypeINT8) {
				if (dType1 == DataType::TypeINT32) {
					Scan::Test<char, int, SCAN::LF_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT32) {
					Scan::Test<char, float, SCAN::LF_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<char, double, SCAN::LF_SCAN>(size, size);
				}
			}
			if (dType0 == DataType::TypeINT32) {
				if (dType1 == DataType::TypeINT32) {
					Scan::Test<int, int, SCAN::LF_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT32) {
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<int, double, SCAN::LF_SCAN>(size, size);
				}
			}
			else if (dType0 == DataType::TypeFLOAT32) {
				if (dType1 == DataType::TypeUINT32) {
				}
				else if (dType1 == DataType::TypeFLOAT32) {
					Scan::Test<float, float, SCAN::LF_SCAN>(size, size);
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<float, double, SCAN::LF_SCAN>(size, size);
				}
			}
			else if (dType0 == DataType::TypeFLOAT64) {
				if (dType1 == DataType::TypeUINT32) {
				}
				else if (dType1 == DataType::TypeFLOAT32) {
				}
				else if (dType1 == DataType::TypeFLOAT64) {
					Scan::Test<double, double, SCAN::LF_SCAN>(size, size);
				}
			}
		}
	}
#undef Scan
#endif
	std::cout << "------------------------------------------------------" << std::endl;;
}
