
#include "cudaLib.cuh"
#include <stdio.h>
#include <vector>
#include <memory>
#include "integralImage.h"

namespace SerielScan {

	static const int WARP_SIZE = 32;
	static const int BLOCK_SIZE = WARP_SIZE;
	/***********************************************:
	 BRLT-ScanRow Alg.
	************************************************/
	template<typename TSrc, typename TDst, uint BLOCK_SIZE, uint SMEM_COUNT, uint BLOCK_DIM_X>
	__global__ void serielScan(const TSrc* __restrict dataIn, TDst* dataOut, uint width, uint widthStride, uint height, uint heightStride) {
		__shared__ TDst _smem[SMEM_COUNT][BLOCK_SIZE][WARP_SIZE + 1];
		__shared__ TDst smemSum[BLOCK_SIZE];
		auto smem = _smem[0];

		uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
		uint tidy = blockIdx.y * blockDim.y + threadIdx.y;
		uint warpId = threadIdx.x >> 5;
		uint laneId = threadIdx.x & 31;
		const uint warpCount = BLOCK_DIM_X >> 5;

		/***********************************************:
	 	declear 32 registers, BLOCK_SIZE is fixed as 32
		************************************************/
		TDst data[BLOCK_SIZE];

		//for (uint y = tidy*BLOCK_SIZE; y < height; y += gridDim.y*BLOCK_SIZE) 
		{
			const uint y = tidy*BLOCK_SIZE;
			if (warpId == 0) {
				smemSum[laneId] = 0;
			}
			__syncthreads();
			for (uint x = tidx, cnt = 0; x < width; x += blockDim.x, cnt++) {
				uint offset = y*widthStride + x;
				{
					uint _y = y;
#pragma unroll
					for (uint s = 0; s < BLOCK_SIZE; s++) {
						if (_y < height) {
							data[s] = ldg(&dataIn[offset]);
							offset += widthStride;
							_y++;
						}
					}
				}
				/***********************************************:
	 			Block-Register-Local-Transpose, as paper Alg.5
				************************************************/
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
				}
				{
					TDst sum = data[0];
#pragma unroll
					for (uint s = 1; s < BLOCK_SIZE; s++) {
						sum += data[s];
						data[s] = sum;
					}
					__syncthreads();
				}
				smem[warpId][laneId] = data[BLOCK_SIZE - 1];
				__syncthreads();

				if (warpId == 0) {
					TDst sum = smem[0][laneId];
#pragma unroll
					for (uint s = 1; s < warpCount; s++) {
						sum += smem[s][laneId];
						smem[s][laneId] = sum;
					}
				}
				__syncthreads();
				if (warpId > 0) {
					TDst sum = smem[warpId - 1][laneId];
#pragma unroll
					for (uint s = 0; s < BLOCK_SIZE; s++) {
						data[s] += sum;
					}
				}
				__syncthreads();
				if (cnt > 0) {
					TDst sum = smemSum[laneId];
#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						data[s] += sum;
					}
				}
				__syncthreads();

				if (warpId == warpCount - 1) {
					smemSum[laneId] = data[BLOCK_SIZE - 1];
				}
				__syncthreads();

				uint _x = y & (~uint(31));
				uint _y = x & (~uint(31));
				offset = _y*heightStride + _x;
#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					if (_y < width) {
						dataOut[offset + laneId] = data[s];
						offset += heightStride;
					}
				}
				__syncthreads();
			}
		}
	}
	template<typename TSrc, typename TDst, int SCAN_TYPE>
	void TestX(int width, int height) {
		DISPLAY_FUNCTION;
		std::cout << GetDataType<TSrc>::name() << "-->" << GetDataType<TDst>::name() <<
			", ScanName=" << ScanName(SCAN_TYPE) << std::endl;
		std::cout << "begin : BRLT-ScanRow" << std::endl;
		float inc = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		typedef TDst DataType;

		const uint BLOCK_SIZE = 32;
		const uint BLOCK_DIM_X = 256 * 4 * sizeof(int) / sizeof(TDst);
		//int width = 1024 * 2;
		//int height = 1024 * 2;
		int size = width*height;
		std::vector<TSrc> vecA(size);
		std::vector<TDst> vecTmp(size);
		//for (int i = 0; i < height-16; i += 32) std::fill(vecA.begin()+i*width, vecA.begin() + (i+16)*width, 1);

		std::fill(vecA.begin(), vecA.end(), 1);


		DevData<TSrc> devA(width, height);
		DevData<TDst> devTmp(height, width);
		devA.CopyFromHost(&vecA[0], width, width, height);

		//DevStream SM;
		dim3 block_size(BLOCK_DIM_X, 1);
		dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE));
		dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
		float tm = 0;
		tm = timeGetTime();
		cudaEventRecord(start, 0);
		SerielScan::serielScan<TSrc, TDst, BLOCK_SIZE, 16 * sizeof(uint) / sizeof(DataType), BLOCK_DIM_X> << <grid_size1, block_size >> > (devA.GetData(), devTmp.GetData(), width, devA.DataPitch(), height, devTmp.DataPitch());
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		CUDA_CHECK_ERROR;

		tm = timeGetTime() - tm;

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&inc, start, stop);

		devTmp.CopyToHost(&vecTmp[0], devTmp.width, devTmp.width, devTmp.height);
		printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);
		//cudaSyncDevice();
		{
			std::vector<DataType> vecCmpTmp(size), vecCmp(size);
			IntegralImageAllRow(&vecA[0], &vecCmpTmp[0], width, height);
			Transpose(&vecCmpTmp[0], width, height, &vecCmp[0], height, width);
			bool bCmp = Compare(&vecTmp[0], &vecCmp[0], height, width);
			printf("compare = %s\n", bCmp ? "successed" : "failed");
		}
		std::cout << "end : BRLT-ScanRow" << std::endl;

		//SaveToRaw(StringFormat("./%d-%d.raw", width, height).c_str(), &vecB[0], width, height);
		//SaveToText("./tmp.txt", &vecTmp[0], devTmp.width, devTmp.height);
		//SaveToText("./vecB.txt", &vecB[0], width, height);

	}

	template<typename TSrc, typename TDst, int SCAN_TYPE>
	void Test(int width, int height) {
		DISPLAY_FUNCTION;
		std::cout << GetDataType<TSrc>::name() << "-->" << GetDataType<TDst>::name() <<
			", ScanName=" << ScanName(SCAN_TYPE) << std::endl;

		const int REPEAT_COUNT=1;
		float inc = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//typedef TDst DataType;

		const int SMEM_COUNT = 8;
		const uint BLOCK_SIZE = 32;
		const uint BLOCK_DIM_X = 256 * 4 * sizeof(int) / sizeof(TDst);
		//int width = 1024 * 2;
		//int height = 1024 * 2;
		int size = width*height;
		std::vector<TSrc> vecA(size);
		std::vector<TDst> vecB(size), vecTmp(size);

		srand(time(NULL));
#if 0
		std::fill(vecA.begin(), vecA.end(), 1);
#else
		for (int i = 0; i < vecA.size(); i++) vecA[i] = (TSrc)(abs(rand()) % 2);
#endif

		DevData<TSrc> devA(width, height);
		DevData<TDst> devB(width, height), devTmp(height, width);
		devA.CopyFromHost(&vecA[0], width, width, height);

		//DevStream SM;
		dim3 block_size(BLOCK_DIM_X, 1);
		dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE));
		dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
		float tm = 0;
		ShowGridBlockDim("BRLT-ScanRow 0", grid_size1, block_size);
		ShowGridBlockDim("BRLT-ScanRow 1", grid_size2, block_size);
		tm = timeGetTime();
		cudaEventRecord(start, 0);
		#pragma unroll
		for (int k=0; k<REPEAT_COUNT; k ++){
				SerielScan::serielScan<TSrc, TDst, BLOCK_SIZE, SMEM_COUNT * sizeof(uint) / sizeof(TDst), BLOCK_DIM_X> << <grid_size1, block_size >> > (devA.GetData(), devTmp.GetData(), width, devA.DataPitch(), height, devTmp.DataPitch());
				SerielScan::serielScan<TDst, TDst, BLOCK_SIZE, SMEM_COUNT * sizeof(uint) / sizeof(TDst), BLOCK_DIM_X> << <grid_size2, block_size >> > (devTmp.GetData(), devB.GetData(), devTmp.width, devTmp.DataPitch(), devB.width, devB.DataPitch());
		}
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		CUDA_CHECK_ERROR;


		tm = timeGetTime() - tm;

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&inc, start, stop);

		devB.CopyToHost(&vecB[0], width, width, height);
		devTmp.CopyToHost(&vecTmp[0], devTmp.width, devTmp.width, devTmp.height);
		printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);
		//cudaSyncDevice();
		{
			std::vector<TDst> vecCmp(size);
			IntegralImageSerial(&vecA[0], &vecCmp[0], width, height);
			bool bCmp = Compare(&vecB[0], &vecCmp[0], width, height);
			printf("compare = %s\n", bCmp ? "successed" : "failed");
		}
		std::cout << "end : BRLT-ScanRow" << std::endl;

		//SaveToRaw(StringFormat("./%d-%d.raw", width, height).c_str(), &vecB[0], width, height);
		//SaveToText("./tmp.txt", &vecTmp[0], devTmp.width, devTmp.height);
		//SaveToText("./vecB.txt", &vecB[0], width, height);

	}
};


//BRLT-Scan
void TestSerielScan(int argc, char** argv) {
	std::cout << "------------------------------------------------------" << std::endl;
#if 0
	//SerielScan::TestX(1024 * 5, 1024 * 5);
	SerielScan::Test<uchar, float>(1024 * 1, 1024 * 1);
#else
	std::cout << "------------------------------------------------------" << std::endl;
	int nScanType = SCAN::LF_SCAN;
	int dType0 = DataType::TypeFLOAT32;
	int dType1 = DataType::TypeFLOAT32;
	int repeat = 1;
#define Scan SerielScan
	GetArgs(argc, argv, nScanType, dType0, dType1, repeat);

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
#endif
#undef Scan
	std::cout << "------------------------------------------------------" << std::endl;;
}
