//#define __CUDA_ARCH__ 350
#include "cudaLib.cuh"
#include <stdio.h>
#include <vector>
#include <memory>
#include "integralImage.h"

namespace BlockScan {
	template<typename T> __device__ __forceinline__
		void WarpPrefixSumLF(T& val, const uint& laneId, T& data) {
#pragma unroll
		for (int i = 1; i <= 32; i <<= 1) {
			val = __shfl(data, i - 1, i << 1);
			if ((laneId & ((i << 1) - 1)) >= i) {
				data += val;
			}
		}
	}

	/***********************************************:
	Paper ScanRow-BRLT algorithm
	************************************************/	
	static const uint WARP_SIZE = 32;
	template<typename TSrc, typename TDst, int SCAN_TYPE, uint BLOCK_SIZE, uint SMEM_COUNT>
	__global__ void blockScan(const TSrc* __restrict__ dataIn, TDst* dataOut, uint width, uint widthStride, uint height, uint heightStride) {
		uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
		uint tidy = blockIdx.y * blockDim.y + threadIdx.y;
		uint warpId = threadIdx.x >> 5;
		uint laneId = threadIdx.x & 31;
		const uint warpCount = blockDim.x >> 5;

		/***********************************************:
	 	declear 32 registers, BLOCK_SIZE is fixed as 32
		************************************************/		
		TDst data[BLOCK_SIZE], val;
		__shared__ TDst _smem[SMEM_COUNT][BLOCK_SIZE][WARP_SIZE + 1];
		__shared__ TDst smemSum[BLOCK_SIZE];

		auto smem = _smem[0];

		//for (uint y = tidy*BLOCK_SIZE; y < height; y += gridDim.y*BLOCK_SIZE)
		{
			const uint y = tidy*BLOCK_SIZE;
			if (warpId == 0) {
				smemSum[laneId] = 0;
			}
			__syncthreads();
			for (uint x = tidx, cnt = 0; x < width; x += blockDim.x, cnt ++) {
				uint offset = y*widthStride + x;
				#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					if (y + s < height) {
						data[s] = ldg(&dataIn[offset]);
						//data[s] = dataIn[offset];
						offset += widthStride;
						//WarpPrefixSumLF(val, laneId, data[s]);

						if (SCAN_TYPE == KoggeStone_SCAN) {//KoggeStone_SCAN implementation
							#pragma unroll
							for (int i = 1; i <= 32; i <<= 1) {
								/*the first row of the matrix*/
								val = __shfl_up(data[s], i);
								if (laneId >= i) {
									data[s] += val;
								}
							}
						}
						else if (SCAN_TYPE == LF_SCAN) { //LF_SCAN implementation
							#pragma unroll
							for (int i = 1; i <= 32; i <<= 1) {
								val = __shfl(data[s], i - 1, i << 1);
								if ((laneId & ((i << 1) - 1)) >= i) {
									data[s] += val;
								}
							}
						}
						if (laneId == WARP_SIZE - 1) smem[s][warpId] = data[s];
					}
				}
				__syncthreads();

				if (warpId == 0) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						TDst a = smem[s][threadIdx.x];
						//WarpPrefixSumLF(val, laneId, a);
						if (SCAN_TYPE == KoggeStone_SCAN) {
							#pragma unroll
							for (int i = 1; i <= 32; i <<= 1) {
								/*the first row of the matrix*/
								val = __shfl_up(a, i);
								if (laneId >= i) {
									a += val;
								}
							}
						}else if (SCAN_TYPE == LF_SCAN) {
							#pragma unroll
							for (int i = 1; i <= 32; i <<= 1) {
								val = __shfl(a, i - 1, i << 1);
								if ((laneId & ((i << 1) - 1)) >= i) {
									a += val;
								}
							}
						}
						smem[s][threadIdx.x] = a;
					}
				}
				__syncthreads();
	
				if (warpId > 0) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						data[s] += smem[s][warpId - 1];
					}
				}
				//__syncthreads();
				if (cnt > 0) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						data[s] += smemSum[s];
					}
				}
				__syncthreads();
				if (threadIdx.x == blockDim.x - 1) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						smemSum[s] = data[s];
					}
				}
				__syncthreads();
#if 0
#if 0
				offset = x*heightStride + y;
				#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					dataOut[offset + s] = data[s];
				}
				//__syncthreads();
#else
				offset = y*widthStride + x;
				#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					if (y + s < height) {
						dataOut[offset] = data[s];
						offset += widthStride;
					}
				}
#endif
#else
				/***********************************************:
	 			Block-Register-Local-Transpose, as paper Alg.5
				************************************************/
				for (int k = 0; k < warpCount; k += SMEM_COUNT) {
					if (warpId >= k && warpId < k + SMEM_COUNT) {
						auto csMem = _smem[warpId-k];
						assert(warpId >= k);
#pragma unroll
						for (int s = 0; s < BLOCK_SIZE; s++) {
							csMem[s][laneId] = data[s];
						}
#pragma unroll
						for (int s = 0; s < BLOCK_SIZE; s++) {
							data[s] = csMem[laneId][s];
						}
					}
					__syncthreads();
				}
				uint _x = y & (~uint(31));
				uint _y = x & (~uint(31));
				offset = _y*heightStride + _x;
#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					dataOut[offset+laneId] = data[s];
					offset += heightStride;
				}
				__syncthreads();
#endif
			}
		}
	}

	template<typename TSrc, typename TDst, int SCAN_TYPE>
	static void Test(int width, int height) {
		DISPLAY_FUNCTION;
		std::cout << GetDataType<TSrc>::name() << "-->" << GetDataType<TDst>::name() <<
			", ScanName="<< ScanName(SCAN_TYPE)<<std::endl;
		std::cout << "begin : ScanRow-BRLT" << std::endl;
		const int REPEAT_COUNT = 1;
		float inc = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//typedef float DataType;

		const int SMEM_COUNT = 8;
		const uint BLOCK_SIZE = 32;
		const uint BLOCK_DIM_X = 1024 * sizeof(int) / sizeof(TDst);
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

		DevStream SM;
		dim3 block_size(BLOCK_DIM_X, 1);
		dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE));
		dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
		ShowGridBlockDim("BlockScan::blockScan 0", grid_size1, block_size);
		ShowGridBlockDim("BlockScan::blockScan 1", grid_size2, block_size);
		float tm = 0;
		tm = timeGetTime();
		cudaEventRecord(start, 0);
		#pragma unroll
		for (int k=0; k<REPEAT_COUNT; k ++){
				BlockScan::blockScan<TSrc, TDst, SCAN_TYPE, BLOCK_SIZE, SMEM_COUNT * sizeof(uint) / sizeof(TDst)> << <grid_size1, block_size, 0, SM.stream >> > (devA.GetData(), devTmp.GetData(), width, devA.DataPitch(), devA.height, devTmp.DataPitch());
				BlockScan::blockScan<TDst, TDst, SCAN_TYPE, BLOCK_SIZE, SMEM_COUNT * sizeof(uint) / sizeof(TDst)> << <grid_size2, block_size, 0, SM.stream >> > (devTmp.GetData(), devB.GetData(), devTmp.width, devTmp.DataPitch(), devTmp.height, devB.DataPitch());
		}
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		CUDA_CHECK_ERROR;


		tm = timeGetTime() - tm;

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&inc, start, stop);

		devB.CopyToHost(&vecB[0], width, width, height);
		printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);

		{
			std::vector<TDst> vecTmp(size);
			IntegralImageSerial(&vecA[0], &vecTmp[0], width, height);
			bool bCmp = Compare(&vecB[0], &vecTmp[0], width, height);
			printf("compare = %s\n", bCmp ? "successed" : "failed");
		}


		//SaveToRaw(StringFormat("./%d-%d.raw", width, height).c_str(), &vecB[0], width, height);
		//SaveToText("./tmp.txt", &vecTmp[0], devTmp.width, devTmp.height);
		//SaveToText("./vecB.txt", &vecB[0], width, height);

		//cudaSyncDevice();
		std::cout << "end : ScanRow-BRLT" << std::endl;
	}
};


//ScanRow-BRLT
void TestBlockScan(int argc, char** argv){
	std::cout << "------------------------------------------------------" << std::endl;
	int nScanType = SCAN::LF_SCAN;
	int dType0 = DataType::TypeFLOAT32;
	int dType1 = DataType::TypeFLOAT32;
	int repeat = 1;
#define Scan BlockScan
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
