#ifndef __MycudaLib_H
#define __MycudaLib_H

#include <cuda.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <assert.h>

#define uint   unsigned int
#define ushort unsigned short
#define uchar  unsigned char

#define MY_CUDA_CHECK_ERROR 
#define CUDA_CHECK_ERROR 

#define WarpSize 32
__device__ __host__ __forceinline__ int Align(int x, int y) 
{ return (x + y - 1) / y*y; }
__device__ __host__ __forceinline__ int UpDivide(int x, int y) 
{ return (x + y - 1) / y; }
__device__ __host__ __forceinline__ int UpRound(int x, int y) 
{ return UpDivide(x, y)*y; }

template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}


template<class T, int DIM>
struct GpuDataPtr {
	size_t width;
	size_t data_pitch;
	size_t height;
	size_t depth;
	T* data;
};

struct GpuStream{
	cudaStream_t stream;
};


template<class T>
struct GpuDataRef {
	inline GpuDataRef() {
		width = 0; 
		height = 0; 
		depth = 0; 
		data_pitch = 0;
		data = 0;
	}
	inline GpuDataRef(T* ptr, size_t ww, size_t pitch, size_t hh, size_t dd) {
		data = ptr; 
		width = ww; 
		height = hh; 
		depth = dd; 
		data_pitch = pitch;
	}
	inline GpuDataRef& operator=(const GpuDataRef& dtRef) {
		data = dtRef.data;
		width = dtRef.width;
		height = dtRef.height;
		depth = dtRef.depth; 
		data_pitch = dtRef.data_pitch;
		return *this;
	}
	inline void Zero() {
		cudaMemset2D(data, data_pitch * sizeof(*data), 0, width * sizeof(*data), height*depth);
	}
	inline void CopyToHost(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d) const {
		cudaMemcpy2D(ptr, sizeof(*ptr)*_pitch, data, data_pitch * sizeof(*data), width * sizeof(*data), _h*_d, cudaMemcpyDeviceToHost);
	}
	inline void CopyFromHost(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d) {
		cudaMemcpy2D(data, data_pitch * sizeof(*data), ptr, sizeof(*ptr)*_pitch, _w * sizeof(*data), _h*_d, cudaMemcpyHostToDevice);
	}
	T* data;
	size_t width;
	size_t data_pitch;
	size_t height;
	size_t depth;
};


template<class T>
struct GpuData {
	typedef T DataType;
	typedef GpuDataRef<T> DataRef;
	inline virtual ~GpuData() {
		this->FreeBuffer();
	}
	inline GpuData() :width(0), height(0), depth(0), channels(0) {
		memset(&dev_data, 0, sizeof(dev_data));
		memset(&extent, 0, sizeof(extent));
	}
	inline GpuData(size_t w, size_t h = 1, size_t d = 1) : width(0), height(0), depth(0), channels(0) {
		memset(&dev_data, 0, sizeof(dev_data));
		memset(&extent, 0, sizeof(extent));
		MallocBuffer(w, h, d);
	}
	inline GpuData& operator=(const GpuData& obj) {
		MallocBuffer(obj.width, obj.height, obj.depth);
		GetDataRef().CopyFromDevice(obj.GetData(), obj.width, obj.DataPitch(), obj.height, obj.depth, cudaStreamDefault);
		return *this;
	}
	inline DataRef GetDataRef() const {
		return DataRef(GetData(), width, DataPitch(), height, depth);
	}
	inline GpuData& MallocBuffer(size_t w, size_t h, size_t d) {
		if (width == w && height == h && depth == d) {
		} else {
			//reuse the device aligned buffer
			if (sizeof(T)*w < this->dev_data.pitch && h*d > 0 && h*d <= this->extent.height*this->extent.depth) {
				width = w;
				height = h;
				depth = d;
			}
			else {
				FreeBuffer();
				extent = make_cudaExtent(sizeof(T)*w, h, d);
				cudaMalloc3D(&dev_data, extent);
				CUDA_CHECK_ERROR;
				width = w;
				height = h;
				depth = d;
				if (d == 1) {
					if (h == 1) channels = 1;
					else        channels = 2;
				}
				else           channels = 3;
			}
		}
		return *this;
	}
	inline GpuData& MallocBuffer(size_t w, size_t h) {
		return MallocBuffer(w, h, 1);
	}
	inline GpuData& MallocBuffer(size_t w) {
		return MallocBuffer(w, 1, 1);
	}
	inline void FreeBuffer() {
		if (dev_data.ptr) {
			cudaFree(dev_data.ptr);
		}
		memset(&dev_data, 0, sizeof(dev_data));
		memset(&extent, 0, sizeof(extent));
		width = height = depth = 0;
		channels = 0;
	}
	inline void CopyToHost(T* ptr, size_t ww, size_t _pitch, size_t hh, size_t dd = 1) {
		GetDataRef().CopyToHost(ptr, ww, _pitch, hh, dd);
	}
	inline void CopyToDevice(T* ptr, size_t ww, size_t _pitch, size_t hh, size_t dd = 1) {
		GetDataRef().CopyToDevice(ptr, ww, _pitch, hh, dd);
	}
	inline void CopyFromHost(const T* ptr, size_t ww, size_t _pitch, size_t hh, size_t dd = 1) {
		GetDataRef().CopyFromHost(ptr, ww, _pitch, hh, dd);
	}
	inline void CopyFromDevice(const T* ptr, size_t ww, size_t _pitch, size_t hh, size_t dd = 1) {
		GetDataRef().CopyFromDevice(ptr, ww, _pitch, hh, dd);
	}
	inline GpuData& Zero() {
		GetDataRef().Zero();
		return *this;
	}
	inline size_t DataPitch() const {
		return dev_data.pitch / sizeof(T);
	}
	inline T* GetData() const {
		return (T*)dev_data.ptr;
	}
	cudaPitchedPtr dev_data;
	cudaExtent extent;
	size_t width;
	size_t height;
	size_t depth;
	size_t channels;
};

#define DevData GpuData
#define CUDA_CHECK_ERROR MY_CUDA_CHECK_ERROR
#define DevStream GpuStream

#if (CUDART_VERSION >= 9000)
#pragma message("CUDART_VERSION >= 9000")
#define __my_shfl_up(var, delta) __shfl_up_sync(0xFFFFFFFF, var, delta)
#define __my_shfl_down(var, delta) __shfl_down_sync(0xFFFFFFFF, var, delta)
#define __my_shfl(var, srcLane) __shfl_sync(0xFFFFFFFF, var, srcLane)
#else
#pragma message("CUDART_VERSION < 9000")
#define __my_shfl_up(var, delta) __shfl_up(var, delta)
#define __my_shfl_down(var, delta) __shfl_down(var, delta)
#define __my_shfl(var, srcLane) __shfl(var, srcLane)
#endif

#endif //__MycudaLib_H
