#ifndef __INTEGRAL_IMAGE_H
#define __INTEGRAL_IMAGE_H

#include <vector>
#include <memory>
#include "cudaLib.cuh"
#include <iostream>
#include <chrono>
#include <string.h>

#ifdef _WIN32
#include <Windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#else
#include <stdarg.h>
static int timeGetTime() {
	return 0;
}
#endif

enum DataType{
	TypeUINT8    = 0,
	TypeINT8     = 1,
	TypeUINT16   = 2,
	TypeINT16    = 3,
	TypeUINT32   = 4,
	TypeINT32    = 5,
	TypeFLOAT32  = 6,
	TypeFLOAT64  = 7,
	TypeAll      = 8,
};

enum SCAN{
	KoggeStone_SCAN = 0,
	LF_SCAN = 1,
};

inline std::string ScanName(int type) {
	const char* name = "error Scan type!";
	switch (type)
	{
	case KoggeStone_SCAN:
		name = "KoggeStone_SCAN";
		break;
	case LF_SCAN:
		name = "LF_SCAN";
		break;
	default:
		return "error Scan type!";
		break;
	}
	return name;
}

static void GetArgs(int argc, char** argv, int& nScanType, int& dType0, int& dType1, int& repeat) {
	if (argc >= 6) {
		nScanType = atoi(argv[2]);
		dType0 = atoi(argv[3]);
		dType1 = atoi(argv[4]);
		repeat = atoi(argv[5]);
	}
}

template<typename dim3> inline
void ShowGridBlockDim(std::string name, dim3& grid, dim3& block) {
	printf("%s,grid=(%d,%d,%d),block=(%d,%d,%d)\n", name.c_str(), grid.x, grid.y, grid.z, block.x, block.y, block.z);
}


/***********************************************:
prefix sum for a array
************************************************/
template<typename TSrc, typename TDst> inline
void IntegralRow(const TSrc* src, TDst* dst, int count) {
	typedef TDst T;
	T sum = 0;
	for (int i = 0; i < count; i++) {
		sum += src[i];
		dst[i] = sum;
	}
}

/***********************************************:
prefix sum for a array, stride can be specified, 
so it can be used to compute prefix sum.
if nStrideSrc == 1 , compute prefix sum in row
if nStrideSrc == ImageWidth, compute prefix in column
************************************************/
template<typename TSrc, typename TDst> inline
void IntegralStride(const TSrc* src, int nStrideSrc, TDst* dst, int nStrideDst, int count) {
	typedef TDst T;
	T sum = 0;
	const TSrc* p0 = src;
	T* p1 = dst;
	for (int i = 0; i < count; i++, p0 += nStrideSrc, p1 += nStrideDst) {
		sum += *p0;
		*p1 = sum;
	}
}

/***********************************************:
naive SAT on CPU. as Paper Alg.1
************************************************/
template<typename TSrc, typename TDst> inline
void IntegralImageSerial(const TSrc* src, TDst* dst, int width, int height) {
	typedef TDst T;
	T sum = 0;

	const TSrc* p0 = src;
	T* p1 = dst;
	T* p2 = dst;
	for (int j = 0; j < width; j++, p0 ++, p1 ++) {
		sum += *p0;
		*p1 = sum;
	}
	for (int i = 1; i < height; i++) {
		sum = 0;
		for (int j = 0; j < width; j++, p0 ++, p1 ++, p2 ++) {
			sum += *p0;
			*p1 = *p2 + sum;
		}
	}
}

/***********************************************:
Parallel compute SAT on CPU by OpenMP
************************************************/
template<typename TSrc, typename TDst> inline
void IntegralImageParallel(const TSrc* src, TDst* dst, int width, int height) {
	typedef TDst T;
	T sum = 0;

	const T* p0 = src;
	T* p1 = dst;

#pragma omp parallel for
	for (int i = 0; i < height; i++) {
		IntegralRow(src + i*width, dst + i*width, width);
	}

#pragma omp parallel for
	for (int i = 0; i < width; i++) {
		IntegralStride(dst + i, width, dst + i, width, height);
	}
}

template<typename TSrc, typename TDst> inline
void IntegralImageAllRow(const TSrc* src, TDst* dst, int width, int height) {
	typedef TDst T;
	T sum = 0;

	const TSrc* p0 = src;
	T* p1 = dst;

#pragma omp parallel for
	for (int i = 0; i < height; i++) {
		IntegralRow(src + i*width, dst + i*width, width);
	}
}


struct Timer {
	float elapsed;
	cudaEvent_t m_start_event;
	cudaEvent_t m_stop_event;
	Timer(){
		elapsed = 0.f;
		cudaEventCreate(&m_start_event);
		cudaEventCreate(&m_stop_event);
	}
	~Timer()
	{
		cudaEventDestroy(m_start_event);
		cudaEventDestroy(m_stop_event);
	}
	void start()
	{
		cudaEventRecord(m_start_event, 0);
	}
	void stop()
	{
		cudaEventRecord(m_stop_event, 0);
		cudaEventSynchronize(m_stop_event);
		cudaEventElapsedTime(&elapsed, m_start_event, m_stop_event);
	}
	float elapsedInMs() const
	{
		return elapsed;
	}
};

inline float getTime() {
#ifdef _WIN32
	return timeGetTime();
#endif
}

template<typename T> inline
float GetAvgTime(const T* src, T* dst, int width, int height, int type) {
	const int N = 6;
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < N; i++) {
		if (type == 0) IntegralImageSerial(src, dst, width, height);
		if (type == 1) IntegralImageParallel(src, dst, width, height);
	}
	auto end = std::chrono::system_clock::now();
	auto diff = end - start;
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / 1000000.f;
	return duration / N;
}

template<typename T>
struct GetDataType {
	static std::string name() {
		return "unknow";
	}
};

template<>
struct GetDataType<uchar> {
	static std::string name() {
		return "uchar";
	}
};

template<>
struct GetDataType<char> {
	static std::string name() {
		return "char";
	}
};

template<>
struct GetDataType<float> {
	static std::string name() {
		return "float32";
	}
};

template<>
struct GetDataType<uint> {
	static std::string name() {
		return "uint32";
	}
};

template<>
struct GetDataType<int> {
	static std::string name() {
		return "int32";
	}
};

template<>
struct GetDataType<double> {
	static std::string name() {
		return "float64";
	}
};



#define DISPLAY_FUNCTION printf("%s:%d:%s\n", __FILE__, __LINE__, __func__);

template<typename T0, typename T1> inline
bool Compare(const T0* p0, const T1* p1, int width, int height) {
	return true;
	for (int i = 0; i < width*height; i++) {
		if (p0[i] != p1[i]) {
			int x = i%width;
			int y = i/width;
			printf("error : %d %d (%d)\n", x, y, (y+1)*(x+1));
			return false;
		}
	}
	return true;
}

template<typename T0, typename T1> inline 
void Transpose(const T0* src, int src_width, int src_height, T1* dst, int dst_width, int dst_height) {
	assert(src_width == dst_height && src_height == dst_width);
	for (int y = 0; y < dst_height; y++) {
		for (int x = 0; x < dst_width; x++) {
			dst[x + y*dst_width] = src[y + x*src_width];
		}
	}
}

inline std::string StringFormat(const char* format, ...) {
	va_list args;
	va_start(args, format);
	char buf[1024 * 4] = "";
	vsprintf(buf, format, args);
	va_end(args);
	return std::string(buf);
}

template<typename T> inline
void SaveToRaw(const char* path, const T* data, int width, int height) {
	DISPLAY_FUNCTION;
	std::cout << "save path:" << path << std::endl;
	FILE* fp = fopen(path, "wb");
	if (fp) {
		fwrite(data, sizeof(T), width*height, fp);
		fclose(fp);
	}
	else {
		std::cout << "open file failed:" << path << std::endl;
	}
}

template<typename T> inline
void SaveToText(const char* path, const T* data, int width, int height) {
	DISPLAY_FUNCTION;
	FILE* fp = fopen(path, "wt");
	std::cout << "save path:" << path << std::endl;
	if (fp) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				fprintf(fp, "%.1f ", (float)data[x + y*width]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	else {
		std::cout << "Open File Failed :" << path << std::endl;
	}
}

static void TestCPU(int width, int height) {
	//int width = 1024*1;
	//int height = 1024*2;
	int size = width*height;
	std::vector<int> src(size), dst1(size), dst2(size);
	std::fill(src.begin(), src.end(), 1);
	{
		float tm1 = GetAvgTime(&src[0], &dst1[0], width, height, 0);
		float tm2 = GetAvgTime(&src[0], &dst2[0], width, height, 1);
		printf("%d, %d, IntegralImageSerial tm1 = %f \n    IntegralImageParallel tm2 = %f\n", width, height, tm1, tm2);
		bool bCmp = Compare(&dst1[0], &dst2[0], width, height);
		printf("compare = %s\n", bCmp?"successed":"failed");
		FILE* flog = fopen("d:/log.csv", "at");
		if(flog){
			fprintf(flog, "%f, ", tm1);
			fclose(flog);
		}
	}

	if (memcmp(&dst1[0], &dst2[0], size*sizeof(dst2[0])) == 0) {
		printf("memcmp is ok\n");
	}
	else {
		printf("memcmp is failed\n");
	}
}

static void Test() {
	for(int i = 1; i < 10; i++){
		TestCPU(i*1024, i*1024);
	}
}


#endif //__INTEGRAL_IMAGE_H
