#include "NearestNeighbor.h"
#include <stdlib.h>
#include <cutil.h>
#include <cutil_inline.h>

static int h_datanum[3], h_dst_total;
__constant__ int d_datanum[3], d_total[1], d_dst_total[1];
__constant__ float d_min[3], d_max[3], d_interval[3];
float *d_xary, *d_yary, *d_zary, *d_data_ary, *d_out_ary;
static int sh_total;

// use by internal 
__device__ inline void NearestNeighbor_GetDistance(float x1, float x2, float y1, float y2, float z1, float z2, float *res)
{
	*res = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
}

__device__ inline void NearestNeighbor_GetNearestPoint(float x, float y, float z, float *res, float *data_ary
						, float *xary, float *yary, float *zary)
{
	float min = 1.0e+38f, now;
	for (int i = 0;i < *d_total;i++)
	{
		NearestNeighbor_GetDistance(x, xary[i], y, yary[i], z, zary[i], &now);
		if (min > now)
		{
			min = now;
			*res = data_ary[i];
		}
	}
}

__global__ void NearestNeighbor_GetNearest(float *out_ary, float *data_ary, float *xary, float *yary, float *zary)
{
	//int tid = threadIdx.x;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int inc = blockDim.x * gridDim.x;
	int xindex, yindex, zindex;
	float x ,y, z, res;
	for (int i = idx ; i < *d_dst_total ; i += inc)
	{
		xindex = i % d_datanum[0];
		yindex = (i / d_datanum[0]) % d_datanum[1];
		zindex = i / (d_datanum[0] * d_datanum[1]);
		x = d_min[0] + xindex * d_interval[0];
		y = d_min[1] + yindex * d_interval[1];
		z = d_min[2] + zindex * d_interval[2];
		NearestNeighbor_GetNearestPoint(x, y, z, &res, data_ary, xary, yary, zary);
		out_ary[i] = res;
	}
}

// use by external, return need float ary size
__host__ int NearestNeighbor_SetData( const InterpolationInfo *h_info )
{
	int sum_of_use_memory = 0;
	// set in data
	sh_total = h_info->total;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_total, &sh_total, sizeof(int)));
	int size = sizeof(float)*sh_total;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_xary, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_yary, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_zary, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_ary, size));
	sum_of_use_memory += size*4;
	printf("size of use input data on gpu: %d bytes\n", size*4);
	 // get source memory on gpu
	CUDA_SAFE_CALL(cudaMemcpy(d_xary, h_info->posary[0], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_yary, h_info->posary[1], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_zary, h_info->posary[2], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_ary, h_info->data_ary, size, cudaMemcpyHostToDevice));
	// set out data
	const float *h_min = h_info->min, *h_max = h_info->max, *h_interval = h_info->interval;
	for (int i=0;i<3;i++)
		h_datanum[i] = floor((h_max[i]-h_min[i])/h_interval[i])+1;
	h_dst_total = h_datanum[0] * h_datanum[1] * h_datanum[2];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dst_total, &h_dst_total, sizeof(int))); //set out total
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_out_ary, sizeof(float)*h_dst_total)); // get dst memory on gpu.
	sum_of_use_memory += sizeof(float)*h_dst_total;
	int size_float3 = sizeof(float)*3;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_datanum, h_datanum, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_min, h_min, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_max, h_max, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_interval, h_interval, size_float3));
	printf("size of out data on gpu: %d bytes\n", sizeof(float)*h_dst_total);
	printf("size of use memory on gpu: %d bytes\n", sum_of_use_memory);
	return h_dst_total;
}

void NearestNeighbor_ComputeData(_out float *dstdata)
{
	int threadsPerBlock = THREADS;
	int blocksPerGrid = (sh_total + threadsPerBlock - 1) / threadsPerBlock;
	NearestNeighbor_GetNearest<<<blocksPerGrid, threadsPerBlock>>>
		(d_out_ary, d_data_ary, d_xary, d_yary, d_zary);
	cutilCheckMsg("kernel launch failure");
#ifdef _DEBUG
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
#endif
	CUDA_SAFE_CALL( cudaMemcpy(dstdata, d_out_ary, sizeof(float)*h_dst_total, cudaMemcpyDeviceToHost) );
	if (d_xary) cudaFree(d_xary);
	if (d_yary) cudaFree(d_yary);
	if (d_zary) cudaFree(d_zary);
	if (d_data_ary) cudaFree(d_data_ary);
	if (d_out_ary) cudaFree(d_out_ary);
	CUDA_SAFE_CALL( cudaThreadExit() );
}
