#include "NearestNeighbor.h"
#include <stdlib.h>
#include <cutil.h>
#include <cutil_inline.h>

static int h_datanum_nn[3], h_dst_total_nn;
__constant__ int d_datanum_nn[3], d_total_nn[1], d_dst_total_nn[1];
__constant__ float d_min_nn[3], d_max_nn[3], d_interval_nn[3];
float *d_xary_nn, *d_yary_nn, *d_zary_nn, *d_data_ary_nn, *d_out_ary_nn;
static int sh_total_nn;

// use by internal 
__device__ inline void NearestNeighbor_GetDistance(float x1, float x2, float y1, float y2, float z1, float z2, float *res)
{
	*res = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
}

__device__ inline void NearestNeighbor_GetNearestPoint(float x, float y, float z, float *res, float *data_ary
						, float *xary, float *yary, float *zary)
{
	float min = 1.0e+38f, now;
	for (int i = 0;i < *d_total_nn;i++)
	{
		NearestNeighbor_GetDistance(x, xary[i], y, yary[i], z, zary[i], &now);
		if (min > now)
		{
			min = now;
			*res = data_ary[i];
		}
	}
}

__device__ inline void NearestNeighbor_GetNearestPointN(float x, float y, float z, float *res, float *data_ary
						, float *xary, float *yary, float *zary, int tid)
{
	float min = 1.0e+38f, now;
	int idx, i, j;
	__shared__ float sh[256][3];
	for ( i = 0; i < *d_total_nn; i+=256)
	{
		idx = i+tid;
		if (tid<256)
		{
			sh[tid][0] = xary[idx];
			sh[tid][1] = yary[idx];
			sh[tid][2] = zary[idx];
		}
        __syncthreads();
		for ( j=0; j<256; j++)
		{
			if (i+j < *d_total_nn)
			{
				NearestNeighbor_GetDistance(x, sh[j][0], y, sh[j][1], z, sh[j][2], &now);
				if (min > now)
				{
					min = now;
					*res = data_ary[i+j];
				}
			}
			else
				break;
		}

		__syncthreads();
	}
}

//not use shared memory
__global__ void NearestNeighbor_GetNearest(float *out_ary, float *data_ary, float *xary, float *yary, float *zary)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int inc = blockDim.x * gridDim.x;
	int xindex, yindex, zindex;
	float x ,y, z, res;
	for (int i = idx ; i < *d_dst_total_nn ; i += inc)
	{
		xindex = i % d_datanum_nn[0];
		yindex = (i / d_datanum_nn[0]) % d_datanum_nn[1];
		zindex = i / (d_datanum_nn[0] * d_datanum_nn[1]);
		x = d_min_nn[0] + xindex * d_interval_nn[0];
		y = d_min_nn[1] + yindex * d_interval_nn[1];
		z = d_min_nn[2] + zindex * d_interval_nn[2];

		NearestNeighbor_GetNearestPoint(x, y, z, &res, data_ary, xary, yary, zary);
		out_ary[i] = res;
	}
}

//use shared memory
__global__ void NearestNeighbor_GetNearestSH(float *out_ary, float *data_ary, float *xary, float *yary, float *zary)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int inc = blockDim.x * gridDim.x;
	int xindex, yindex, zindex;
	float x ,y, z, res;
	for (int i = idx ; i < *d_dst_total_nn ; i += inc)
	{
		xindex = i % d_datanum_nn[0];
		yindex = (i / d_datanum_nn[0]) % d_datanum_nn[1];
		zindex = i / (d_datanum_nn[0] * d_datanum_nn[1]);
		x = d_min_nn[0] + xindex * d_interval_nn[0];
		y = d_min_nn[1] + yindex * d_interval_nn[1];
		z = d_min_nn[2] + zindex * d_interval_nn[2];

		NearestNeighbor_GetNearestPointN(x, y, z, &res, data_ary, xary, yary, zary, threadIdx.x);
		out_ary[i] = res;
	}
}

// use by external, return need float ary size
__host__ int NearestNeighbor_SetData( const InterpolationInfo *h_info )
{
	int sum_of_use_memory = 0;
	// set in data
	sh_total_nn = h_info->m_total;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_total_nn, &sh_total_nn, sizeof(int)));
	int size = sizeof(float)*sh_total_nn;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_xary_nn, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_yary_nn, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_zary_nn, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_ary_nn, size));
	sum_of_use_memory += size*4;
	printf("size of use input data on gpu: %f MB\n", size*4/1024.0/1024.0);
	 // get source memory on gpu
	CUDA_SAFE_CALL(cudaMemcpy(d_xary_nn, h_info->m_posAry[0], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_yary_nn, h_info->m_posAry[1], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_zary_nn, h_info->m_posAry[2], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_ary_nn, h_info->m_data_ary, size, cudaMemcpyHostToDevice));
	// set out data
	const float *h_min = h_info->min, *h_max = h_info->max, *h_interval = h_info->interval;
	for (int i=0;i<3;i++)
		h_datanum_nn[i] = (int)floor((h_max[i]-h_min[i])/h_interval[i])+1;
	h_dst_total_nn = h_datanum_nn[0] * h_datanum_nn[1] * h_datanum_nn[2];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dst_total_nn, &h_dst_total_nn, sizeof(int))); //set out total
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_out_ary_nn, sizeof(float)*h_dst_total_nn)); // get dst memory on gpu.
	sum_of_use_memory += sizeof(float)*h_dst_total_nn;
	int size_float3 = sizeof(float)*3;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_datanum_nn, h_datanum_nn, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_min_nn, h_min, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_max_nn, h_max, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_interval_nn, h_interval, size_float3));
	printf("size of out data on gpu: %f MB\n", sizeof(float)*h_dst_total_nn/1024.0/1024.0);
	printf("size of use memory on gpu: %f MB\n", sum_of_use_memory/1024.0/1024.0);
	return h_dst_total_nn;
}

void NearestNeighbor_ComputeData(_out float *dstdata, int th, bool useShMem)
{
	int threadsPerBlock = 256;
	if (useShMem)
		threadsPerBlock = 256;
	else
		threadsPerBlock = th;
	int blocksPerGrid = (sh_total_nn + threadsPerBlock - 1) / threadsPerBlock;

	if (useShMem)
		NearestNeighbor_GetNearestSH<<<blocksPerGrid, threadsPerBlock>>>
			(d_out_ary_nn, d_data_ary_nn, d_xary_nn, d_yary_nn, d_zary_nn);
	else
		NearestNeighbor_GetNearest<<<blocksPerGrid, threadsPerBlock>>>
			(d_out_ary_nn, d_data_ary_nn, d_xary_nn, d_yary_nn, d_zary_nn);

	cutilCheckMsg("kernel launch failure");
#ifdef _DEBUG
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
#endif
	CUDA_SAFE_CALL( cudaMemcpy(dstdata, d_out_ary_nn, sizeof(float)*h_dst_total_nn, cudaMemcpyDeviceToHost) );
	if (d_xary_nn) cudaFree(d_xary_nn);
	if (d_yary_nn) cudaFree(d_yary_nn);
	if (d_zary_nn) cudaFree(d_zary_nn);
	if (d_data_ary_nn) cudaFree(d_data_ary_nn);
	if (d_out_ary_nn) cudaFree(d_out_ary_nn);
	CUDA_SAFE_CALL( cudaThreadExit() );
}
