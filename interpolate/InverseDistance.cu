#include "InverseDistance.h"
#include <stdlib.h>
#include <cutil.h>
#include <cutil_inline.h>

static int h_datanum_id[3], h_dst_total_id;
__constant__ int d_datanum_id[3], d_total_id[1], d_dst_total_id[1];
__constant__ float d_min_id[3], d_max_id[3], d_interval_id[3], d_power_id[1];
float *d_xary_id, *d_yary_id, *d_zary_id, *d_data_ary_id, *d_out_ary_id;
static int sh_total_id;

// use by internal 
__device__ inline void InverseDistance_GetDistance(float x1, float x2, float y1, float y2, float z1, float z2, float *res)
{
	*res = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
}

__device__ inline void InverseDistance_GetNearestPoint(float x, float y, float z, float *res, float *data_ary
						, float *xary, float *yary, float *zary)
{
	float now, sum = 0, sum2 = 0, tmp;
	for (int i = 0;i < *d_total_id;i++)
	{
		InverseDistance_GetDistance(x, xary[i], y, yary[i], z, zary[i], &now);
		if (now<0.001)
		{
			*res = data_ary[i];
			return ;
		}
		tmp = pow(now, -d_power_id[0]);
		sum += tmp;
		sum2 += tmp*data_ary[i];
	}
	sum2 /= sum;
	*res = sum2;
}

__device__ inline void InverseDistance_GetNearestPointN(float x, float y, float z, float *res, float *data_ary
						, float *xary, float *yary, float *zary, int tid)
{
	float now, sum = 0, sum2 = 0, tmp;
	int idx, i, j;

	__shared__ float sh[256][4];
	for ( i = 0;i < *d_total_id;i+=256)
	{
		idx = i+tid;
        __syncthreads();
		if (tid<256)
		{
			sh[tid][0] = xary[idx];
			sh[tid][1] = yary[idx];
			sh[tid][2] = zary[idx];
			sh[tid][3] = data_ary[idx];
		}
        __syncthreads();
		for ( j=0; j<256 && i+j<*d_total_id; j++)
		{
			InverseDistance_GetDistance(x, sh[j][0], y, sh[j][1], z, sh[j][2], &now);

			if (now<0.001)
			{
				*res = sh[j][3];
				return;
			}
			tmp = pow(now, -d_power_id[0]);
			sum += tmp;
			sum2 += tmp*sh[j][3];
		}
	}
	sum2 /= sum;
	*res = sum2;
}

//not use shared memory
__global__ void InverseDistance_GetNearest(float *out_ary, float *data_ary, float *xary, float *yary, float *zary)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int inc = blockDim.x * gridDim.x;
	int xindex, yindex, zindex;
	float x ,y, z, res;
	for (int i = idx ; i < *d_dst_total_id ; i += inc)
	{
		xindex = i % d_datanum_id[0];
		yindex = (i / d_datanum_id[0]) % d_datanum_id[1];
		zindex = i / (d_datanum_id[0] * d_datanum_id[1]);
		x = d_min_id[0] + xindex * d_interval_id[0];
		y = d_min_id[1] + yindex * d_interval_id[1];
		z = d_min_id[2] + zindex * d_interval_id[2];
		InverseDistance_GetNearestPoint(x, y, z, &res, data_ary, xary, yary, zary);
		out_ary[i] = res;
	}
}

//use shared memory
__global__ void InverseDistance_GetNearestSH(float *out_ary, float *data_ary, float *xary, float *yary, float *zary)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int inc = blockDim.x * gridDim.x;
	int xindex, yindex, zindex;
	float x ,y, z, res;
	for (int i = idx ; i < *d_dst_total_id ; i += inc)
	{
		xindex = i % d_datanum_id[0];
		yindex = (i / d_datanum_id[0]) % d_datanum_id[1];
		zindex = i / (d_datanum_id[0] * d_datanum_id[1]);
		x = d_min_id[0] + xindex * d_interval_id[0];
		y = d_min_id[1] + yindex * d_interval_id[1];
		z = d_min_id[2] + zindex * d_interval_id[2];
		InverseDistance_GetNearestPointN(x, y, z, &res, data_ary, xary, yary, zary,threadIdx.x);
		out_ary[i] = res;
	}
}

// use by external, return need float ary size
__host__ int InverseDistance_SetData( const InterpolationInfo *h_info, float power )
{
	int sum_of_use_memory = 0;
	// set in data
	sh_total_id = h_info->m_total;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_total_id, &sh_total_id, sizeof(int)));
	int size = sizeof(float)*sh_total_id;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_xary_id, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_yary_id, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_zary_id, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_ary_id, size));
	sum_of_use_memory += size*4;
	printf("size of use input data on gpu: %f MB\n", size*4/1024.0/1024.0);
	 // get source memory on gpu
	CUDA_SAFE_CALL(cudaMemcpy(d_xary_id, h_info->m_posAry[0], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_yary_id, h_info->m_posAry[1], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_zary_id, h_info->m_posAry[2], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_ary_id, h_info->m_data_ary, size, cudaMemcpyHostToDevice));
	// set out data
	const float *h_min = h_info->min, *h_max = h_info->max, *h_interval = h_info->interval;
	for (int i=0;i<3;i++)
		h_datanum_id[i] = (int)floor((h_max[i]-h_min[i])/h_interval[i])+1;
	h_dst_total_id = h_datanum_id[0] * h_datanum_id[1] * h_datanum_id[2];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dst_total_id, &h_dst_total_id, sizeof(int))); //set out total
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_out_ary_id, sizeof(float)*h_dst_total_id)); // get dst memory on gpu.
	sum_of_use_memory += sizeof(float)*h_dst_total_id;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_power_id, &power, sizeof(float))); //set out power
	int size_float3 = sizeof(float)*3;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_datanum_id, h_datanum_id, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_min_id, h_min, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_max_id, h_max, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_interval_id, h_interval, size_float3));
	printf("size of out data on gpu: %f MB\n", sizeof(float)*h_dst_total_id/1024.0/1024.0);
	printf("size of use memory on gpu: %f MB\n", sum_of_use_memory/1024.0/1024.0);
	return h_dst_total_id;
}

void InverseDistance_ComputeData(_out float *dstdata, int th, bool useShMem)
{
	int threadsPerBlock = 256;
	if (!useShMem)
		threadsPerBlock = th;
	int blocksPerGrid = (sh_total_id + threadsPerBlock - 1) / threadsPerBlock;

	if (useShMem)
		InverseDistance_GetNearest<<<blocksPerGrid, threadsPerBlock>>>
			(d_out_ary_id, d_data_ary_id, d_xary_id, d_yary_id, d_zary_id);
	else
		InverseDistance_GetNearestSH<<<blocksPerGrid, threadsPerBlock>>>
			(d_out_ary_id, d_data_ary_id, d_xary_id, d_yary_id, d_zary_id);

	cutilCheckMsg("kernel launch failure");
#ifdef _DEBUG
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
#endif
	CUDA_SAFE_CALL( cudaMemcpy(dstdata, d_out_ary_id, sizeof(float)*h_dst_total_id, cudaMemcpyDeviceToHost) );
	if (d_xary_id) cudaFree(d_xary_id);
	if (d_yary_id) cudaFree(d_yary_id);
	if (d_zary_id) cudaFree(d_zary_id);
	if (d_data_ary_id) cudaFree(d_data_ary_id);
	if (d_out_ary_id) cudaFree(d_out_ary_id);
	CUDA_SAFE_CALL( cudaThreadExit() );
}
