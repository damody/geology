#include "LimitedInverseDistance.h"
#include <stdlib.h>
#include <cutil.h>
#include <cutil_inline.h>

static int h_datanum_lid[3], h_dst_total_lid;
__constant__ int d_datanum_lid[3], d_total_lid[1], d_dst_total_lid[1], d_point_maximum_lid[1];
__constant__ float d_min_lid[3], d_max_lid[3], d_interval_lid[3], d_power_lid[1], d_radius_lid[1], d_nullvalue_lid[1];
float *d_xary_lid, *d_yary_lid, *d_zary_lid, *d_data_ary_lid, *d_out_ary_lid;
static int sh_total_lid;

// use by internal 
__device__ inline void LimitedInverseDistance_GetDistance(float x1, float x2, float y1, float y2, float z1, float z2, float *res)
{
	*res = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
}

__device__ inline void LimitedInverseDistance_GetNearestPoint(float x, float y, float z, float *res, float *data_ary
						, float *xary, float *yary, float *zary)
{
	float now, sum = 0, sum2 = 0, tmp = 0;
	for (int i = 0;i < *d_total_lid;i++)
	{
		LimitedInverseDistance_GetDistance(x, xary[i], y, yary[i], z, zary[i], &now);
		if (now<0.001)
		{
			*res = data_ary[i];
			return ;
		}
		if (now < *d_radius_lid)
		{
			tmp = pow(now, -d_power_lid[0]);
			sum += tmp;
			sum2 += tmp*data_ary[i];
		}
	}
	if (sum2 == 0)
		*res = d_nullvalue_lid[0];
	else
		*res = sum2 / sum;
}

__global__ void LimitedInverseDistance_GetNearest(float *out_ary, float *data_ary, float *xary, float *yary, float *zary)
{
	//int tid = threadIdx.x;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int inc = blockDim.x * gridDim.x;
	int xindex, yindex, zindex;
	float x ,y, z, res;
	for (int i = idx ; i < *d_dst_total_lid ; i += inc)
	{
		xindex = i % d_datanum_lid[0];
		yindex = (i / d_datanum_lid[0]) % d_datanum_lid[1];
		zindex = i / (d_datanum_lid[0] * d_datanum_lid[1]);
		x = d_min_lid[0] + xindex * d_interval_lid[0];
		y = d_min_lid[1] + yindex * d_interval_lid[1];
		z = d_min_lid[2] + zindex * d_interval_lid[2];
		LimitedInverseDistance_GetNearestPoint(x, y, z, &res, data_ary, xary, yary, zary);
		out_ary[i] = res;
	}
}

// use by external, return need float ary size
__host__ int LimitedInverseDistance_SetData( const InterpolationInfo *h_info, float h_power, float h_radius, int h_point_maximum, float h_nullvalue )
{
	h_radius += 0.001; // because gpu has small error need to add radius to fixed it
	int sum_of_use_memory = 0;
	// set in data
	sh_total_lid = h_info->m_total;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_total_lid, &sh_total_lid, sizeof(int)));
	int size = sizeof(float)*sh_total_lid;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_xary_lid, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_yary_lid, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_zary_lid, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_ary_lid, size));
	sum_of_use_memory += size*4;
	printf("size of use input data on gpu: %d bytes\n", size*4);
	 // get source memory on gpu
	CUDA_SAFE_CALL(cudaMemcpy(d_xary_lid, h_info->m_posAry[0], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_yary_lid, h_info->m_posAry[1], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_zary_lid, h_info->m_posAry[2], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data_ary_lid, h_info->m_data_ary, size, cudaMemcpyHostToDevice));
	// set out data
	const float *h_min = h_info->min, *h_max = h_info->max, *h_interval = h_info->interval;
	for (int i=0;i<3;i++)
		h_datanum_lid[i] = (int)floor((h_max[i]-h_min[i])/h_interval[i])+1;
	h_dst_total_lid = h_datanum_lid[0] * h_datanum_lid[1] * h_datanum_lid[2];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dst_total_lid, &h_dst_total_lid, sizeof(int))); //set out total
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_out_ary_lid, sizeof(float)*h_dst_total_lid)); // get dst memory on gpu.
	sum_of_use_memory += sizeof(float)*h_dst_total_lid;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_power_lid, &h_power, sizeof(float))); //set out power
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_radius_lid, &h_radius, sizeof(float))); //set out radius
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_nullvalue_lid, &h_nullvalue, sizeof(float))); //set out nullvalue
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_point_maximum_lid, &h_point_maximum, sizeof(int))); //set out point_maximum
	int size_float3 = sizeof(float)*3;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_datanum_lid, h_datanum_lid, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_min_lid, h_min, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_max_lid, h_max, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_interval_lid, h_interval, size_float3));
	printf("size of out data on gpu: %d bytes\n", sizeof(float)*h_dst_total_lid);
	printf("size of use memory on gpu: %d bytes\n", sum_of_use_memory);
	return h_dst_total_lid;
}

void LimitedInverseDistance_ComputeData(_out float *dstdata, int th)
{
	int threadsPerBlock = th;
	int blocksPerGrid = (sh_total_lid + threadsPerBlock - 1) / threadsPerBlock;
	LimitedInverseDistance_GetNearest<<<blocksPerGrid, threadsPerBlock>>>
		(d_out_ary_lid, d_data_ary_lid, d_xary_lid, d_yary_lid, d_zary_lid);
	cutilCheckMsg("kernel launch failure");
#ifdef _DEBUG
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
#endif
	CUDA_SAFE_CALL( cudaMemcpy(dstdata, d_out_ary_lid, sizeof(float)*h_dst_total_lid, cudaMemcpyDeviceToHost) );
	if (d_xary_lid) cudaFree(d_xary_lid);
	if (d_yary_lid) cudaFree(d_yary_lid);
	if (d_zary_lid) cudaFree(d_zary_lid);
	if (d_data_ary_lid) cudaFree(d_data_ary_lid);
	if (d_out_ary_lid) cudaFree(d_out_ary_lid);
	CUDA_SAFE_CALL( cudaThreadExit() );
}
