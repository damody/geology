#include <stdlib.h>
#include <stdio.h>
#include "SimpleOrdinaryKriging.h"
#include <cutil.h>
#include <cutil_inline.h>


static int h_datanum_OK[3], h_dst_total_OK;
__constant__ int d_datanum_OK[3], d_total_OK[1], d_dst_total_OK[1], d_smp_num_OK[1];
__constant__ float d_min_OK[3], d_max_OK[3], d_interval_OK[3], d_Nugget_OK[1], d_Sill_OK[1], d_Range_OK[1];
float *d_xary_OK, *d_yary_OK, *d_zary_OK, *d_data_ary_OK, *d_out_ary_OK, *d_mat_OK;
static int sh_total_OK;

__device__ float SphericalModel(float d)
{
	if (d > d_Range_OK[0])
		return d_Nugget_OK[0] + d_Sill_OK[0];
	float sp = d / d_Range_OK[0];
	return d_Nugget_OK[0] + d_Sill_OK[0] * (1.5f * sp  - sp*sp*sp/ 2.0f);
}

__device__ inline void OK_GetDistance(float x1, float x2, float y1, float y2, float z1, float z2, float *res)
{
	*res = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
}

__device__ void OK_ComputePoint(float x, float y, float z, float *res, float *data_ary
						, float *xary, float *yary, float *zary, float* mat)
{
	float w, d, total, r;
	int i, j;
	total = 0;
// 	for (i=0; i<*d_smp_num_OK; i++)
// 	{
// 		w=0;
// 		for (j=0; j<*d_smp_num_OK; j++)
// 		{
// 			OK_GetDistance(x, xary[j], y, yary[j], z, zary[j], &d);
// 			w = w + mat[(*d_smp_num_OK+1)*i+j]*SphericalModel(d);
// 		}
// 		w = w + mat[(*d_smp_num_OK+1)*i + j];
// 		w = w * data_ary[i];
// 
// 		total = total + w;
// 	}

	
	for (i=0; i<*d_smp_num_OK; i++)
	{
		w=0;
		OK_GetDistance(x, xary[i], y, yary[i], z, zary[i], &d);
		r = SphericalModel(d);
		for (j=0; j<*d_smp_num_OK; j++)
		{
			w = w + mat[(*d_smp_num_OK+1)*j+i]* r *data_ary[j];
		}
		w += mat[(*d_smp_num_OK+1)*i + j]*data_ary[i];
		total = total + w;
	}

	*res=total;
}


__global__ void OK_Compute(float *out_ary, float *data_ary, float *xary, float *yary, float *zary, float* mat)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int inc = blockDim.x * gridDim.x;
	int xindex, yindex, zindex;
	float x ,y, z, res = 0;
	int i;

	for (i = idx ; i < *d_dst_total_OK ; i += inc)
	{
		xindex = i % d_datanum_OK[0];
		yindex = (i / d_datanum_OK[0]) % d_datanum_OK[1];
		zindex = i / (d_datanum_OK[0] * d_datanum_OK[1]);
		x = d_min_OK[0] + xindex * d_interval_OK[0];
		y = d_min_OK[1] + yindex * d_interval_OK[1];
		z = d_min_OK[2] + zindex * d_interval_OK[2];
		OK_ComputePoint(x, y, z, &res, data_ary, xary, yary, zary, mat);
		out_ary[i] = res;
	}
}

//use shared memory
__device__ void OK_ComputePointSH(float x, float y, float z, float *res, float *data_ary
						, float *xary, float *yary, float *zary, float* mat, int tid)
{
	float w, d, total, r;
	int i, j, k, idx;
	total = 0;
	__shared__ float shmat[256];
	__shared__ float shval[256];
	
	for (i=0; i<*d_smp_num_OK; i++)
	{
		w=0;
		OK_GetDistance(x, xary[i], y, yary[i], z, zary[i], &d);
		r = SphericalModel(d);

		for (j=0; j<*d_smp_num_OK; j+=256)
		{
			idx = j+tid;
			//idx*=2;
			if (tid<256)
			{
				__syncthreads();
				shmat[tid] = mat[(*d_smp_num_OK+1)*idx + i];
				shval[tid] = data_ary[idx];
// 				shmat[idx+1] = mat[(*d_smp_num_OK+1)*(idx+1) + i];
// 				shval[idx+1] = data_ary[idx+1];
			}
			__syncthreads();

			for (k=0; j+k<*d_smp_num_OK && k<256; k++)
				w = w + shmat[k]* r * shval[k];
		}	

		w += mat[(*d_smp_num_OK+1)*i + *d_smp_num_OK]*data_ary[*d_smp_num_OK];
		total = total + w;
	}

	*res=total;
}


__global__ void OK_ComputeSH(float *out_ary, float *data_ary, float *xary, float *yary, float *zary, float* mat)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int inc = blockDim.x * gridDim.x;
	int xindex, yindex, zindex;
	float x ,y, z, res = 0;
	int i;

	for (i = idx ; i < *d_dst_total_OK ; i += inc)
	{
		xindex = i % d_datanum_OK[0];
		yindex = (i / d_datanum_OK[0]) % d_datanum_OK[1];
		zindex = i / (d_datanum_OK[0] * d_datanum_OK[1]);
		x = d_min_OK[0] + xindex * d_interval_OK[0];
		y = d_min_OK[1] + yindex * d_interval_OK[1];
		z = d_min_OK[2] + zindex * d_interval_OK[2];
		OK_ComputePointSH(x, y, z, &res, data_ary, xary, yary, zary, mat, threadIdx.x);
		out_ary[i] = res;
	}
}



// use by external, return need float ary size
__host__ int OK_SetData( const InterpolationInfo *h_info, const float *h_kmat, float nugget, float sill, float range )
{
	int sum_of_use_memory = 0;
	const float kb_to_mb = 1/1024.0f/1024.0f;
	// set in data
	sh_total_OK = h_info->m_total;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_smp_num_OK, &sh_total_OK, sizeof(int)));
	int size = sizeof(float)*sh_total_OK;
	int matsize = sizeof(float)*(sh_total_OK+1)*(sh_total_OK+1);
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_xary_OK, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_yary_OK, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_zary_OK, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_ary_OK, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_mat_OK, matsize));
	sum_of_use_memory += size*4;
	sum_of_use_memory += matsize;
	printf("size of use input data on gpu: %f MB\n", sum_of_use_memory*kb_to_mb);
	 // get source memory on gpu
	CUDA_SAFE_CALL(cudaMemcpy(d_xary_OK, h_info->m_posAry[0], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_yary_OK, h_info->m_posAry[1], size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_zary_OK, h_info->m_posAry[2], size, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpy(d_data_ary_OK, h_info->m_data_ary, size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_mat_OK, h_kmat, matsize, cudaMemcpyHostToDevice));
	// set out data
	const float *h_min_OK = h_info->min, *h_max_OK = h_info->max, *h_interval_OK = h_info->interval;
	for (int i=0;i<3;i++)
		h_datanum_OK[i] = (int)floor((h_max_OK[i]-h_min_OK[i])/h_interval_OK[i])+1;
	h_dst_total_OK = h_datanum_OK[0] * h_datanum_OK[1] * h_datanum_OK[2];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dst_total_OK, &h_dst_total_OK, sizeof(int))); //set out total
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_out_ary_OK, sizeof(float)*h_dst_total_OK)); // get dst memory on gpu.
	sum_of_use_memory += sizeof(float)*h_dst_total_OK;
	int size_float3 = sizeof(float)*3;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_datanum_OK, h_datanum_OK, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_min_OK, h_min_OK, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_max_OK, h_max_OK, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_interval_OK, h_interval_OK, size_float3));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Nugget_OK, &nugget, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Sill_OK, &sill, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_Range_OK, &range, sizeof(float)));

	printf("size of out data on gpu: %f MB\n", sizeof(float)*h_dst_total_OK*kb_to_mb);
	printf("size of use memory on gpu: %f MB\n", sum_of_use_memory*kb_to_mb);

	return h_dst_total_OK;
}

void OK_ComputeData(_out float *dstdata, int th, bool useShMem)
{

	int threadsPerBlock = 256;
	if (!useShMem)
		threadsPerBlock = th;
	int blocksPerGrid = (sh_total_OK + threadsPerBlock - 1) / threadsPerBlock;

	if (useShMem)
		OK_ComputeSH<<<blocksPerGrid, threadsPerBlock>>>
		(d_out_ary_OK, d_data_ary_OK, d_xary_OK, d_yary_OK, d_zary_OK, d_mat_OK);
	else
		OK_Compute<<<blocksPerGrid, threadsPerBlock>>>
		(d_out_ary_OK, d_data_ary_OK, d_xary_OK, d_yary_OK, d_zary_OK, d_mat_OK);

#ifdef _DEBUG
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
#endif
	CUDA_SAFE_CALL( cudaMemcpy(dstdata, d_out_ary_OK, sizeof(float)*h_dst_total_OK, cudaMemcpyDeviceToHost) );
	if (d_xary_OK) cudaFree(d_xary_OK);
	if (d_yary_OK) cudaFree(d_yary_OK);
	if (d_zary_OK) cudaFree(d_zary_OK);
	if (d_data_ary_OK) cudaFree(d_data_ary_OK);
	if (d_mat_OK) cudaFree(d_mat_OK);
	if (d_out_ary_OK) cudaFree(d_out_ary_OK);
	CUDA_SAFE_CALL( cudaThreadExit() );
}
