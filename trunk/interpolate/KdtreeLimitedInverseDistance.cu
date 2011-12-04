#include "KdtreeLimitedInverseDistance.h"
#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>
#include <cutil_inline.h>

static int h_datanum[3], h_dst_total;
__constant__ int d_datanum[3], d_total[1], d_dst_total[1], d_point_maximum[1];
__constant__ float d_min[3], d_max[3], d_interval[3], d_power[1], d_kdradius[1], d_nullvalue[1];
float *d_data_ary, *d_out_ary;
static int sh_total;
//Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]: \t" fmt, \
	blockIdx.y*gridDim.x+blockIdx.x,\
	threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
	__VA_ARGS__)
__constant__ int d_max_result[1];
static point* d_points;
static kd_fast_node* d_kd_fnodes;
static int **d_each_res;

__device__ inline bool IsCollision( const float* p, const float* bounding )
{
	if ((bounding[0] <= p[0] && bounding[1] >= p[0]) &&
		(bounding[2] <= p[1] && bounding[3] >= p[1]) &&
		(bounding[4] <= p[2] && bounding[5] >= p[2]))
		return true;
	return false;
}

__device__ inline bool FBoundingCollision( const float* b1, const float* range, int dir )
{
	int i = dir*2;
	if (b1[i] <= range[0] && b1[i+1] >= range[0])
		return true;
	if (b1[i] <= range[1] && b1[i+1] >= range[1])
		return true;
	if (range[0] <= b1[i] && range[1] >= b1[i])
		return true;
	if (range[0] <= b1[i+1] && range[1] >= b1[i+1])
		return true;
	return false;
}

__device__ void FSearchKdtreeByBounding(int *tmp_ary, const float* bounding, const float* _pos,
			const kd_fast_node* kd_fnodes, const point* points)
{
	// init result
	for (int i=0;i<*d_max_result;i++)
		tmp_ary[i] = -1;
	// search
	int stack[18];
	memset(stack, 0, sizeof(stack));
	int s_idx = 0, res_idx = 0;
#define push(X) stack[s_idx++] = (X)
#define get() stack[s_idx-1]
#define pop() --s_idx
	push(0);
	for (;s_idx != 0;)
	{
		const kd_fast_node* n = &kd_fnodes[get()];
		pop();
		if (n->dir == -1) // is leaf
		{
			if (n->pts[0] != -1 && IsCollision(points[n->pts[0]].p, bounding)) 
			{
				tmp_ary[res_idx++] = n->pts[0];
				if (res_idx >= *d_max_result) return;
				if (n->pts[1] != -1 && IsCollision(points[n->pts[1]].p, bounding)) 
				{
					tmp_ary[res_idx++] = n->pts[1];
					if (res_idx >= *d_max_result) return;
				}
			}
		}
		else if (FBoundingCollision(bounding, n->bounds, n->dir))	// has node
		{
			if (n->dir!=-1)
			{
				float div = (n->bounds[n->dir*2+1]+n->bounds[n->dir*2])*0.5;
				float dl = abs(_pos[n->dir]-(div+n->bounds[n->dir*2])*0.5);
				float dr = abs(_pos[n->dir]-(div+n->bounds[n->dir*2+1])*0.5);
				//printf("%.2f %.2f %.2f\n", _pos[0], _pos[1], _pos[2]);
				if (dl<dr)
				{	// left first
					push(n->right);
					push(n->left);
				}
				else
				{	// right first
					push(n->left);
					push(n->right);
				}
			}
			else
			{
				push(n->left);
				push(n->right);	
			}
		}
	}
}

#define MAX_IDX 800
int** tmps, res_size;
void kdtree_setdata( kd_tree* kdt, point* pts, int size, int max_result )
{
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_kd_fnodes, 
		sizeof(kd_fast_node)*kdt->m_fnodes.size()));
	CUDA_SAFE_CALL(cudaMemcpy(d_kd_fnodes, &(kdt->m_fnodes[0]), 
		sizeof(kd_fast_node)*kdt->m_fnodes.size(), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_points, sizeof(point)*size));
	CUDA_SAFE_CALL(cudaMemcpy(d_points, pts, 
		sizeof(point)*size, cudaMemcpyHostToDevice));
	//set max size for result
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_max_result, &max_result, sizeof(d_max_result))); 
	tmps = (int**)malloc(sizeof(int*)*MAX_IDX);
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_each_res, sizeof(int*)*MAX_IDX));
	res_size = max_result;
	for (int i=0;i<MAX_IDX;++i)
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)&(tmps[i]), sizeof(int)*max_result));
	}
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_each_res, sizeof(int*)*MAX_IDX));
	CUDA_SAFE_CALL(cudaMemcpy(d_each_res, tmps, 
		sizeof(int*)*MAX_IDX, cudaMemcpyHostToDevice));
	cutilCheckMsg("kernel launch failure");
}
void kdtree_freedata()
{
	CUDA_SAFE_CALL(cudaFree(d_points));
	CUDA_SAFE_CALL(cudaFree(d_kd_fnodes));
	CUDA_SAFE_CALL(cudaFree(d_each_res));
	for (int i=0;i<MAX_IDX;++i)
	{
		CUDA_SAFE_CALL(cudaFree(tmps[i]));
	}
	free(tmps);
}

// use by internal 
__device__ inline void KdtreeLimitedInverseDistance_GetDistance(
	float x1, float x2, float y1, float y2, float z1, float z2, float *res)
{
	*res = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
}

__device__ inline void KdtreeLimitedInverseDistance_Weightx(
	const float *t_pos, float *res, const float *data_ary,
	const kd_fast_node* kd_fnodes, const point *pts, const float *bounding, int *tmp_ary)
{
	float now, sum = 0, sum2 = 0, tmp = 0;
	//int idx = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i=0;i<d_max_result[0];i++)
		tmp_ary[i] = -1;
	//kdtree search
	FSearchKdtreeByBounding(tmp_ary, bounding, t_pos, kd_fnodes, pts);
	for (int j = 0;j < d_max_result[0];j++)
	{
		int index = tmp_ary[j];
		if (index == -1)
		{
			//CUPRINTF("%d\n", j);
			break;
		}
		KdtreeLimitedInverseDistance_GetDistance(
			t_pos[0], pts[index].x, t_pos[1], pts[index].y,
			t_pos[2], pts[index].z, &now);
		if (now<0.001)
		{
			*res = data_ary[index];
			return ;
		}
		tmp = pow(now, -d_power[0]);
		sum += tmp;
		sum2 += tmp*data_ary[index];
	}
	if (sum2 == 0)
		*res = d_nullvalue[0];
	else
		*res = sum2 / sum;
}

__global__ void KdtreeLimitedInverseDistance_Weight(float *out_ary, const float *data_ary, 
					const kd_fast_node* kd_fnodes, const point *pts, int **tmp_ary)
{
 	int idx = blockDim.x * blockIdx.x + threadIdx.x;
 	int inc = blockDim.x * gridDim.x;
 	int xindex, yindex, zindex;
 	float t_pos[3], res, t_bound[6];
 	for (int i = idx ; i < d_dst_total[0] ; i += inc)
 	{
 		xindex = i % d_datanum[0];
 		yindex = (i / d_datanum[0]) % d_datanum[1];
 		zindex = i / (d_datanum[0] * d_datanum[1]);
 		t_pos[0] = d_min[0] + xindex * d_interval[0];
 		t_pos[1] = d_min[1] + yindex * d_interval[1];
 		t_pos[2] = d_min[2] + zindex * d_interval[2];
 		t_bound[0] = t_pos[0]-d_kdradius[0];
 		t_bound[1] = t_pos[0]+d_kdradius[0];
 		t_bound[2] = t_pos[1]-d_kdradius[0];
 		t_bound[3] = t_pos[1]+d_kdradius[0];
 		t_bound[4] = t_pos[2]-d_kdradius[0];
 		t_bound[5] = t_pos[2]+d_kdradius[0];
  		KdtreeLimitedInverseDistance_Weightx(
  			t_pos, &res, data_ary, kd_fnodes, pts, t_bound, tmp_ary[idx]);
		//printf("%.2f %.2f %.2f\n", t_pos[0], t_pos[1], t_pos[2]);
 		out_ary[i] = res;
 	}
}

// use by external, return need float ary size
__host__ int KdtreeLimitedInverseDistance_SetData( const InterpolationInfo *h_info, float h_power, float h_radius, int h_point_maximum, float h_nullvalue )
{
	h_radius += 0.001; // because gpu has small error need to add radius to fixed it
	int sum_of_use_memory = 0;
	// set in data
	sh_total = h_info->m_total;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_total, &sh_total, sizeof(int)));
	int size = sizeof(float)*sh_total;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_ary, size));
	sum_of_use_memory += size*4;
	printf("size of use input data on gpu: %d bytes\n", size*4);
	// get source memory on gpu
	CUDA_SAFE_CALL(cudaMemcpy(d_data_ary, h_info->m_data_ary, size, cudaMemcpyHostToDevice));
	// set out data
	const float *h_min = h_info->min, *h_max = h_info->max, *h_interval = h_info->interval;
	for (int i=0;i<3;i++)
		h_datanum[i] = (int)floor((h_max[i]-h_min[i])/h_interval[i])+1;
	h_dst_total = h_datanum[0] * h_datanum[1] * h_datanum[2];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dst_total, &h_dst_total, sizeof(int))); //set out total
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_out_ary, sizeof(float)*h_dst_total)); // get dst memory on gpu.
	sum_of_use_memory += sizeof(float)*h_dst_total;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_power, &h_power, sizeof(float))); //set out power
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_kdradius, &h_radius, sizeof(float))); //set out radius
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_nullvalue, &h_nullvalue, sizeof(float))); //set out nullvalue
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_point_maximum, &h_point_maximum, sizeof(int))); //set out point_maximum
	int size_float3 = sizeof(float)*3;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_datanum, h_datanum, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_min, h_min, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_max, h_max, size_float3));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_interval, h_interval, size_float3));
	printf("size of out data on gpu: %d bytes\n", sizeof(float)*h_dst_total);
	printf("size of use memory on gpu: %d bytes\n", sum_of_use_memory);
	return h_dst_total;
}

void KdtreeLimitedInverseDistance_ComputeData(_out float *dstdata, int th)
{
	int threadsPerBlock = th;
	int blocksPerGrid = (sh_total + threadsPerBlock - 1) / threadsPerBlock;
	KdtreeLimitedInverseDistance_Weight<<<blocksPerGrid, threadsPerBlock>>>
		(d_out_ary, d_data_ary, d_kd_fnodes, d_points, d_each_res);
	cutilCheckMsg("kernel launch failure");
#ifdef _DEBUG
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
#endif
	CUDA_SAFE_CALL( cudaMemcpy(dstdata, d_out_ary, sizeof(float)*h_dst_total, cudaMemcpyDeviceToHost) );
	if (d_data_ary) cudaFree(d_data_ary);
	if (d_out_ary) cudaFree(d_out_ary);
}
