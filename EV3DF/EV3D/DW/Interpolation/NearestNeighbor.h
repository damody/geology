#pragma once
#define THREADS 256


struct InterpolationInfo
{
	InterpolationInfo(int datasize)
	{
		total = datasize;
		posary[0] = (float*)malloc(sizeof(float)*datasize);
		posary[1] = (float*)malloc(sizeof(float)*datasize);
		posary[2] = (float*)malloc(sizeof(float)*datasize);
		data_ary = (float*)malloc(sizeof(float)*datasize);
	}
	~InterpolationInfo()
	{
		free(posary[0]);
		free(posary[1]);
		free(posary[2]);
		free(data_ary);
	}
	void GetPosFromXYZArray(float *data)
	{
		for (int i=0;i<total;i++)
		{
			posary[0][i] = data[i*4];
			posary[1][i] = data[i*4+1];
			posary[2][i] = data[i*4+2];
			data_ary[i] = data[i*4+3];
		}
	}
	void GetPosFromXYZArray(double *data)
	{
		for (int i=0;i<total;i++)
		{
			posary[0][i] = data[i*4];
			posary[1][i] = data[i*4+1];
			posary[2][i] = data[i*4+2];
			data_ary[i] = data[i*4+3];
		}
	}
	float max[3], min[3], interval[3];
	int total;
	float *data_ary, *posary[3];
};

#define _out
#define _in
// use by vtkNearestNeighborFilterCuda
int NearestNeighbor_SetData(const InterpolationInfo *h_info);
void NearestNeighbor_ComputeData(_out float *dstdata);

// use by internal 
//void NearestNeighbor_GetNearest(float *d_pos, float *res);


