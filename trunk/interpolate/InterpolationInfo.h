#pragma once
#include <memory>
#include "vtkBounds.h"

#define CUDA_THREADS 64

struct InterpolationInfo
{
	InterpolationInfo(int datasize);
	~InterpolationInfo();
	void GetPosFromXYZArray(float *data);
	void GetPosFromXYZArray(double *data);
	float max[3], min[3], interval[3];
	void SetBounds(const vtkBounds& b)
	{
		min[0] = (float)b.xmin;
		min[1] = (float)b.ymin;
		min[2] = (float)b.zmin;
		max[0] = (float)b.xmax;
		max[1] = (float)b.ymax;
		max[2] = (float)b.zmax;
	}
	int m_total;
	float *m_data_ary, *m_posAry[3];
};


