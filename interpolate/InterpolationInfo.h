// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once
#include <memory>
#include "vtkBounds.h"

#define CUDA_THREADS 64
//data information for GPU to use
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

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)
