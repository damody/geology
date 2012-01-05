// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once
#include "InterpolationInfo.h"
#define _out
#define _in
// use by vtkInverseDistanceFilterCuda
//set all data and parameter
int LimitedInverseDistance_SetData(const InterpolationInfo *h_info, float h_power, float h_radius, int h_point_maximum, float h_nullvalue);
//start compute
void LimitedInverseDistance_ComputeData(_out float *dstdata, int th);

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)
