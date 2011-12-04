#pragma once
#include "InterpolationInfo.h"
#define _out
#define _in
// use by vtkInverseDistanceFilterCuda
int LimitedInverseDistance_SetData(const InterpolationInfo *h_info, float h_power, float h_radius, int h_point_maximum, float h_nullvalue);
void LimitedInverseDistance_ComputeData(_out float *dstdata, int th);

// use by internal 
//void InverseDistance_GetNearest(float *d_pos, float *res);


