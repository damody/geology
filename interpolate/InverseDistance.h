#pragma once
#include "InterpolationInfo.h"
#define _out
#define _in
// use by vtkInverseDistanceFilterCuda
int InverseDistance_SetData(const InterpolationInfo *h_info, float h_power);
void InverseDistance_ComputeData(_out float *dstdata, int th, bool useShMem);

// use by internal 
//void InverseDistance_GetNearest(float *d_pos, float *res);


