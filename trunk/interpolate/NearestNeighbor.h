#pragma once
#include "InterpolationInfo.h"
#define _out
#define _in
// use by vtkNearestNeighborFilterCuda
int NearestNeighbor_SetData(const InterpolationInfo *h_info);
void NearestNeighbor_ComputeData(_out float *dstdata, int th, bool useShMem);

// use by internal 
//void NearestNeighbor_GetNearest(float *d_pos, float *res);


