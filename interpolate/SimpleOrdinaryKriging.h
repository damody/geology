#pragma once
#include "InterpolationInfo.h"
#define _out
#define _in
// use by vtkNearestNeighborFilterCuda
int OK_SetData(const InterpolationInfo *h_info, const float *h_kmat, float nugget, float sill, float range);
void OK_ComputeData(_out float *dstdata, int th, bool useShMem );

// use by internal 
//void NearestNeighbor_GetNearest(float *d_pos, float *res);
