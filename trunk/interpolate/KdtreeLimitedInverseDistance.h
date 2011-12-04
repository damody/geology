#pragma once
#include "InterpolationInfo.h"
#include "kd_tree.h"
#define _out
#define _in
// use by vtkInverseDistanceFilterCuda
int KdtreeLimitedInverseDistance_SetData(const InterpolationInfo *h_info, float h_power, float h_radius, int h_point_maximum, float h_nullvalue);
void KdtreeLimitedInverseDistance_ComputeData(_out float *dstdata, int th);
void kdtree_setdata( kd_tree* kdt, point* pts, int size, int max_result );
void kdtree_freedata();
// use by internal 
//void InverseDistance_GetNearest(float *d_pos, float *res);


