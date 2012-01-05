// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once
#include "InterpolationInfo.h"
#include "kd_tree.h"
#define _out
#define _in
// use by vtkKdtreeLimitedInverseDistanceFilterCuda
//set sample and result data and parameter
int KdtreeLimitedInverseDistance_SetData(const InterpolationInfo *h_info, float h_power, float h_radius, int h_point_maximum, float h_nullvalue);

void KdtreeLimitedInverseDistance_ComputeData(_out float *dstdata, int th);
//set k-d tree data
void kdtree_setdata( kd_tree* kdt, point* pts, int size, int max_result );
//release k-d tree data
void kdtree_freedata();

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)


