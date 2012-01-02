// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
#pragma once
#include "InterpolationInfo.h"
#define _out
#define _in
// use by vtkNearestNeighborFilterCuda
int NearestNeighbor_SetData(const InterpolationInfo *h_info);
void NearestNeighbor_ComputeData(_out float *dstdata);

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
