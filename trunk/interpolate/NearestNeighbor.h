// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once
#include "InterpolationInfo.h"
#define _out
#define _in
/**
 * @brief 
 *
 * set information data to gpu memory
 * @param h_info	information structure
 */
int NearestNeighbor_SetData(const InterpolationInfo *h_info);

/**
 * @brief 
 *
 * start GPU NN compute
 * @param dstdata	destination data
 * @param th	number of thread
 * @param useShMem	whether use shared memory or not
 */
void NearestNeighbor_ComputeData(_out float *dstdata, int th, bool useShMem);

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

