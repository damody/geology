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
int InverseDistance_SetData(const InterpolationInfo *h_info, float h_power);

/**
 * @brief 
 *
 * start GPU ordinary Inverse Distance compute
 * @param th	number of thread
 * @param useShMem	whether use shared memory or not
 */
void InverseDistance_ComputeData(_out float *dstdata, int th, bool useShMem);

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

