#pragma once
#include "InterpolationInfo.h"
#define _out
#define _in
/**
 * @brief 
 *
 * set information data to gpu memory
 * @param h_info	information structure
 * @param h_kmat	kriging matrix 
 * @param nugget, sill, range	kriging parameter
 */
int OK_SetData(const InterpolationInfo *h_info, const float *h_kmat, float nugget, float sill, float range);


/**
 * @brief 
 *
 * start GPU ordinary kriging compute
 * @param dstdata	destination data
 * @param th	number of thread
 * @param useShMem	whether use shared memory or not
 */
void OK_ComputeData(_out float *dstdata, int th, bool useShMem );

