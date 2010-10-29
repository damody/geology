/************************************************************************
     Main File:   SJCConstant.cpp

     File:        SJCConstant.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Stephen Chenney, schenney@cs.wisc.edu
  
     Comment:     The global variables
     
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include "SJCConstants.h"

//****************************************************************************
//
// * Global variables
//
//*****************************************************************************

const SJCVector3d  SJCConstants::SJC_vXAxis3d = SJCVector3d(1.0, 0.0, 0.0);
const SJCVector3d  SJCConstants::SJC_vYAxis3d = SJCVector3d(0.0, 1.0, 0.0);
const SJCVector3d  SJCConstants::SJC_vZAxis3d = SJCVector3d(0.0, 0.0, 1.0);
const SJCVector3d  SJCConstants::SJC_vUpDir3d = SJCVector3d(0.0, 0.0, 1.0f);
const SJCVector3d  SJCConstants::SJC_vForwardDir3d = SJCVector3d(1.0,0.0,0.0);

const SJCVector3f  SJCConstants::SJC_vXAxis3f = SJCVector3f(1.0f, 0.0, 0.0);
const SJCVector3f  SJCConstants::SJC_vYAxis3f = SJCVector3f(0.0, 1.0f, 0.0);
const SJCVector3f  SJCConstants::SJC_vZAxis3f = SJCVector3f(0.0, 0.0, 1.0f);
const SJCVector3f  SJCConstants::SJC_vUpDir3f = SJCVector3f(0.f, 0.f, 1.f);
const SJCVector3f  SJCConstants::SJC_vForwardDir3f = SJCVector3f(1.f,0.f,0.f);
