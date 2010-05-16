/************************************************************************
     Main File:   Main.cpp

     File:        SJCConstant.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Stephen Chenney, schenney@cs.wisc.edu
  
     Comment:     The global variables
     
     Compiler:    g++

     Platform:    Linux
*************************************************************************/
#ifndef _SJCCONSTANTS_H
#define _SJCCONSTANTS_H

#include "SJC.h"
#include "SJCVector3.h"


//****************************************************************************
//
// * Global variables
//
//*****************************************************************************
class SJCDLL SJCConstants {
 public:
  // Global constants
  static const SJCVector3d  SJC_vXAxis3d;
  static const SJCVector3d  SJC_vYAxis3d;
  static const SJCVector3d  SJC_vZAxis3d;
  static const SJCVector3d  SJC_vUpDir3d;
  static const SJCVector3d  SJC_vForwardDir3d;

  static const SJCVector3f  SJC_vXAxis3f;
  static const SJCVector3f  SJC_vYAxis3f;
  static const SJCVector3f  SJC_vZAxis3f;
  static const SJCVector3f  SJC_vUpDir3f;
  static const SJCVector3f  SJC_vForwardDir3f;
  
  
};

#endif

