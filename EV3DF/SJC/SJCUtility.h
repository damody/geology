/************************************************************************
     Main File:

     File:        MathUtility.h

     Author:     
                  Lucas , yu-chi@cs.wisc.edu
                   
     Comment:     Common math operation

     Function:
                 1. acosSafe(double): Returns the arccos of the argument,
                    which is clamped to [-1,1]
                 2. asinSafe(double): Returns the arcsine of the argument,
                    which is clamped to [-1,1]
 
     Compiler:    g++

     Platform:    Linux
*************************************************************************/
#ifndef _SJC_UTILITY_H
#define _SJC_UTILITY_H

#include "SJC.h"
#include "SJCConstants.h"

// C library
#include <assert.h>

// C++ library
#include <vector>


// * Returns the arccos/arcsine of the argument, which is clamped to [-1,1];
SJCDLL double SJCACos( double cosAngle );
SJCDLL double SJCASin( double sinAngle );


#endif
