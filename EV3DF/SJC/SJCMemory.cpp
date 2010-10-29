
/************************************************************************
     Main File:

     File:        SJCScalarField2.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
    Comment:     Class to handle the scalar field in 2D

   Constructors:
                  1. 0 : the default contructor
                  2. 6 : constructor to set up all value by input parameters
                  3. 1 : set up the class by using the scalar field
                  4. 1 : copy contructor
                   
     Functions:  what r for?
                 1. = : Assign operator which copy the parameter of random
                 2. (): Get the value of the scalar field
                 3. value: get the value of the scalar field
                 4. grad: get the gradient of the scalar field
                 5. curl: get the curl of the scalar field
                 6. MinX, MinY, MaxX, MaxY: get the maximum and minimum value 
                    of X, y
************************************************************************/
     
#include <SJC/SJCMemory.h>

//*****************************************************************************
//
// * Memory Allocation Functions
//============================================================================
void *AllocAligned(size_t size) 
//============================================================================
{
#ifndef L1_CACHE_LINE_SIZE
#define L1_CACHE_LINE_SIZE 64
#endif
  return memalign(L1_CACHE_LINE_SIZE, size);
}

//*****************************************************************************
//
// * 
//============================================================================
void FreeAligned(void *ptr) 
//============================================================================
{
#ifdef WIN32 // NOBOOK
  _aligned_free(ptr);
#else // NOBOOK
  free(ptr);
#endif // NOBOOK
}
