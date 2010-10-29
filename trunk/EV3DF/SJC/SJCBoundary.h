/************************************************************************
     Main File:

     File:        SJCBoundary.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     Enum for boundary condition


************************************************************************/  
#ifndef _SJC_BOUNDARY_H
#define _SJC_BOUNDARY_H
enum SJCBoundary { 
  BOUNDARY_WRAP,            // Periodic 
  BOUNDARY_NOWRAP,          // Include zero, free, fixed
  BOUNDARY_NOWRAP_FREE
};

#endif
