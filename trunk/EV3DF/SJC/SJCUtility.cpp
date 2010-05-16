/************************************************************************
     Main File:

     File:        MathUtility.h

     Author:     
                  Lucas , yu-chi@cs.wisc.edu
                   
     Comment:     Common math operation

    Function:
                 1. mySign(double): return the sign
                 2. mySign(int): return the sign
                 3. mySignCheck( double, double): return true if two are same 
                    sign
                 4. mySignCheck( int, int): return true if two are the same 
                    sign
                 5. myFabs(double): return absolute value
                 6. myAbs(int): return absolute value
                 7. myMax(a, b): return maximum value
                 8. myMin(a, b): return minimum value
                 9. acosSafe(double): Returns the arccos of the argument,
                    which is clamped to [-1,1]
                10. asinSafe(double): Returns the arcsine of the argument,
                    which is clamped to [-1,1]
                11. ceilSafe(double,double): Computes the ceiling of the given
                    number while attempting to account for roundoff.  
                    If n - int(n) < tol, this returns n; 
                    otherwise it returns int(n) + 1.  
                12. ceilSafe(double): the same as previous but tol = 1E-8
                13. floorSafe(double,double): Computes the ceiling of the given
                    number while attempting to account for roundoff.  
                    If n - int(n) < tol, this returns n; 
                    otherwise it returns int(n) + 1.  
                14. floorSafe(double): the same as previous but tol = 1E-8

                15. atanUp(double,double):
                16. atanDown(double, double): Calculates, respectively, the 
                    planar rotation necessary to align the given (x,y) pair 
                    with the positive/negative y axis.
                17. fitQuadratic(double*, double*, int, double*):
                    find the best-fit quadratic curve in 2D
                  
                18. maximizeContinuity(Quaternion*, int)
                19. maximizeContinuity(double*, int ): Given a composed rotation
                    in angle or in Quaternion to find the best smooth 
                    transition
                20. filter(double*, int, double*, int, double*):
                21. filter(double**, int, int, double*, int, bootl, double**):
                22. filter(vector*, int, double*, int, SJCVector*):
                23. filterRotations(Quaternion*,int, double*, int, Quaternion*):
                24. medianFilter(double*, int, int, double*):
                    Convolves the given signal with the given kernel
                25. getKernel(double*, int, convolveKernelT): Fill in kernel by
                    the supplied type

                26. averageVelocity(double*, int): average speed in 1D
                27. averageVelocity(double**, int, int, bool, double*):
                28. averageVelocity(Quaternion*, int, double *):

                29. matrixMultiply(double**,double*, int, int, double*):
                    matrix multiply by a std::vector
                30. matrixMultiply(double**, double**, int, int, int, double**):
                    matrix multiply by a matrix
                31. extractAngle(SJCVector2, SJCVector2, bool): rotate first 
                    to second
 
                32. transformPnt(SJCVector2, double, SJCVector2): 2D transform
                33. transformPnt(SJCVector, double, SJCVector2): 3D transform
                34. transformPnt(SJCVector, double, SJCVector): 3D transformation

                35. composeTransform(double, std::vector2, double, std::vector, bool)
                36. composeTransform(Quat, std::vector, Quat, std::vector, bool): 
                    compose two transformation into one

                37. inverseTransform(double, std::vector2): compute the inverse 
                    transformation in 2D

                38. canonicalRotation(double, std::vector2):
                39. canonicalRotation(Quaternion, std::vector):
                    Converts a rotation about some point to a rotation about
                    the origin plus a translation.

                40. subtractOffComponent(double*,double*):
                    remove the first component of second

                41. projectOnto(double*, double*): project the first std::vector onto
                    second

                42. isLeft(double*, double*, double*):
                43. isLeftOrOn(double*, double*, double*): 
                    Returns true if the given 2D point is to the left of
                    the given 2D segment
                44. projectPointOntoSegment(double*,double*,double*,double, int, double*)
                    Given a point (pt) and a line segment specified by one
                    endpoint (segStart), a (unit length) direction (segDir),
                    and a scalar lambda such that the other endpoint is
                    segStart + lambda*segDir, this finds the closest point
                    on the segment to pt.  The next to last argument specifies
                    the dimension of the point/segment

                45. findClosestRotation(Quaternion*, double *, Quaternion*): 
                    This calculates the rotation about the given axis that is
                    as close as  possible to the first argument in a 
                    great-arc sense.

                  (a). alignPointClouds:
                  (b). newAlignPointClouds:  Given two point clouds, this 
                       finds the optimal rotation about the y-axis 
                       and translation in the xz plane that aligns the given 
                       point clouds. 
                  (c). svdcmp(double**, int, int, double*, double**):
                       svd decomposition

                  (d). svbksb(double**, double*, double**, int, int, double*, double*): 
                       solve A * x = b
                  (e). choldc(double**, int, double*): Cholesky decomposition
                  (f). cholsl(double**, int, double*, double*, double*): 
                       Solve A * x = b

                  (g). jacobiEigen4x4(double**, double*, double **): 
                       compute all eigen values and eigenvector of a real 
                       symmetric 4 * 4 matrix

     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include "SJCUtility.h"
#include <math.h>


//*****************************************************************************
//
// * Return the asin in safe region
//============================================================================
SJCDLL double SJCASin( double sinAngle ) 
//============================================================================
{
  if( sinAngle < -1 ) {
    sinAngle = -1;
  }
  else if( sinAngle > 1 ) {
    sinAngle = 1;
  }
  
  return( asin(sinAngle) );
}

//*****************************************************************************
//
// * Return the acos in safe region
//============================================================================
SJCDLL double SJCACos( double cosAngle ) 
//============================================================================
{
  if( cosAngle < -1 ) {
    cosAngle = -1;
  }
  else if( cosAngle > 1 ) {
    cosAngle = 1;
  }

  return( acos(cosAngle) );
}

