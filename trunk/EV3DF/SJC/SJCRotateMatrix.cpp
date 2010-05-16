/************************************************************************
     Main File:

     File:        SimMatrix.cpp

     Author:      
                  Steven Chenney, schenney@cs.wisc.edu
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
 
     Comment:     The 3 * 3 matrix operation
		  Math Matrix Operation
	
 	
     Contructor:
		  1. 0 paras: identity 
		  2. 9 TData: a00, a01, a02...
		  3. 3 std::vector: row1, row2, row3
		  4. 1 matrix
     Function:
		  1. (i, j) = return aij
		  2. [i] = return row i
		  3. x(), y(), z(): return x, y, z std::vector
		  4. x(vector), y(vector), z(vector): set x, y, z std::vector
		  5. = (matrix): assign the matrix to this
		  6. * (matrix B): This * B
		  7. * (vector V): This * V
		  8. toIdentity(): create identity matrix
		  9. inverse(): calculate the inverse function
		 10. toAngleAxis(TData&, std::vector&): return the rotation angle 
                     and axis
		 11. rotationMatrix(TData, std::vector): create rotation matrix 
                     from angel and axis
                 12. extractEuler(): extact the Euler rotation angle in ZXY
                 13. calcEuler: extact the Euler rotation angle in ZXY
  	
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include <math.h>
#include <float.h>

#include "SJCRotateMatrix.h"
#include "SJCQuaternion.h"
#include "SJCUtility.h"

#ifndef M_PI
#define M_PI            3.14159265358979323846
#define M_PI_2          1.57079632679489661923
#define M_PI_4          0.78539816339744830962
#define M_1_PI          0.31830988618379067154
#define M_2_PI          0.63661977236758134308
#define M_SQRT2         1.41421356237309504880
#define M_SQRT1_2       0.70710678118654752440
#endif

#ifndef ANGLE_TOL
  #define ANGLE_TOL  0.002
#endif
#ifndef RAD_TO_DEG
  #define RAD_TO_DEG 57.29577951
#endif

//*************************************************************************
//
// * Return a matrix from to std::vector
//=========================================================================
SJCRotateMatrixd SJCRotateMatrixd::
outerProduct(const SJCVector3d &v1, const SJCVector3d &v2)
//=========================================================================
{
  return SJCRotateMatrixd(v1.x() * v2.x(), v1.x() * v2.y(), v1.x() * v2.z(),
			 v1.y() * v2.x(), v1.y() * v2.y(), v1.y() * v2.z(),
			 v1.z() * v2.x(), v1.z() * v2.y(), v1.z() * v2.z());
}

//*************************************************************************
//
// Return the rotation matrix from and rotation angle and axis
//=========================================================================
SJCRotateMatrixd::
SJCRotateMatrixd(const double angle, const SJCVector3d& axis)
//=========================================================================
{
  double 	cos_ang = cos(angle);
  double 	sin_ang = sin(angle);
  SJCVector3d	u = axis.normal();
  
  // Formulation taken from OpenGL Programming Guide
  SJCRotateMatrixd outer_prod = outerProduct(u,u);
  
  SJCRotateMatrixd s(             0.0, -u.z() * sin_ang,  u.y() * sin_ang,
		     u.z() * sin_ang,              0.0, -u.x() * sin_ang,
		    -u.y() * sin_ang,  u.x() * sin_ang,              0.0);
  
  data[0][0] = outer_prod(0,0) * ( 1 - cos_ang ) + cos_ang + s(0,0);
  data[0][1] = outer_prod(0,1) * ( 1 - cos_ang ) + s(0,1);
  data[0][2] = outer_prod(0,2) * ( 1 - cos_ang ) + s(0,2);
  data[1][0] = outer_prod(1,0) * ( 1 - cos_ang ) + s(1,0);
  data[1][1] = outer_prod(1,1) * ( 1 - cos_ang ) + cos_ang + s(1,1);
  data[1][2] = outer_prod(1,2) * ( 1 - cos_ang ) + s(1,2);
  data[2][0] = outer_prod(2,0) * ( 1 - cos_ang ) + s(2,0);
  data[2][1] = outer_prod(2,1) * ( 1 - cos_ang ) + s(2,1);
  data[2][2] = outer_prod(2,2) * ( 1 - cos_ang ) + cos_ang + s(2,2);
}



//*************************************************************************
//
// From the current matrix information, return the rotation angle and axis
//=========================================================================
void SJCRotateMatrixd::
toAngleAxis(double &angle, SJCVector3d &axis) const
//=========================================================================
{
  /*
  ** Let (x,y,z) be the unit-length axis and let A be an angle of rotation.
  ** The rotation matrix is R = I + sin(A)*P + (1-cos(A))*P^2 where
  ** I is the identity and
  **
  **       +-        -+
  **   P = |  0 +z -y |
  **       | -z  0 +x |
  **       | +y -x  0 |
  **       +-        -+
  **
  ** Some algebra will show that
  **
  **   cos(A) = (trace(R)-1)/2  and  R - R^t = 2*sin(A)*P
  **
  ** In the event that A = pi, R-R^t = 0 which prevents us from extracting
  ** the axis through P.  Instead note that R = I+2*P^2 when A = pi, so
  ** P^2 = (R-I)/2.  The diagonal entries of P^2 are x^2-1, y^2-1, and
  ** z^2-1. Solve for |x|, |y|, and |z| where the axis is (x,y,z).  There
  ** is a choice of sign on |y| and |z|.  I test |R*axis-axis|^2 for each
  ** of the four sign possibilities and return on the first squared length
  ** which is nearly zero.
  */

  double  trace = data[0][0] + data[1][1] + data[2][2];
  double  cs = 0.5*(trace-1.0);
  double  length;
  
  if ( -1 < cs )   {
    if ( cs < 1 )
      angle = acos(cs);
    else
      angle = 0.0;
  }
  else    {
    angle = M_PI;
  }

  axis[0] = data[2][1] - data[1][2];
  axis[1] = data[0][2] - data[2][0];
  axis[2] = data[1][0] - data[0][1];
  length = axis.length();
  
  if ( length > DBL_EPSILON )     {
    axis[0] /= length;
    axis[1] /= length;
    axis[2] /= length;
  }
  else {  /* angle is 0 or pi */ 
    
    if ( angle > 1.0 ) {/* any number strictly between 0 and pi works */ 
	        
      SJCVector3d	test;
      double  	temp;
      
      /* angle must be pi */
      if ( ( temp = 0.5 * ( 1.0 + data[0][0] ) ) < 0.0 )
	temp = 0.0;
      axis[0] = sqrt(temp);
      if ( ( temp = 0.5 * ( 1.0 + data[1][1] ) ) < 0.0 )
	temp = 0.0;
      axis[1] = sqrt(temp);
      if ( ( temp = 0.5 * ( 1.0 + data[2][2] ) ) < 0.0 )
	temp = 0.0;
      axis[2] = sqrt(temp);
      
      /* determine signs of axis components */
      test = ( (*this) * axis ) - axis;
      length = test * test;
      if ( length < DBL_EPSILON )
	    return;
      
      axis[1] = -axis[1];
      test = ( (*this) * axis ) - axis;
      length = test * test;
      if ( length < DBL_EPSILON )
	return;
      
      axis[2] = -axis[2];
      test = ( (*this) * axis ) - axis;
      length = test * test;
      if ( length < DBL_EPSILON )
	return;
      
      axis[1] = -axis[1];
      test = ( (*this) * axis ) - axis;
      length = test * test;
      if ( length < DBL_EPSILON )
	return;
    }
    else {
      /* Angle is zero, matrix is the identity, no unique axis, so
      ** return (1,0,0) for as good a guess as any.
      */
      axis[0] = 1.0;
      axis[1] = 0.0;
      axis[2] = 0.0;
    }
  }
}


//*************************************************************************
//
// * Output operator
//=========================================================================
std::ostream& operator<<(std::ostream& o, const SJCRotateMatrixd& m) 
//=========================================================================
{
  o << "[";
  for ( int i = 0 ; i < 3 ; i++ ) {
    o << "[";
    for ( int j = 0 ; j < 3 ; j++ ) {
      o << m.data[i][j];
      if ( j < 2 )
        o << ",";
    }
    o << "]";
    if ( i < 2 )
      o << ",";
  }
  return o;
}


//*************************************************************************
//
// * Extract the Euler angle in Z X Y
//=========================================================================
double *SJCRotateMatrixd::
extractEuler(void) 
//=========================================================================
{
  double xAngle, yAngle, zAngle;
  
  calcEuler( xAngle, yAngle, zAngle );	
  
  double *eulers = new double[3];
  eulers[0] = zAngle * RAD_TO_DEG;
  eulers[1] = xAngle * RAD_TO_DEG;
  eulers[2] = yAngle * RAD_TO_DEG;
  
  return( eulers );
}

//*************************************************************************
//
// * Extract the Euler angle in Z X Y
//=========================================================================
void SJCRotateMatrixd::
extractEuler( double *result ) 
//=========================================================================
{
  double xAngle, yAngle, zAngle;
  calcEuler( xAngle, yAngle, zAngle );
  
  result[0] = zAngle * RAD_TO_DEG;
  result[1] = xAngle * RAD_TO_DEG;
  result[2] = yAngle * RAD_TO_DEG;
}

//*************************************************************************
//
// * Extract the Euler angle in Z X Y
//=========================================================================
void SJCRotateMatrixd::
calcEuler( double &xAngle, double &yAngle, double &zAngle ) 
//=========================================================================
{
  xAngle = SJCASin( data[2][1] );
  double cosX = cos( xAngle );

  if( cosX < ANGLE_TOL ) {

    // Gimbal lock.  Shove all rotation to be around the Y axis.
    zAngle = 0;
    yAngle = SJCASin( -data[0][1] / data[2][1] );
  } else {
    yAngle = atan2( -data[2][0], data[2][2] );
    zAngle = atan2( -data[0][1], data[1][1] );
  }
}

//*************************************************************************
//
// * Return a matrix from to std::vector
//=========================================================================
SJCRotateMatrixf
SJCRotateMatrixf::outerProduct(const SJCVector3f &v1, const SJCVector3f &v2)
//=========================================================================
{
  return SJCRotateMatrixf(v1.x() * v2.x(), v1.x() * v2.y(), v1.x() * v2.z(),
			  v1.y() * v2.x(), v1.y() * v2.y(), v1.y() * v2.z(),
			  v1.z() * v2.x(), v1.z() * v2.y(), v1.z() * v2.z());
}


//*************************************************************************
//
// Return the rotation matrix from and rotation angle and axis
//=========================================================================
SJCRotateMatrixf::
SJCRotateMatrixf(const float angle, const SJCVector3f& axis)
//=========================================================================
{
  float 	cos_ang = cos(angle);
  float 	sin_ang = sin(angle);
  SJCVector3f	u = axis.normal();
  
  // Formulation taken from OpenGL Programming Guide
  SJCRotateMatrixf outer_prod = outerProduct(u,u);
  
  SJCRotateMatrixf s(0.0, -u.z() * sin_ang, u.y() * sin_ang,
		    u.z() * sin_ang, 0.0, -u.x() * sin_ang,
		    -u.y() * sin_ang, u.x() * sin_ang, 0.0);

  data[0][0] = outer_prod(0,0) * ( 1 - cos_ang ) + cos_ang + s(0,0);
  data[0][1] = outer_prod(0,1) * ( 1 - cos_ang ) + s(0,1);
  data[0][2] = outer_prod(0,2) * ( 1 - cos_ang ) + s(0,2);
  data[1][0] = outer_prod(1,0) * ( 1 - cos_ang ) + s(1,0);
  data[1][1] = outer_prod(1,1) * ( 1 - cos_ang ) + cos_ang + s(1,1);
  data[1][2] = outer_prod(1,2) * ( 1 - cos_ang ) + s(1,2);
  data[2][0] = outer_prod(2,0) * ( 1 - cos_ang ) + s(2,0);
  data[2][1] = outer_prod(2,1) * ( 1 - cos_ang ) + s(2,1);
  data[2][2] = outer_prod(2,2) * ( 1 - cos_ang ) + cos_ang + s(2,2);
}


//*************************************************************************
//
// From the current matrix information, return the rotation angle and axis
//=========================================================================
void SJCRotateMatrixf::
toAngleAxis(float &angle, SJCVector3f &axis) const
//=========================================================================
{
  /*
  ** Let (x,y,z) be the unit-length axis and let A be an angle of rotation.
  ** The rotation matrix is R = I + sin(A)*P + (1-cos(A))*P^2 where
  ** I is the identity and
  **
  **       +-        -+
  **   P = |  0 +z -y |
  **       | -z  0 +x |
  **       | +y -x  0 |
  **       +-        -+
  **
  ** Some algebra will show that
  **
  **   cos(A) = (trace(R)-1)/2  and  R - R^t = 2*sin(A)*P
  **
  ** In the event that A = pi, R-R^t = 0 which prevents us from extracting
  ** the axis through P.  Instead note that R = I+2*P^2 when A = pi, so
  ** P^2 = (R-I)/2.  The diagonal entries of P^2 are x^2-1, y^2-1, and
  ** z^2-1. Solve for |x|, |y|, and |z| where the axis is (x,y,z).  There
  ** is a choice of sign on |y| and |z|.  I test |R*axis-axis|^2 for each
  ** of the four sign possibilities and return on the first squared length
  ** which is nearly zero.
  */

  float  trace = data[0][0] + data[1][1] + data[2][2];
  float  cs = 0.5*(trace-1.0);
  float  length;
  
  if ( -1 < cs )   {
    if ( cs < 1 )
      angle = acos(cs);
    else
      angle = 0.0;
  }
  else    {
    angle = M_PI;
  }

  axis[0] = data[2][1] - data[1][2];
  axis[1] = data[0][2] - data[2][0];
  axis[2] = data[1][0] - data[0][1];
  length = axis.length();
  
  if ( length > DBL_EPSILON )     {
    axis[0] /= length;
    axis[1] /= length;
    axis[2] /= length;
  }
  else {  /* angle is 0 or pi */ 
    
    if ( angle > 1.0 ) {/* any number strictly between 0 and pi works */ 
	        
      SJCVector3f test;
      float  	  temp;
      
      /* angle must be pi */
      if ( ( temp = 0.5 * ( 1.0 + data[0][0] ) ) < 0.0 )
	temp = 0.0;
      axis[0] = sqrt(temp);
      if ( ( temp = 0.5 * ( 1.0 + data[1][1] ) ) < 0.0 )
	temp = 0.0;
      axis[1] = sqrt(temp);
      if ( ( temp = 0.5 * ( 1.0 + data[2][2] ) ) < 0.0 )
	temp = 0.0;
      axis[2] = sqrt(temp);
      
      /* determine signs of axis components */
      test = ( (*this) * axis ) - axis;
      length = test * test;
      if ( length < DBL_EPSILON )
	    return;
      
      axis[1] = -axis[1];
      test = ( (*this) * axis ) - axis;
      length = test * test;
      if ( length < DBL_EPSILON )
	return;
      
      axis[2] = -axis[2];
      test = ( (*this) * axis ) - axis;
      length = test * test;
      if ( length < DBL_EPSILON )
	return;
      
      axis[1] = -axis[1];
      test = ( (*this) * axis ) - axis;
      length = test * test;
      if ( length < DBL_EPSILON )
	return;
    }
    else {
      /* Angle is zero, matrix is the identity, no unique axis, so
      ** return (1,0,0) for as good a guess as any.
      */
      axis[0] = 1.0;
      axis[1] = 0.0;
      axis[2] = 0.0;
    }
  }
}


//*************************************************************************
//
// * Output operator
//=========================================================================
std::ostream& operator<<(std::ostream& o, const SJCRotateMatrixf& m) 
//=========================================================================
{
  o << "[";
  for ( int i = 0 ; i < 3 ; i++ ) {
    o << "[";
    for ( int j = 0 ; j < 3 ; j++ ) {
      o << m.data[i][j];
      if ( j < 2 )
        o << ",";
    }
    o << "]";
    if ( i < 2 )
      o << ",";
  }
  return o;
}


//*************************************************************************
//
// * Extract the Euler angle in Z X Y
//=========================================================================
float *SJCRotateMatrixf::
extractEuler(void) 
//=========================================================================
{
  float xAngle, yAngle, zAngle;
  
  calcEuler( xAngle, yAngle, zAngle );	
  
  float *eulers = new float[3];
  eulers[0] = zAngle * RAD_TO_DEG;
  eulers[1] = xAngle * RAD_TO_DEG;
  eulers[2] = yAngle * RAD_TO_DEG;
  
  return( eulers );
}

//*************************************************************************
//
// * Extract the Euler angle in Z X Y
//=========================================================================
void SJCRotateMatrixf::
extractEuler( float *result ) 
//=========================================================================
{
  float xAngle, yAngle, zAngle;
  calcEuler( xAngle, yAngle, zAngle );
  
  result[0] = zAngle * RAD_TO_DEG;
  result[1] = xAngle * RAD_TO_DEG;
  result[2] = yAngle * RAD_TO_DEG;
}

//*************************************************************************
//
// * Extract the Euler angle in Z X Y
//=========================================================================
void SJCRotateMatrixf::
calcEuler( float &xAngle, float &yAngle, float &zAngle ) 
//=========================================================================
{
  xAngle = SJCASin( data[2][1] );
  float cosX = cos( xAngle );

  if( cosX < ANGLE_TOL ) {

    // Gimbal lock.  Shove all rotation to be around the Y axis.
    zAngle = 0;
    yAngle = SJCASin( -data[0][1] / data[2][1] );
  } else {
    yAngle = atan2( -data[2][0], data[2][2] );
    zAngle = atan2( -data[0][1], data[1][1] );
  }
}


