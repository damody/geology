/************************************************************************
     Main File:

     File:        SimTransform.cpp

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     The rigid body transformation
    
     Contructor:
		 0 paras: no scale, no rotation, no translation transform 
                          matrix 
		 12 Real: a00, a01, a02, a03, a20....
		 Quaternion, std::vector: rotation and then translate transform
     Function:
		 x(), y(), z(): return 0, 1, 2 row's Vector4
		 x(Vector4), y(Vector4), z(Vector4): set 0, 1, 2 row's value
		 (i, j) = aij
		 [i] = return ith row
		 = (transform): assign the transform to this
		 * (transfrom A): this * A 
		 * (vector): this * V
		 * (Vector4): this * V4
		 % (vector): extract(this)'s scale and rotation * V
		 toIdentity(): create identity transform
		 toComponent(angle, axis, translate): extract the rotation and
                                                      translate component
		 inverse(): return the inverse transform
		 invert(): let the transform become invert
		 rotateTransform(angle, axis): return transform after this 
                                               rotation
		 scaleTransform(angle, axis): return transform after this 
                                              scale
		 translateTransform(angle, axis): return transform after 
                                                  this translation    

     Compiler:    g++

     Platform:    Linux
*************************************************************************/


#include <math.h>
#include <float.h>
#include "SJCRigidTransform.h"

#ifndef M_PI
#define M_PI            3.14159265358979323846
#define M_PI_2          1.57079632679489661923
#define M_PI_4          0.78539816339744830962
#define M_1_PI          0.31830988618379067154
#define M_2_PI          0.63661977236758134308
#define M_SQRT2         1.41421356237309504880
#define M_SQRT1_2       0.70710678118654752440
#endif

//****************************************************************************
//
// Constructor
//=============================================================================
SJCRigidTransformd::
SJCRigidTransformd(const SJCQuaterniond &q, const SJCVector3d &v)
//=============================================================================
{
  double	xx = q.x() * q.x();
  double	yy = q.y() * q.y();
  double	zz = q.z() * q.z();
  
  double	xy = q.x() * q.y();
  double	xz = q.x() * q.z();
  double	yz = q.y() * q.z();
  
  double	wx = q.w() * q.x();
  double	wy = q.w() * q.y();
  double	wz = q.w() * q.z();
  
  data[0][0] = 1 - 2*yy - 2*zz;
  data[0][1] = 2*xy - 2*wz;
  data[0][2] = 2*xz + 2*wy;
  data[0][3] = v.x();
  data[1][0] = 2*xy + 2*wz;
  data[1][1] = 1 - 2*xx - 2*zz;
  data[1][2] = 2*yz - 2*wx;
  data[1][3] = v.y();
  data[2][0] = 2*xz - 2*wy;
  data[2][1] = 2*yz + 2*wx;
  data[2][2] = 1 - 2*xx - 2*yy;
  data[2][3] = v.z();
}


//****************************************************************************
//
// Create the matrix from output product with two std::vector
//=============================================================================
SJCRigidTransformd
SJCRigidTransformd::outerProduct(const SJCVector3d &v1, const SJCVector3d &v2)
//=============================================================================
{
  return SJCRigidTransformd(
		v1.x() * v2.x(), v1.x() * v2.y(), v1.x() * v2.z(), 0.0,
		v1.y() * v2.x(), v1.y() * v2.y(), v1.y() * v2.z(), 0.0,
		v1.z() * v2.x(), v1.z() * v2.y(), v1.z() * v2.z(), 0.0);
}


//****************************************************************************
//
// * Do the rotation transform
//=============================================================================
SJCRigidTransformd
SJCRigidTransformd::rotateTransform(const double angle, 
				    const SJCVector3d& axis)
//=============================================================================
{
  double 	cos_ang = cos(angle);
  double 	sin_ang = sin(angle);
  SJCVector3d	u = axis.normal();
  
  // Formulation taken from OpenGL Programming Guide
  SJCRigidTransformd outer_prod = outerProduct(u,u);
  
  SJCRigidTransformd s(0.0, -u.z() * sin_ang, u.y() * sin_ang, 0.0,
		      u.z() * sin_ang, 0.0, -u.x() * sin_ang, 0.0,
		      -u.y() * sin_ang, u.x() * sin_ang, 0.0, 0.0);
  
  return SJCRigidTransformd(
		outer_prod(0,0) * ( 1 - cos_ang ) + cos_ang + s(0,0),
		outer_prod(0,1) * ( 1 - cos_ang ) + s(0,1),
		outer_prod(0,2) * ( 1 - cos_ang ) + s(0,2),
		0.0,
		outer_prod(1,0) * ( 1 - cos_ang ) + s(1,0),
		outer_prod(1,1) * ( 1 - cos_ang ) + cos_ang + s(1,1),
		outer_prod(1,2) * ( 1 - cos_ang ) + s(1,2),
		0.0,
		outer_prod(2,0) * ( 1 - cos_ang ) + s(2,0),
		outer_prod(2,1) * ( 1 - cos_ang ) + s(2,1),
		outer_prod(2,2) * ( 1 - cos_ang ) + cos_ang + s(2,2),
		0.0);
}


//****************************************************************************
//
// * Decompose the transform matrix into rotation and translation
//=============================================================================
void SJCRigidTransformd::
toComponents(double &angle, SJCVector3d &axis, SJCVector3d &tran)const
//=============================================================================
{
  tran.x(data[0][3]);
  tran.y(data[1][3]);
  tran.z(data[2][3]);
  
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
  else   {
    angle = M_PI;
  }
  
  axis[0] = data[2][1] - data[1][2];
  axis[1] = data[0][2] - data[2][0];
  axis[2] = data[1][0] - data[0][1];
  length = axis.length();
  
  if ( length > DBL_EPSILON )   {
    axis[0] /= length;
    axis[1] /= length;
    axis[2] /= length;
  }
  else { /* angle is 0 or pi */
    
    if ( angle > 1.0 ) { /* any number strictly between 0 and pi works */
      
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
    else  {
      /* Angle is zero, matrix is the identity, no unique axis, so
      ** return (1,0,0) for as good a guess as any.
      */
      axis[0] = 1.0;
      axis[1] = 0.0;
      axis[2] = 0.0;
    }
  }
}


//**************************************************************************** 
// 
// * Matrix inversion
//============================================================================ 
void SJCRigidTransformd::
invert(void) 
//============================================================================ 
{ 
  *this = inverse(); 
} 
 

 
//*************************************************************************** 
// 
// * Output operator
//============================================================================ 
std::ostream& operator<<(std::ostream& o, const SJCRigidTransformd& m) 
//============================================================================ 
{ 
  o << "["; 
  for ( int i = 0 ; i < 3 ; i++ ) { 
    o << "["; 
    for ( int j = 0 ; j <= 3 ; j++ ) { 
      o << m.data[i][j]; 
      if ( j < 3 ) 
	o << ","; 
    } // end of for j
    o << "]"; 
    if ( i < 2 ) 
      o << ","; 
  } // end of for i
  return o; 
}



//****************************************************************************
//
// Constructor
//=============================================================================
SJCRigidTransformf::
SJCRigidTransformf(const SJCQuaternionf &q, const SJCVector3f &v)
//=============================================================================
{
  float	xx = q.x() * q.x();
  float	yy = q.y() * q.y();
  float	zz = q.z() * q.z();
  
  float	xy = q.x() * q.y();
  float	xz = q.x() * q.z();
  float	yz = q.y() * q.z();
  
  float	wx = q.w() * q.x();
  float	wy = q.w() * q.y();
  float	wz = q.w() * q.z();
  
  data[0][0] = 1 - 2*yy - 2*zz;
  data[0][1] = 2*xy - 2*wz;
  data[0][2] = 2*xz + 2*wy;
  data[0][3] = v.x();
  data[1][0] = 2*xy + 2*wz;
  data[1][1] = 1 - 2*xx - 2*zz;
  data[1][2] = 2*yz - 2*wx;
  data[1][3] = v.y();
  data[2][0] = 2*xz - 2*wy;
  data[2][1] = 2*yz + 2*wx;
  data[2][2] = 1 - 2*xx - 2*yy;
  data[2][3] = v.z();
}


//****************************************************************************
//
// Create the matrix from output product with two std::vector
//=============================================================================
SJCRigidTransformf
SJCRigidTransformf::outerProduct(const SJCVector3f &v1, const SJCVector3f &v2)
//=============================================================================
{
  return SJCRigidTransformf(
		v1.x() * v2.x(), v1.x() * v2.y(), v1.x() * v2.z(), 0.0,
		v1.y() * v2.x(), v1.y() * v2.y(), v1.y() * v2.z(), 0.0,
		v1.z() * v2.x(), v1.z() * v2.y(), v1.z() * v2.z(), 0.0);
}


//****************************************************************************
//
// * Do the rotation transform
//=============================================================================
SJCRigidTransformf
SJCRigidTransformf::rotateTransform(const float angle, const SJCVector3f& axis)
//=============================================================================
{
  float 	cos_ang = cos(angle);
  float 	sin_ang = sin(angle);
  SJCVector3f	u = axis.normal();
  
  // Formulation taken from OpenGL Programming Guide
  SJCRigidTransformf outer_prod = outerProduct(u,u);
  
  SJCRigidTransformf s(0.0, -u.z() * sin_ang, u.y() * sin_ang, 0.0,
		      u.z() * sin_ang, 0.0, -u.x() * sin_ang, 0.0,
		      -u.y() * sin_ang, u.x() * sin_ang, 0.0, 0.0);
  
  return SJCRigidTransformf(
		outer_prod(0,0) * ( 1 - cos_ang ) + cos_ang + s(0,0),
		outer_prod(0,1) * ( 1 - cos_ang ) + s(0,1),
		outer_prod(0,2) * ( 1 - cos_ang ) + s(0,2),
		0.0,
		outer_prod(1,0) * ( 1 - cos_ang ) + s(1,0),
		outer_prod(1,1) * ( 1 - cos_ang ) + cos_ang + s(1,1),
		outer_prod(1,2) * ( 1 - cos_ang ) + s(1,2),
		0.0,
		outer_prod(2,0) * ( 1 - cos_ang ) + s(2,0),
		outer_prod(2,1) * ( 1 - cos_ang ) + s(2,1),
		outer_prod(2,2) * ( 1 - cos_ang ) + cos_ang + s(2,2),
		0.0);
}


//****************************************************************************
//
// * Decompose the transform matrix into rotation and translation
//=============================================================================
void
SJCRigidTransformf::toComponents(float &angle, SJCVector3f &axis,
                                SJCVector3f &tran)const
//=============================================================================
{
  tran.x(data[0][3]);
  tran.y(data[1][3]);
  tran.z(data[2][3]);
  
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
  else   {
    angle = M_PI;
  }
  
  axis[0] = data[2][1] - data[1][2];
  axis[1] = data[0][2] - data[2][0];
  axis[2] = data[1][0] - data[0][1];
  length = axis.length();
  
  if ( length > DBL_EPSILON )   {
    axis[0] /= length;
    axis[1] /= length;
    axis[2] /= length;
  }
  else { /* angle is 0 or pi */
    
    if ( angle > 1.0 ) { /* any number strictly between 0 and pi works */
      
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
    else  {
      /* Angle is zero, matrix is the identity, no unique axis, so
      ** return (1,0,0) for as good a guess as any.
      */
      axis[0] = 1.0;
      axis[1] = 0.0;
      axis[2] = 0.0;
    }
  }
}


//**************************************************************************** 
// 
// * Matrix inversion
//============================================================================ 
void SJCRigidTransformf::
invert(void) 
//============================================================================ 
{ 
  *this = inverse(); 
} 
 

 
//*************************************************************************** 
// 
// * Output operator
//============================================================================ 
std::ostream& operator<<(std::ostream& o, const SJCRigidTransformf& m) 
//============================================================================ 
{ 
  o << "["; 
  for ( int i = 0 ; i < 3 ; i++ ) { 
    o << "["; 
    for ( int j = 0 ; j <= 3 ; j++ ) { 
      o << m.data[i][j]; 
      if ( j < 3 ) 
	o << ","; 
    } // end of for j
    o << "]"; 
    if ( i < 2 ) 
      o << ","; 
  } // end of for i
  return o; 
}



