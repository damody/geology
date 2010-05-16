/************************************************************************
     Main File:

     File:        SJCQuaternion.cpp

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     The Quaternion operation
	
     Contructor:
		 1. 0 paras: 0, 0, 0, 1 
		 2. 4 double: x, y, z, w
		 3. double, std::vector: rotation angle and axis 
		 4. 1 matrix: rotation matrix
		 5. 1 Quarternion:
                 6. double, double, double: creates a quaternion
                    equivalent to the rotation specified by the given 
                  . sequence of euler angles in YXZ order.  
                 8. double, double, double, character: Euler roation
                 9. std::vector, std::vector: rotation to align two std::vector
                10. std::vector, std::vector, std::vector: rotate from to to by axis
     Function:
		 1. x(), y(), z(), w(): return x, y, z, w value
		 2. x(double), y(double), z(double), w(double0: 
                    set x, y, z, w value
                 3. * (Quarternion): return this * Q
		 4. *=(Quarternion): this = this * Q
		 5. * (vector V): this * (0, v)
                 6. -(): inverse

 		 7. set(angle, axis): set the quarternion from rotation 
                    angle and axis
                 8. set(src, dst): calculate the q to rotate from src to dst
                 9. set(src, dst, rot_axis): calculate the q to rotate from 
                    src to dst by using the rot_axis
                10. set(matrix): set up q from matrix
                11. set(z, x, y): set up q from euler zxy rotation
                12. set(w, x, y, z): set up q from four real number
                13. set(q): set up q from another q
                14. set(a1, a2, a3, order): 
                15. SetCoord(x, up): set up the coordinate from x and up std::vector

  		16. inverse(): calculate the inverse function
		17. normalize(): let normal become 1
		18. identity(): create identity rotation quarternion

                19. distance(Quaternion): Returns the (minimum) 
                    great-arc distance to the given quaternion.;
    
		20. toAngleAxis(double&, std::vector&): return the rotation 
                    angle and axis
		21. toMatrix(matrix): return rotation matrix
                22. toLogMap(array): extract information to logmap
                23. toEulers(array): extract ZXY rotation
                24. Alignment(source, destination)
                25. multiply(char, angle): rotate about X, Y, Z axis
                26. mult(quaternion): multiply another quaternion
                27. slerp(Quaternion, SJCQuaternion, double, Quaternion):
                    slerp interpolation the quaternion
                28. pickClosest(Quaternion): Determines which of the two 
                    equivalent quaternions represented by this rotation 
                    is closest to the given quaternion.
                29. switchSide(Quaternion, Quaternion, Quaternion):
                    Fills in r such that b*r = a*b

                30. glMatrix(array): output to glMatrix format
  
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include <stdio.h>  // fprintf()
#include <stdlib.h> // exit()
#include <math.h>   // acos()
#include <float.h>  // DBL_EPSILON

#include "SJCQuaternion.h"
#include "SJCRotateMatrix.h"
#include "SJCException.h"
#include "SJCUtility.h"


#define SLERP_TOL 1e-10

#define SIN_ANGLE_TOL 1e-6

//****************************************************************************
//
// * Constructor from a roatio axis and a angle in radian
//=============================================================================
SJCQuaterniond::
SJCQuaterniond(const double angle, const SJCVector3d &axis )
//=============================================================================
{
  double  s_rad = sin(angle * 0.5);
  x_ = axis.x() * s_rad;
  y_ = axis.y() * s_rad;
  z_ = axis.z() * s_rad;
  w_ = cos(angle * 0.5);
}


//*****************************************************************************
//
// * Set the quaternion to the roation about the axis
//============================================================================
void SJCQuaterniond::
set(double angle, const SJCVector3d &axis) 
//============================================================================
{
  double  s_rad = sin(angle * 0.5);
  x_ = axis.x() * s_rad;
  y_ = axis.y() * s_rad;
  z_ = axis.z() * s_rad;
  w_ = cos(angle * 0.5);
}

//*****************************************************************************
//
// * Set the coordinates to make this Quaternion a rotation that takes the 
//   first std::vector into the second.
//============================================================================
void SJCQuaterniond::
set(const SJCVector3d& s, const SJCVector3d& d) 
//============================================================================
{

  SJCVector3d source = s.normal();
  SJCVector3d destination = d.normal();

  // Compute the dot product and normalized cross product
  double dotProd = source * destination;
  if( dotProd < -1 ) {
    dotProd = -1;
  } 
  else if( dotProd > 1 ) {
    dotProd = 1;
  }

  // Calculate the cross product which should be the rotation axis
  SJCVector3d crossProd = source % destination;
  double norm = crossProd.length();

  if( norm < SJC_EPSILON2 ) { // When two axis are paralley
    //**********************************************************************
    // The std::vectors point nearly along the same line.  
    // 1. If the angle between them is small (closer to 0 than PI),
    //    then set this to the identity rotation.
    // 2. Otherwise assume
    //    the std::vectors are 180 degrees apart and pick an arbitrary axis 
    //    perpendicular to the
    //    line that passes through them.
    //***********************************************************************

    if( dotProd > 0 ) { // Point in the same direction
      x_ = y_ = z_ =0.0;
      w_ = 1.0;
      return;
    }  // end of if
    else {
      // Set up the arbitrary rotation axis to the biggest component of x, y, z
      SJCVector3d secondaryAxis(0, 0, 0);

      if( (SJCAbs(source[0]) <= SJCAbs(source[1]))  && 
	  (SJCAbs(source[0]) <= SJCAbs(source[2])) ) {
	secondaryAxis[0] = 1;
      } 
      else if( (SJCAbs(source[1]) <= SJCAbs(source[0])) && 
	       (SJCAbs(source[1]) <= SJCAbs(source[2])) ) {
	secondaryAxis[1] = 1;
      } 
      else {
	secondaryAxis[2] = 1;
      } 

      // Calcualte the rotation axis 
      crossProd = source % secondaryAxis;
      norm = crossProd.length();			
      
      // That should have worked
      if( norm < SJC_EPSILON2 ) {
	std::cerr << "Warning: Quaternion::setCoords( double *vec1, double *vec2 )"
             << " is indeterminate" << std::endl;

	x_ = y_ = z_ =0.0;
	w_ = 1.0;
	return;
      }// end of if norm < SJC_EPSILON2 (protection against the cross
    }// end of else 
  }// end of if norm < SJC_EPSILON2
  
  crossProd.normalize();
  
  // Fill in the values for this quaternion.
  double axisLength = sqrt( (1.0 - dotProd) / 2.0 );
  
  w_ = sqrt( (1.0 + dotProd) / 2.0);
  x_ = crossProd[0] * axisLength;
  y_ = crossProd[1] * axisLength;
  z_ = crossProd[2] * axisLength;
  
  // Just to be sure
  normalize();
}


//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaterniond::
set( const SJCVector3d& s, const SJCVector3d& d, const SJCVector3d& r ) 
//============================================================================
{

  // Project each std::vector onto the plane perpendicular to the rotation axis
  SJCVector3d source = s.normal();
  SJCVector3d destination = d.normal();
  SJCVector3d rotAxis = r.normal();
  
  double dotProd1 = source * rotAxis;

  double dotProd2 = destination * rotAxis;

  SJCVector3d v1 = source - dotProd1 * rotAxis;
  SJCVector3d v2 = destination - dotProd2 * rotAxis;


  v1.normalize();
  v2.normalize();
  
  double angle = SJCACos(v1 * v2);
  
  SJCVector3d crossProd =  v1 % v2;
  if( (crossProd * rotAxis) < 0 ) {
    angle = -angle;
  }

  set( angle, rotAxis);
}


//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaterniond::
set(const SJCRotateMatrixd &m) 
//============================================================================
{
  double  tr, s, q[4];
  int     i, j, k;
  
  int     nxt[3] = {1, 2, 0};
  
  tr = m[0][0] + m[1][1] + m[2][2];

  // check the diagonal
  if ( tr > 0.0 )  {
//     s = sqrt (tr + 1.0);
//     w_ = s / 2.0;
//     s = 0.5 / s;
//     x_ = (m[1][2] - m[2][1]) * s;
//     y_ = (m[2][0] - m[0][2]) * s;
//     z_ = (m[0][1] - m[1][0]) * s;

    s = sqrt (tr + 1.0);
    w_ = s / 2.0;
    s = 0.5 / s;
    x_ = (m[2][1] - m[1][2]) * s;
    y_ = (m[0][2] - m[2][0]) * s;
    z_ = (m[1][0] - m[0][1]) * s;
  }
  else   {                
    // diagonal is negative
    i = 0;
    if ( m[1][1] > m[0][0] )
      i = 1;
    if ( m[2][2] > m[i][i] )
      i = 2;
    j = nxt[i];
    k = nxt[j];
    
    s = sqrt((m[i][i] - (m[j][j] + m[k][k])) + 1.0);
    
    q[i] = s * 0.5;
    
    if ( s > DBL_EPSILON )
      s = 0.5 / s;
    
    //    q[3] = (m[j][k] - m[k][j] ) * s;
    q[3] = (m[k][j] - m[j][k]) * s;
    q[j] = (m[i][j] + m[j][i]) * s;
    q[k] = (m[i][k] + m[k][i]) * s;
    
    x_ = q[0];
    y_ = q[1];
    z_ = q[2];
    w_ = q[3];

  }
  normalize();
  
}

//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaterniond::
SetCoord(const SJCVector3d x_dir, const SJCVector3d up) 
//============================================================================
{
  SJCVector3d X, Y, Z, UP;

  X  = x_dir.normal();
  UP = up.normal();   


  double cos_x_up = up * x_dir;
  if(WithinEpsilon(cos_x_up, 1.f)) {
    double angle = -90.f * SJC_DEG_TO_RAD;
    set(angle, SJCConstants::SJC_vYAxis3d);
    return;
  }


  Y = UP % X;   Y.normalize();
  Z = X  % Y;   Z.normalize();

 
  double   m[3][3];

  m[0][0] = X.x(); m[0][1] = X.y(); m[0][2] = X.z();
  m[1][0] = Y.x(); m[1][1] = Y.y(); m[1][2] = Y.z();
  m[2][0] = Z.x(); m[2][1] = Z.y(); m[2][2] = Z.z();
 
  double  tr, s, q[4];
  int    i, j, k;
  
  int    nxt[3] = {1, 2, 0};
  
  tr = m[0][0] + m[1][1] + m[2][2];
  
  // check the diagonal
  if ( tr > 0.0 )  {
    s = sqrt (tr + 1.0);
    w_ = s / 2.0;
    s = 0.5 / s;
    x_ = (m[1][2] - m[2][1]) * s;
    y_ = (m[2][0] - m[0][2]) * s;
    z_ = (m[0][1] - m[1][0]) * s;
  }
  else   {                
    // diagonal is negative
    i = 0;
    if ( m[1][1] > m[0][0] )
      i = 1;
    if ( m[2][2] > m[i][i] )
      i = 2;
    j = nxt[i];
    k = nxt[j];
    
    s = sqrt((m[i][i] - (m[j][j] + m[k][k])) + 1.0);
    
    q[i] = s * 0.5;
    
    if ( s > DBL_EPSILON )
      s = 0.5 / s;
    
    q[3] = (m[j][k] - m[k][j]) * s;
    q[j] = (m[i][j] + m[j][i]) * s;
    q[k] = (m[i][k] + m[k][i]) * s;
    
    x_ = q[0];
    y_ = q[1];
    z_ = q[2];
    w_ = q[3];
  }
  normalize();
}


//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaterniond::
set(const double Z, const double X, const double Y) 
//============================================================================
{
  w_ = 1.0;
  x_ = y_ = z_ = 0.0;

  multiply( 'z', Z);
  multiply( 'x', X);
  multiply( 'y', Y);

  normalize();
}

//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaterniond::
set(const double angle1, const double angle2, const double angle3, 
    const char* order)
//============================================================================
{
  w_ = 1.0;
  x_ = y_ = z_ = 0.0;

  switch(order[0]){
    case 'x':
    case 'X':
      multiply( 'x', angle1);
      break;

    case 'y':
    case 'Y':
      multiply( 'y', angle1);
      break;

    case 'z':
    case 'Z':
      multiply( 'z', angle1);
      break;
    default:
      throw new SJCException("Quaternion: no such order");
  }

  switch(order[1]){
    case 'x':
    case 'X':
      multiply( 'x', angle2);
      break;

    case 'y':
    case 'Y':
      multiply( 'y', angle2);
      break;

    case 'z':
    case 'Z':
      multiply( 'z', angle2);
      break;
    default:
      throw new SJCException("Quaternion: no such order");
  }
  switch(order[2]){
    case 'x':
    case 'X':
      multiply( 'x', angle3);
      break;

    case 'y':
    case 'Y':
      multiply( 'y', angle3);
      break;

    case 'z':
    case 'Z':
      multiply( 'z', angle3);
      break;
    default:
      throw new SJCException("Quaternion: no such order");
  }
  normalize();
}

//*****************************************************************************
//
// * Normalize the quaternion
//============================================================================
void SJCQuaterniond::
normalize(void) 
//============================================================================
{
  double mag = sqrt(x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_);
  if ( mag == 0.0 )
    throw new SJCException("SJCQuaterniond::normalize: Zero length std::vector");
  w_ /= mag;
  x_ /= mag;
  y_ /= mag;
  z_ /= mag;
}

//*****************************************************************************
//
// * Normalize the quaternion
//============================================================================
SJCQuaterniond SJCQuaterniond::
Normal (void) const
//=============================================================================
{
  double mag = sqrt(x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_);
  if ( mag == 0.0 )
    throw new SJCException("SJCQuaterniond::normalize: Zero length std::vector");
  double factor = 1.0f / mag;
  return SJCQuaterniond(x_ * factor, y_ * factor, z_ * factor, w_ * mag);
}
//*****************************************************************************
//
// * The distance between two quaternion
//============================================================================
double SJCQuaterniond::
distance(SJCQuaterniond& quat ) 
//============================================================================
{
  SJCQuaterniond quatCopy( quat );
  
  double dotProd = 
    w_ * quat.w() + x_ * quat.x() + y_ * quat.y() + z_ * quat.z();
   
  // If the antipode is closer, use it instead.
  if( dotProd < 0 ) {
    quatCopy.set(quatCopy.w() * -1, quatCopy.x() * -1, 
		 quatCopy.y() * -1, quatCopy.z() * -1);
  }
  
  // Represent this Quaternion as an offset from the given Quaternion.
  quatCopy.inverse();
  quatCopy *= *this;

  double angle = SJCACos(quatCopy.w()) * 2;
  
  if( angle > M_PI ) {
    angle -= M_2PI;
  }
  
  return( fabs(angle) );
}

//*****************************************************************************
//
// * Transform to the constant axis
//============================================================================
void SJCQuaterniond::
toAngleAxis(double &angle, SJCVector3d &axis) const
//============================================================================
{
  double  half_angle;
  double  sin_half_angle;
  
  if ( w_ > 1.0 )
    half_angle = 0.0;
  else
    half_angle = acos(w_);
  sin_half_angle = sin(half_angle);
  angle = half_angle * 2;
  if ( sin_half_angle > -DBL_EPSILON && sin_half_angle < DBL_EPSILON )  {
    axis.x(0.0);
    axis.y(0.0);
    axis.z(1.0);
  }
  else   {
    sin_half_angle = 1.0 / sin_half_angle;
    axis.x(x_ * sin_half_angle);
    axis.y(y_ * sin_half_angle);
    axis.z(z_ * sin_half_angle);
  }
}

//*****************************************************************************
//
// * Extract the log map information
//   Log of the quaternion.  Define as log(q) = v * a, where 
//   q = [cos(a),v * sin(a)].
//
//============================================================================
void SJCQuaterniond::
toLogMap( double *result ) 
//============================================================================
{
  double sinAngle = sqrt( 1.0 - w_ * w_ );

  if( sinAngle < SIN_ANGLE_TOL ) {
    memset( result, 0, 3 * sizeof(double) );
  }// end of if
  else {
    double angle = SJCACos( w() ) * 2.0;
   
    // SJCACos returns a number between 0 and PI, 
    // but we would like |angle| <= PI
    if( angle > M_PI ) {
      angle -= M_2PI;
    }
    
    result[0] = x() / sinAngle * angle;
    result[1] = y() / sinAngle * angle;
    result[2] = z() / sinAngle * angle;
  } // end of else
}

//*****************************************************************************
//
// * Extract the Euler information
//============================================================================
void SJCQuaterniond::
toEulers( double *result ) 
//============================================================================
{
  SJCRotateMatrixd matrix = toMatrix();
  matrix.extractEuler(result);
}

//*****************************************************************************
//
// * Exponential of the Quaternion
//============================================================================
SJCQuaterniond SJCQuaterniond::
Exp(void) const
//============================================================================
{
  double angle = SJCACos(w_);

  if (angle > 0.f)   {
    double temp = sinf(angle) / angle;
    return SJCQuaterniond(x_ * temp, y_ * temp, z_ * temp, cos(angle));
  }// if
  else
    return SJCQuaterniond((double)0.f, (double)0.f, (double)0.f, 
			 cos(angle));
}// Exp

//*****************************************************************************
//
// * Transform to the constant axis
//============================================================================
SJCQuaterniond SJCQuaterniond::
Alignment(const SJCVector3d& source, const SJCVector3d& destination)
//=============================================================================
{
  SJCVector3d rotationAxis = source % destination;
  rotationAxis.normalize();
  
  SJCVector3d referenceVector = rotationAxis % source;
  referenceVector.normalize();

  double cosAB = destination * source;
  double angle = acos(cosAB);

  if( referenceVector * destination < 0)
    angle = -angle;

  return SJCQuaterniond(angle, rotationAxis);
}

//*****************************************************************************
//
// * Multiplication
//============================================================================
void SJCQuaterniond::
multiply( char axis, double angle) 
//============================================================================
{
  SJCVector3d rotAxis;
  switch(axis){
    case 'x':
    case 'X':
      rotAxis.set(1, 0, 0);
      break;

    case 'y':
    case 'Y':
      rotAxis.set(0, 1, 0);
      break;

    case 'z':
    case 'Z':
      rotAxis.set(0, 0, 1);
      break;
    default:
      throw new SJCException("Quaternion: no such order");
  }

  SJCQuaterniond quat(angle, rotAxis);  

  mult(quat);

  normalize();
}
//****************************************************************************
//
// Mutiply it with another quaternion, not following lucas
//============================================================================
void SJCQuaterniond::
mult( const SJCQuaterniond& other)
//============================================================================
{
  double	a = (w_ + x_)*(other.w_ + other.x_);
  double	b = (z_ - y_)*(other.y_ - other.z_);
  double	c = (w_ - x_)*(other.y_ + other.z_); 
  double	d = (y_ + z_)*(other.w_ - other.x_);
  double	e = (x_ + z_)*(other.x_ + other.y_);
  double	f = (x_ - z_)*(other.x_ - other.y_);
  double	g = (w_ + y_)*(other.w_ - other.z_);
  double	h = (w_ - y_)*(other.w_ + other.z_);

  w_ = a - (e + f + g + h) * 0.5;
  x_ = c + (e - f + g - h) * 0.5;
  y_ = d + (e - f - g + h) * 0.5;
  z_ = b + (-e - f + g + h) * 0.5;
  
}

//*****************************************************************************
//
// * Slerp operation
//============================================================================
void SJCQuaterniond::
slerp( const SJCQuaterniond& q1, const SJCQuaterniond& q2,  double u, 
       SJCQuaterniond& result ) 
//============================================================================
{
  double dotProd = 
      q1.w() * q2.w() + q1.x() * q2.x() + q1.y() * q2.y() + q1.z() * q2.z();

  double theta;
  
  if( dotProd < 0 ) {
    theta = SJCACos(-dotProd);
  }
  else {
    theta = SJCACos(dotProd);
  }
  
  if( theta < SLERP_TOL ) {
    result = q1;
    return;
  }
  
  double sinTheta = sin(theta);
  
  double coeff1 = sin((1.0 - u) * theta) / sinTheta;
  double coeff2 = sin(u * theta) / sinTheta;
  
  if( dotProd < 0 ) {
    result.w( -coeff1 * q1.w() + coeff2 * q2.w());
    result.x( -coeff1 * q1.x() + coeff2 * q2.x());
    result.y( -coeff1 * q1.y() + coeff2 * q2.y());
    result.z( -coeff1 * q1.z() + coeff2 * q2.z());
  } 
  else {
    result.w( coeff1 * q1.w() + coeff2 * q2.w());
    result.x( coeff1 * q1.x() + coeff2 * q2.x());
    result.y( coeff1 * q1.y() + coeff2 * q2.y());
    result.z( coeff1 * q1.z() + coeff2 * q2.z());
  }
}

//*****************************************************************************
//
// * Perform spherical linear interpolation without checking if an 
//   inversion of the second quaternion would reduce spinning.
//============================================================================
SJCQuaterniond SJCQuaterniond::
SlerpNoInvert(const SJCQuaterniond& quaternion, double factor) const
//============================================================================
{
  double theta = SJCACos(x_ * quaternion.x_ + y_ * quaternion.y_ + 
			 z_ * quaternion.z_ + w_ * quaternion.w_);

  return (*this * sin(theta * (1 - factor)) + 
	  quaternion * sin(theta * factor)) / sinf(theta);
}

//*****************************************************************************
//
// * Perform cubic spherical interpolation along the curve through 
//   the given quaternions by the given factor.
//============================================================================
SJCQuaterniond  SJCQuaterniond::
Squad(const SJCQuaterniond& quaternion2, const SJCQuaterniond& quaternion3, 
      const SJCQuaterniond& quaternion4, double factor) const
//============================================================================
{
  SJCQuaterniond result, temp;

  result = SlerpNoInvert(quaternion2, factor);
  temp   = quaternion3.SlerpNoInvert(quaternion4, factor);
  result = result.SlerpNoInvert(temp, factor * (1.f - factor) * 2);
  return result;
}// Squad


#if 0
//*****************************************************************************
//
//  * Calculate the control point for a spline that fits this quaternion 
//    and the previous and next quaternions given.
//============================================================================
SJCQuaterniond SplineControlPoint(const SJCQuaterniond& prevQuaternion, 
				 const SJCQuaterniond& nextQuaternion) const
//============================================================================
{

   SJCQuaterniond tempQuaternion = -*this;
  return *this * (((tempQuaternion * prevQuaternion).toLogMap() + 
  (tempQuaternion * nextQuaternion).toLogMat()) * -.25f).Exp();
}// SplineControlPoint
#endif

//*****************************************************************************
//
// * Pick closest point operation
//============================================================================
void SJCQuaterniond::
pickClosest(SJCQuaterniond& quat ) 
//============================================================================
{
  double dotProd = 
    w_ * quat.w() + x_ * quat.x() + y_ * quat.y() + z_ * quat.z();
  if( dotProd < 0 ) {
    w_ *= -1;
    x_ *= -1;
    y_ *= -1;
    z_ *= -1;
  }
}


//****************************************************************************
//
// Mutiply it with another quaternion, not following lucas
//============================================================================
void SJCQuaterniond::
switchSide(SJCQuaterniond& a, SJCQuaterniond& b, SJCQuaterniond &r ) 
//============================================================================
{
  r.set(b);
  r.inverse();
  r *= a;
  r *= b;
}

//*****************************************************************************
//
// * Transform into openGL matrix
//============================================================================
void SJCQuaterniond::
glMatrix(double rotMat[16]) 
//============================================================================
{
  for(uint i = 0; i < 16; i++){
    rotMat[i] = 0.f;
  }

  rotMat[0] = 1.0 - 2.0 * y() * y() - 2.0 * z() * z();
  rotMat[1] = 2 * x() * y() + 2 * w() * z();
  rotMat[2] = 2 * x() * z() - 2 * w() * y();
  rotMat[4] = 2 * x() * y() - 2 * w() * z();
  rotMat[5] = 1.0 - 2.0 * x() * x() - 2.0 * z() * z();
  rotMat[6] = 2 * y() * z() + 2 * w() * x();
  rotMat[8] = 2 * x() * z() + 2 * w() * y();
  rotMat[9] = 2 * y() * z() - 2 * w() * x();
  rotMat[10] = 1.0 - 2.0 * x() * x() - 2.0 * y() * y();
  rotMat[15] = 1; 
}

//****************************************************************************
//
// * Constructor from a roatio axis and a angle in radian
//=============================================================================
SJCQuaternionf::
SJCQuaternionf(const float angle, const SJCVector3f &axis )
//=============================================================================
{
  float  s_rad = sin(angle * 0.5);
  x_ = axis.x() * s_rad;
  y_ = axis.y() * s_rad;
  z_ = axis.z() * s_rad;
  w_ = cos(angle * 0.5);
}


//*****************************************************************************
//
// * Set the quaternion to the roation about the axis
//============================================================================
void SJCQuaternionf::
set(float angle, const SJCVector3f &axis) 
//============================================================================
{
  float  s_rad = sin(angle * 0.5);
  x_ = axis.x() * s_rad;
  y_ = axis.y() * s_rad;
  z_ = axis.z() * s_rad;
  w_ = cos(angle * 0.5);
}

//*****************************************************************************
//
// * Set the coordinates to make this Quaternion a rotation that takes the 
//   first std::vector into the second.
//============================================================================
void SJCQuaternionf::
set(const SJCVector3f& s, const SJCVector3f& d) 
//============================================================================
{

  SJCVector3f source = s.normal();
  SJCVector3f destination = d.normal();

  // Compute the dot product and normalized cross product
  float dotProd = source * destination;
  if( dotProd < -1 ) {
    dotProd = -1;
  } 
  else if( dotProd > 1 ) {
    dotProd = 1;
  }

  // Calculate the cross product which should be the rotation axis
  SJCVector3f crossProd = source % destination;
  float norm = crossProd.length();

  if( norm < SJC_EPSILON2 ) { // When two axis are paralley
    //**********************************************************************
    // The std::vectors point nearly along the same line.  
    // 1. If the angle between them is small (closer to 0 than PI),
    //    then set this to the identity rotation.
    // 2. Otherwise assume
    //    the std::vectors are 180 degrees apart and pick an arbitrary axis 
    //    perpendicular to the
    //    line that passes through them.
    //***********************************************************************

    if( dotProd > 0 ) { // Point in the same direction
      x_ = y_ = z_ =0.0;
      w_ = 1.0;
      return;
    }  // end of if
    else {
      // Set up the arbitrary rotation axis to the biggest component of x, y, z
      SJCVector3f secondaryAxis(0, 0, 0);

      if( (SJCAbs(source[0]) <= SJCAbs(source[1]))  && 
	  (SJCAbs(source[0]) <= SJCAbs(source[2])) ) {
	secondaryAxis[0] = 1;
      } 
      else if( (SJCAbs(source[1]) <= SJCAbs(source[0])) && 
	       (SJCAbs(source[1]) <= SJCAbs(source[2])) ) {
	secondaryAxis[1] = 1;
      } 
      else {
	secondaryAxis[2] = 1;
      } 

      // Calcualte the rotation axis 
      crossProd = source % secondaryAxis;
      norm = crossProd.length();			
      
      // That should have worked
      if( norm < SJC_EPSILON2 ) {
	std::cerr << "Warning: Quaternion::setCoords( float *vec1, float *vec2 )"
             << " is indeterminate" << std::endl;

	x_ = y_ = z_ =0.0;
	w_ = 1.0;
	return;
      }// end of if norm < SJC_EPSILON2 (protection against the cross
    }// end of else 
  }// end of if norm < SJC_EPSILON2
  
  crossProd.normalize();
  
  // Fill in the values for this quaternion.
  float axisLength = sqrt( (1.0 - dotProd) / 2.0 );
  
  w_ = sqrt( (1.0 + dotProd) / 2.0);
  x_ = crossProd[0] * axisLength;
  y_ = crossProd[1] * axisLength;
  z_ = crossProd[2] * axisLength;
  
  // Just to be sure
  normalize();
}


//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaternionf::
set( const SJCVector3f& s, const SJCVector3f& d, const SJCVector3f& r ) 
//============================================================================
{

  // Project each std::vector onto the plane perpendicular to the rotation axis
  SJCVector3f source = s.normal();
  SJCVector3f destination = d.normal();
  SJCVector3f rotAxis = r.normal();
  
  float dotProd1 = source * rotAxis;

  float dotProd2 = destination * rotAxis;

  SJCVector3f v1 = source - dotProd1 * rotAxis;
  SJCVector3f v2 = destination - dotProd2 * rotAxis;


  v1.normalize();
  v2.normalize();
  
  float angle = SJCACos(v1 * v2);
  
  SJCVector3f crossProd =  v1 % v2;
  if( (crossProd * rotAxis) < 0 ) {
    angle = -angle;
  }

  set( angle, rotAxis);
}



//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaternionf::
set(const SJCRotateMatrixf &m) 
//============================================================================
{
  float  tr, s, q[4];
  int     i, j, k;
  
  int     nxt[3] = {1, 2, 0};
  
  tr = m[0][0] + m[1][1] + m[2][2];

  // check the diagonal
  if ( tr > 0.0 )  {
//     s = sqrt (tr + 1.0);
//     w_ = s / 2.0;
//     s = 0.5 / s;
//     x_ = (m[1][2] - m[2][1]) * s;
//     y_ = (m[2][0] - m[0][2]) * s;
//     z_ = (m[0][1] - m[1][0]) * s;

    s = sqrt (tr + 1.0);
    w_ = s / 2.0;
    s = 0.5 / s;
    x_ = (m[2][1] - m[1][2]) * s;
    y_ = (m[0][2] - m[2][0]) * s;
    z_ = (m[1][0] - m[0][1]) * s;
  }
  else   {                
    // diagonal is negative
    i = 0;
    if ( m[1][1] > m[0][0] )
      i = 1;
    if ( m[2][2] > m[i][i] )
      i = 2;
    j = nxt[i];
    k = nxt[j];
    
    s = sqrt((m[i][i] - (m[j][j] + m[k][k])) + 1.0);
    
    q[i] = s * 0.5;
    
    if ( s > DBL_EPSILON )
      s = 0.5 / s;
    
    //    q[3] = (m[j][k] - m[k][j] ) * s;
    q[3] = (m[k][j] - m[j][k]) * s;
    q[j] = (m[i][j] + m[j][i]) * s;
    q[k] = (m[i][k] + m[k][i]) * s;
    
    x_ = q[0];
    y_ = q[1];
    z_ = q[2];
    w_ = q[3];

  }
  normalize();
}

//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaternionf::
SetCoord(const SJCVector3f x_dir, const SJCVector3f up) 
//============================================================================
{
  SJCVector3f X, Y, Z, UP;

  X  = x_dir.normal();
  UP = up.normal();   


  float cos_x_up = up * x_dir;
  if(WithinEpsilon(cos_x_up, 1.f)) {
    float angle = -90.f * SJC_DEG_TO_RAD;
    set(angle, SJCConstants::SJC_vYAxis3f);
    return;
  }


  Y = UP % X;   Y.normalize();
  Z = X  % Y;   Z.normalize();

 
  float   m[3][3];

  m[0][0] = X.x(); m[0][1] = X.y(); m[0][2] = X.z();
  m[1][0] = Y.x(); m[1][1] = Y.y(); m[1][2] = Y.z();
  m[2][0] = Z.x(); m[2][1] = Z.y(); m[2][2] = Z.z();
 
  float  tr, s, q[4];
  int    i, j, k;
  
  int    nxt[3] = {1, 2, 0};
  
  tr = m[0][0] + m[1][1] + m[2][2];
  
  // check the diagonal
  if ( tr > 0.0 )  {
    s = sqrt (tr + 1.0);
    w_ = s / 2.0;
    s = 0.5 / s;
    x_ = (m[1][2] - m[2][1]) * s;
    y_ = (m[2][0] - m[0][2]) * s;
    z_ = (m[0][1] - m[1][0]) * s;
  }
  else   {                
    // diagonal is negative
    i = 0;
    if ( m[1][1] > m[0][0] )
      i = 1;
    if ( m[2][2] > m[i][i] )
      i = 2;
    j = nxt[i];
    k = nxt[j];
    
    s = sqrt((m[i][i] - (m[j][j] + m[k][k])) + 1.0);
    
    q[i] = s * 0.5;
    
    if ( s > DBL_EPSILON )
      s = 0.5 / s;
    
    q[3] = (m[j][k] - m[k][j]) * s;
    q[j] = (m[i][j] + m[j][i]) * s;
    q[k] = (m[i][k] + m[k][i]) * s;
    
    x_ = q[0];
    y_ = q[1];
    z_ = q[2];
    w_ = q[3];
  }
  normalize();
}

//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaternionf::
set(const float Z, const float X, const float Y) 
//============================================================================
{
  w_ = 1.0;
  x_ = y_ = z_ = 0.0;

  multiply( 'z', Z);
  multiply( 'x', X);
  multiply( 'y', Y);

  normalize();
}

//*****************************************************************************
//
// *  Set the quaternion related to two std::vector and one rotation axis
//============================================================================
void SJCQuaternionf::
set(const float angle1, const float angle2, const float angle3, 
    const char* order)
//============================================================================
{
  w_ = 1.0;
  x_ = y_ = z_ = 0.0;

  switch(order[0]){
    case 'x':
    case 'X':
      multiply( 'x', angle1);
      break;

    case 'y':
    case 'Y':
      multiply( 'y', angle1);
      break;

    case 'z':
    case 'Z':
      multiply( 'z', angle1);
      break;
    default:
      throw new SJCException("Quaternion: no such order");
  }

  switch(order[1]){
    case 'x':
    case 'X':
      multiply( 'x', angle2);
      break;

    case 'y':
    case 'Y':
      multiply( 'y', angle2);
      break;

    case 'z':
    case 'Z':
      multiply( 'z', angle2);
      break;
    default:
      throw new SJCException("Quaternion: no such order");
  }
  switch(order[2]){
    case 'x':
    case 'X':
      multiply( 'x', angle3);
      break;

    case 'y':
    case 'Y':
      multiply( 'y', angle3);
      break;

    case 'z':
    case 'Z':
      multiply( 'z', angle3);
      break;
    default:
      throw new SJCException("Quaternion: no such order");
  }
  normalize();
}

//*****************************************************************************
//
// * Normalize the quaternion
//============================================================================
void SJCQuaternionf::
normalize(void) 
//============================================================================
{
  float mag = sqrt(x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_);
  if ( mag == 0.0 )
    throw new SJCException("SJCQuaternionf::normalize: Zero length std::vector");
  w_ /= mag;
  x_ /= mag;
  y_ /= mag;
  z_ /= mag;
}

//*****************************************************************************
//
// * Normalize the quaternion
//============================================================================
SJCQuaternionf SJCQuaternionf::
Normal (void) const
//=============================================================================
{
  float mag = sqrt(x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_);
  if ( mag == 0.0 )
    throw new SJCException("SJCQuaternionf::normalize: Zero length std::vector");
  float factor = 1.0f / mag;
  return SJCQuaternionf(x_ * factor, y_ * factor, z_ * factor, w_ * mag);
}
//*****************************************************************************
//
// * The distance between two quaternion
//============================================================================
float SJCQuaternionf::
distance(SJCQuaternionf& quat ) 
//============================================================================
{
  SJCQuaternionf quatCopy( quat );
  
  float dotProd = 
    w_ * quat.w() + x_ * quat.x() + y_ * quat.y() + z_ * quat.z();
   
  // If the antipode is closer, use it instead.
  if( dotProd < 0 ) {
    quatCopy.set(quatCopy.w() * -1, quatCopy.x() * -1, 
		 quatCopy.y() * -1, quatCopy.z() * -1);
  }
  
  // Represent this Quaternion as an offset from the given Quaternion.
  quatCopy.inverse();
  quatCopy *= *this;

  float angle = SJCACos(quatCopy.w()) * 2;
  
  if( angle > M_PI ) {
    angle -= M_2PI;
  }
  
  return( fabs(angle) );
}

//*****************************************************************************
//
// * Transform to the constant axis
//============================================================================
void SJCQuaternionf::
toAngleAxis(float &angle, SJCVector3f &axis) const
//============================================================================
{
  float  half_angle;
  float  sin_half_angle;
  
  if ( w_ > 1.0 )
    half_angle = 0.0;
  else
    half_angle = acos(w_);
  sin_half_angle = sin(half_angle);
  angle = half_angle * 2;
  if ( sin_half_angle > -DBL_EPSILON && sin_half_angle < DBL_EPSILON )  {
    axis.x(0.0);
    axis.y(0.0);
    axis.z(1.0);
  }
  else   {
    sin_half_angle = 1.0 / sin_half_angle;
    axis.x(x_ * sin_half_angle);
    axis.y(y_ * sin_half_angle);
    axis.z(z_ * sin_half_angle);
  }
}

//*****************************************************************************
//
// * Extract the log map information
//   Log of the quaternion.  Define as log(q) = v * a, where 
//   q = [cos(a),v * sin(a)].
//
//============================================================================
void SJCQuaternionf::
toLogMap( float *result ) 
//============================================================================
{
  float sinAngle = sqrt( 1.0 - w_ * w_ );

  if( sinAngle < SIN_ANGLE_TOL ) {
    memset( result, 0, 3 * sizeof(float) );
  }// end of if
  else {
    float angle = SJCACos( w() ) * 2.0;
   
    // SJCACos returns a number between 0 and PI, 
    // but we would like |angle| <= PI
    if( angle > M_PI ) {
      angle -= M_2PI;
    }
    
    result[0] = x() / sinAngle * angle;
    result[1] = y() / sinAngle * angle;
    result[2] = z() / sinAngle * angle;
  } // end of else
}

//*****************************************************************************
//
// * Extract the Euler information
//============================================================================
void SJCQuaternionf::
toEulers( float *result ) 
//============================================================================
{
  SJCRotateMatrixf matrix = toMatrix();
  matrix.extractEuler(result);
}

//*****************************************************************************
//
// * Exponential of the Quaternion
//============================================================================
SJCQuaternionf SJCQuaternionf::
Exp(void) const
//============================================================================
{
  float angle = SJCACos(w_);

  if (angle > 0.f)   {
    float temp = sinf(angle) / angle;
    return SJCQuaternionf(x_ * temp, y_ * temp, z_ * temp, cos(angle));
  }// if
  else
    return SJCQuaternionf((float)0.f, (float)0.f, (float)0.f, 
			 cos(angle));
}// Exp

//*****************************************************************************
//
// * Transform to the constant axis
//============================================================================
SJCQuaternionf SJCQuaternionf::
Alignment(const SJCVector3f& source, const SJCVector3f& destination)
//=============================================================================
{
  SJCVector3f rotationAxis = source % destination;
  rotationAxis.normalize();
  
  SJCVector3f referenceVector = rotationAxis % source;
  referenceVector.normalize();

  float cosAB = destination * source;
  float angle = acos(cosAB);

  if( referenceVector * destination < 0)
    angle = -angle;

  return SJCQuaternionf(angle, rotationAxis);
}

//*****************************************************************************
//
// * Multiplication
//============================================================================
void SJCQuaternionf::
multiply( char axis, float angle) 
//============================================================================
{
  SJCVector3f rotAxis;
  switch(axis){
    case 'x':
    case 'X':
      rotAxis.set(1, 0, 0);
      break;

    case 'y':
    case 'Y':
      rotAxis.set(0, 1, 0);
      break;

    case 'z':
    case 'Z':
      rotAxis.set(0, 0, 1);
      break;
    default:
      throw new SJCException("Quaternion: no such order");
  }

  SJCQuaternionf quat(angle, rotAxis);  

  mult(quat);

  normalize();
}
//****************************************************************************
//
// Mutiply it with another quaternion, not following lucas
//============================================================================
void SJCQuaternionf::
mult( const SJCQuaternionf& other)
//============================================================================
{
  float	a = (w_ + x_)*(other.w_ + other.x_);
  float	b = (z_ - y_)*(other.y_ - other.z_);
  float	c = (w_ - x_)*(other.y_ + other.z_); 
  float	d = (y_ + z_)*(other.w_ - other.x_);
  float	e = (x_ + z_)*(other.x_ + other.y_);
  float	f = (x_ - z_)*(other.x_ - other.y_);
  float	g = (w_ + y_)*(other.w_ - other.z_);
  float	h = (w_ - y_)*(other.w_ + other.z_);

  w_ = a - (e + f + g + h) * 0.5;
  x_ = c + (e - f + g - h) * 0.5;
  y_ = d + (e - f - g + h) * 0.5;
  z_ = b + (-e - f + g + h) * 0.5;
  
}

//*****************************************************************************
//
// * Slerp operation
//============================================================================
void SJCQuaternionf::
slerp( const SJCQuaternionf& q1, const SJCQuaternionf& q2,  float u, 
       SJCQuaternionf& result ) 
//============================================================================
{
  float dotProd = 
      q1.w() * q2.w() + q1.x() * q2.x() + q1.y() * q2.y() + q1.z() * q2.z();

  float theta;
  
  if( dotProd < 0 ) {
    theta = SJCACos(-dotProd);
  }
  else {
    theta = SJCACos(dotProd);
  }
  
  if( theta < SLERP_TOL ) {
    result = q1;
    return;
  }
  
  float sinTheta = sin(theta);
  
  float coeff1 = sin((1.0 - u) * theta) / sinTheta;
  float coeff2 = sin(u * theta) / sinTheta;
  
  if( dotProd < 0 ) {
    result.w( -coeff1 * q1.w() + coeff2 * q2.w());
    result.x( -coeff1 * q1.x() + coeff2 * q2.x());
    result.y( -coeff1 * q1.y() + coeff2 * q2.y());
    result.z( -coeff1 * q1.z() + coeff2 * q2.z());
  } 
  else {
    result.w( coeff1 * q1.w() + coeff2 * q2.w());
    result.x( coeff1 * q1.x() + coeff2 * q2.x());
    result.y( coeff1 * q1.y() + coeff2 * q2.y());
    result.z( coeff1 * q1.z() + coeff2 * q2.z());
  }
}

//*****************************************************************************
//
// * Perform spherical linear interpolation without checking if an 
//   inversion of the second quaternion would reduce spinning.
//============================================================================
SJCQuaternionf SJCQuaternionf::
SlerpNoInvert(const SJCQuaternionf& quaternion, float factor) const
//============================================================================
{
  float theta = SJCACos(x_ * quaternion.x_ + y_ * quaternion.y_ + 
			 z_ * quaternion.z_ + w_ * quaternion.w_);

  return (*this * sin(theta * (1 - factor)) + 
	  quaternion * sin(theta * factor)) / sinf(theta);
}

//*****************************************************************************
//
// * Perform cubic spherical interpolation along the curve through 
//   the given quaternions by the given factor.
//============================================================================
SJCQuaternionf  SJCQuaternionf::
Squad(const SJCQuaternionf& quaternion2, const SJCQuaternionf& quaternion3, 
      const SJCQuaternionf& quaternion4, float factor) const
//============================================================================
{
  SJCQuaternionf result, temp;

  result = SlerpNoInvert(quaternion2, factor);
  temp   = quaternion3.SlerpNoInvert(quaternion4, factor);
  result = result.SlerpNoInvert(temp, factor * (1.f - factor) * 2);
  return result;
}// Squad


#if 0
//*****************************************************************************
//
//  * Calculate the control point for a spline that fits this quaternion 
//    and the previous and next quaternions given.
//============================================================================
SJCQuaternionf SplineControlPoint(const SJCQuaternionf& prevQuaternion, 
				 const SJCQuaternionf& nextQuaternion) const
//============================================================================
{

   SJCQuaternionf tempQuaternion = -*this;
  return *this * (((tempQuaternion * prevQuaternion).toLogMap() + 
  (tempQuaternion * nextQuaternion).toLogMat()) * -.25f).Exp();
}// SplineControlPoint
#endif

//*****************************************************************************
//
// * Pick closest point operation
//============================================================================
void SJCQuaternionf::
pickClosest(SJCQuaternionf& quat ) 
//============================================================================
{
  float dotProd = 
    w_ * quat.w() + x_ * quat.x() + y_ * quat.y() + z_ * quat.z();
  if( dotProd < 0 ) {
    w_ *= -1;
    x_ *= -1;
    y_ *= -1;
    z_ *= -1;
  }
}


//****************************************************************************
//
// Mutiply it with another quaternion, not following lucas
//============================================================================
void SJCQuaternionf::
switchSide(SJCQuaternionf& a, SJCQuaternionf& b, SJCQuaternionf &r ) 
//============================================================================
{
  r.set(b);
  r.inverse();
  r *= a;
  r *= b;
}

//*****************************************************************************
//
// * Transform into openGL matrix
//============================================================================
void SJCQuaternionf::
glMatrix(float rotMat[16]) 
//============================================================================
{
  for(uint i = 0; i < 16; i++){
    rotMat[i] = 0.f;
  }

  rotMat[0] = 1.0 - 2.0 * y() * y() - 2.0 * z() * z();
  rotMat[1] = 2 * x() * y() + 2 * w() * z();
  rotMat[2] = 2 * x() * z() - 2 * w() * y();
  rotMat[4] = 2 * x() * y() - 2 * w() * z();
  rotMat[5] = 1.0 - 2.0 * x() * x() - 2.0 * z() * z();
  rotMat[6] = 2 * y() * z() + 2 * w() * x();
  rotMat[8] = 2 * x() * z() + 2 * w() * y();
  rotMat[9] = 2 * y() * z() - 2 * w() * x();
  rotMat[10] = 1.0 - 2.0 * x() * x() - 2.0 * y() * y();
  rotMat[15] = 1; 
}

