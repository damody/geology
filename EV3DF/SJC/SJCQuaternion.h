/************************************************************************
     Main File:

     File:        SJCQuaternion.h

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
		 2. x(double), y(double), z(double), w(double0: set x, y, z, w value
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


#ifndef SJCQUATERNION_H_
#define SJCQUATERNION_H_

#include "SJC.h"
#include "SJCConstants.h"

#include <math.h>
#include <iostream>

#include "SJCVector3.h"
#include "SJCRotateMatrix.h"
#include "SJCException.h"


class SJCDLL SJCQuaterniond {
 private:
  double x_, y_, z_, w_;
  
 public:
  SJCQuaterniond() { x_ = y_ = z_ = 0.0; w_ = 1.0; }
  SJCQuaterniond(const double x, const double y, const double z,
		const double w) {
    x_ = x; y_ = y; z_ = z; w_ = w;
  }
  SJCQuaterniond(const double angle, const SJCVector3d &axis);
  SJCQuaterniond(const SJCQuaterniond &other) {
    x_ = other.x_; y_ = other.y_; z_ = other.z_; w_ = other.w_; 
  }
  SJCQuaterniond(const SJCRotateMatrixd &m) { set(m); }

  // ZXY rotation
  SJCQuaterniond(const double Z, const double X, const double Y) {
    set(Z,X,Y); }
  
  // 12 possible three combination
  SJCQuaterniond(const double angle1, const double angle2, 
		const double angle3, const char* order) {
      set(angle1, angle2, angle3, order);
  }
  
  SJCQuaterniond(const SJCVector3d& source, const SJCVector3d& destination) {
    set(source, destination);
  }
  
  SJCQuaterniond(const SJCVector3d& source, const SJCVector3d& destination, 
		const SJCVector3d& rotAxis) {
      set(source, destination, rotAxis);
  }
 
 
  double x(void) const { return x_; }
  double y(void) const { return y_; }
  double z(void) const { return z_; }
  double w(void) const { return w_; }

  void    x(const double x) { x_ = x; }
  void    y(const double y) { y_ = y; }
  void    z(const double z) { z_ = z; }
  void    w(const double w) { w_ = w; }

  SJCQuaterniond&  operator=(const SJCRotateMatrixd &m) { 
    set(m); 
    return *this; 
  }

  SJCQuaterniond operator -() const{ return SJCQuaterniond(-x_, -y_, -z_, w_);}
  SJCQuaterniond operator +(const SJCQuaterniond& quaternion) const{
    return SJCQuaterniond(x_ + quaternion.x_, 
			 y_ + quaternion.y_, 
			 z_ + quaternion.z_, 
			 w_ + quaternion.w_).Normal();
  }
  SJCQuaterniond& operator +=(const SJCQuaterniond& quaternion){
    x_ += quaternion.x_;
    y_ += quaternion.y_;
    z_ += quaternion.z_;
    w_ += quaternion.w_;
    SJCAssert(IsUnitLength(), "Invalid quaternion not of unit length.");
    return *this;      
  }
  
  SJCQuaterniond operator -(const SJCQuaterniond& quaternion) const{
    return SJCQuaterniond(x_ - quaternion.x_, 
			 y_ - quaternion.y_, 
			 z_ - quaternion.z_, 
			   w_ - quaternion.w_).Normal();
  }

  SJCQuaterniond& operator -=(const SJCQuaterniond& quaternion){
    x_ -= quaternion.x_;
    y_ -= quaternion.y_;
    z_ -= quaternion.z_;
      w_ -= quaternion.w_;
      SJCAssert(IsUnitLength(), "Invalid quaternion not of unit length.");
      return *this;      
  }

  SJCQuaterniond operator *(double factor) const{
    return SJCQuaterniond(x_ * factor, y_ * factor, z_ * factor, w_ * factor);
  }
  
  SJCQuaterniond& operator *=(double factor){
    x_ *= factor;
    y_ *= factor;
    z_ *= factor;
    w_ *= factor;
    SJCAssert(IsUnitLength(), "Invalid quaternion not of unit length.");
    return *this;
  }
  
  SJCQuaterniond operator /(double factor) const{
    SJCAssert(fabs(factor) > SJC_EPSILON, "Divisor is zero.");
    
    factor = 1.f / factor;
    return SJCQuaterniond(x_ * factor, y_ * factor, z_ * factor, w_ * factor);
  }
  
  SJCQuaterniond& operator /=(double factor){
    SJCAssert(fabs(factor) > SJC_EPSILON, "Divisor is zero.");
    
    factor = 1.f / factor;
    x_ *= factor;
    y_ *= factor;
    z_ *= factor;
    w_ *= factor;
    
    SJCAssert(IsUnitLength(), "Invalid quaternion not of unit length.");
      
    return *this;
  }

  SJCQuaterniond operator*(const SJCQuaterniond &other) const {
    double	a = (w_ + x_)*(other.w_ + other.x_);
    double	b = (z_ - y_)*(other.y_ - other.z_);
    double	c = (w_ - x_)*(other.y_ + other.z_); 
    double	d = (y_ + z_)*(other.w_ - other.x_);
    double	e = (x_ + z_)*(other.x_ + other.y_);
    double	f = (x_ - z_)*(other.x_ - other.y_);
    double	g = (w_ + y_)*(other.w_ - other.z_);
    double	h = (w_ - y_)*(other.w_ + other.z_);
    
    return SJCQuaterniond(a - (e + f + g + h) * 0.5,
			 c + (e - f + g - h) * 0.5,
			 d + (e - f - g + h) * 0.5,
			 b + (-e - f + g + h) * 0.5);
  }
  
  void operator*=(const SJCQuaterniond &other) {
    double	a = (w_ + x_)*(other.w_ + other.x_);
    double	b = (z_ - y_)*(other.y_ - other.z_);
    double	c = (w_ - x_)*(other.y_ + other.z_); 
    double	d = (y_ + z_)*(other.w_ - other.x_);
    double	e = (x_ + z_)*(other.x_ + other.y_);
    double	f = (x_ - z_)*(other.x_ - other.y_);
    double	g = (w_ + y_)*(other.w_ - other.z_);
    double	h = (w_ - y_)*(other.w_ + other.z_);
    
    x_ = a - (e + f + g + h) * 0.5;
    y_ = c + (e - f + g - h) * 0.5;
    z_ = d + (e - f - g + h) * 0.5;
    w_ = b + (-e - f + g + h) * 0.5;
  }

  SJCVector3d operator*(const SJCVector3d &v) const {
    double mx_ = w_ * v.x() + y_ * v.z() - z_ * v.y();
    double my_ = w_ * v.y() + z_ * v.x() - x_ * v.z();
    double mz_ = w_ * v.z() + x_ * v.y() - y_ * v.x();
    double mw_ = - ( x_ * v.x() + y_ * v.y() + z_ * v.z() );
    
    double	a = (mw_ + mx_)*(w_ - x_);
    double	c = (mw_ - mx_)*(-y_ - z_); 
    double	d = (my_ + mz_)*(w_ + x_);
    double	e = (mx_ + mz_)*(-x_ - y_);
    double	f = (mx_ - mz_)*(y_ - x_);
    double	g = (mw_ + my_)*(w_ + z_);
    double	h = (mw_ - my_)*(w_ - z_);
    
    return SJCVector3d(a - (e + f + g + h) * 0.5,
		     c + (e - f + g - h) * 0.5,
		     d + (e - f - g + h) * 0.5);
  }

  void    toAngleAxis(double &angle, SJCVector3d &axis) const;

  SJCRotateMatrixd	toMatrix(void) {
    double	xx = x_ * x_;
    double	yy = y_ * y_;
    double	zz = z_ * z_;
    
    double	xy = x_ * y_;
    double	xz = x_ * z_;
    double	yz = y_ * z_;
    
    double	wx = w_ * x_;
    double	wy = w_ * y_;
    double	wz = w_ * z_;
    
    return SJCRotateMatrixd(1 - 2*yy - 2*zz,     2*xy - 2*wz,     2*xz + 2*wy,
			        2*xy + 2*wz, 1 - 2*xx - 2*zz,     2*yz - 2*wx, 
			        2*xz - 2*wy,     2*yz + 2*wx, 1 - 2*xx - 2*yy);
  }

  void set(double angle, const SJCVector3d &axis) ;

  void set(const SJCVector3d& source, const SJCVector3d& destination);
  void set(const SJCVector3d& source, const SJCVector3d& destination,
	   const SJCVector3d& rotAxis );

  void set(const SJCRotateMatrixd& m);
  // X is the exact direction of x axis and up is relative
  // Y = up % X
  // Z = X % Y
  void SetCoord(const SJCVector3d x_dir, const SJCVector3d up);
 
  void set(const double Z, const double X, const double Y);

  void set(const double w, const double x, const double y, const double z){
    w_ = w; x_ = x; y_ = y; z_ = z;
  }
  void set(const SJCQuaterniond& b){
    w_ = b.w(); x_ = b.x(); y_ = b.y(); z_ = b.z();
  }
  
  void set(const double angle1, const double angle2, const double angle3, 
	   const char* order);
  
  bool IsUnitLength() const { 
    return  fabs(x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_ - 1.f) <= SJC_EPSILON;
  }
  
  SJCQuaterniond Normal() const;
  
  
  void    identity() { x_ = y_ = z_ = 0.0; w_ = 1.0; }
  
  SJCQuaterniond inverse() const {
    return SJCQuaterniond(-x_, -y_, -z_, w_);
  }
  
  double dotProduct(SJCQuaterniond& d)  {
    return (x_ * d.x_ + y_ * d.y_ + z_ * d.z_ + w_ * d.w_);
    
  }
  

  void normalize();
  

  // Returns the (minimum) great-arc distance to the given quaternion.
  double distance( SJCQuaterniond& quat);

  // * Returns the logrithmic map representation of this quarternion,
  //   assuming the identity rotation is the origin.
  void toLogMap( double *result );
  
  // * Converts this rotation to ZXY euler angles; 
  //   that is, the first, second, and third elements in the given array
  //   will respectively hold the Z, X, and Y rotations
  void toEulers( double *result );

 
  static SJCQuaterniond Alignment(const SJCVector3d& directionA, 
				 const SJCVector3d& directionB);
  
  // The second does the same thing, but q is defined by the given axis and 
  // angle (defined in radians).
  void multiply( char axis, double angle);
  
  void mult( const SJCQuaterniond& otherQuat);
  
  // Performs spherical linear interpolation on q1 and q2 and stores the
  //  result in the final argument.  If u = 0, this returns q1; 
  // if u = 1 this returns q2.  This behaves correctly
    // if q1 == result or q2 == result.
  static void slerp( const SJCQuaterniond& q1, const SJCQuaterniond& q2,
		     double u, SJCQuaterniond& result );
  
  // Determines which of the two equivalent quaternions represented by 
  // this rotation is closest to the given quaternion.
  void pickClosest( SJCQuaterniond& quat);
  
  // Fills in r such that b*r = a*b
  static void switchSide(SJCQuaterniond& a, SJCQuaterniond& b, 
			 SJCQuaterniond &r );
  
  void glMatrix(double rotMat[16]) ;
  
  
    //  Exponential of the quaternion.
  SJCQuaterniond Exp() const;

  //  Linearly interpolate between this quaternion and the given  quaternion 
  //  by the given factor.
  SJCQuaterniond Lerp(const SJCQuaterniond& quaternion, double factor) const {
    return (*this + (quaternion - *this) * factor).Normal();
  }
 
  // Perform spherical linear interpolation without checking if an 
  // inversion of the second quaternion would reduce spinning.
  SJCQuaterniond SlerpNoInvert(const SJCQuaterniond& quaternion, 
			       double factor) const;
 
  // Perform cubic spherical interpolation along the curve through 
  // the given quaternions by the given factor.
  SJCQuaterniond Squad(const SJCQuaterniond& quaternion2, 
		      const SJCQuaterniond& quaternion3, 
		      const SJCQuaterniond& quaternion4, 
		      double factor) const;
   
  /*   //  Calculate the control point for a spline that fits this quaternion 
   //  and the previous and next quaternions given.
   SJCQuaterniond SplineControlPoint(const SJCQuaterniond& prevQuaternion, 
   const SJCQuaterniond& nextQuaternion) const;
  */
  
  friend std::ostream& operator<<(std::ostream& o, const SJCQuaterniond& q) {
    o << "[ " << q.x_ << " " << q.y_ << " " << q.z_ << " " << q.w_ << " ]";
    return o;
  }
};

class SJCDLL SJCQuaternionf {
 private:
  float x_, y_, z_, w_;
  
 public:
  SJCQuaternionf() { x_ = y_ = z_ = 0.0; w_ = 1.0; }
  SJCQuaternionf(const float x, const float y, const float z,
		const float w) {
    x_ = x; y_ = y; z_ = z; w_ = w;
  }
  SJCQuaternionf(const float angle, const SJCVector3f &axis);
  SJCQuaternionf(const SJCQuaternionf &other) {
    x_ = other.x_; y_ = other.y_; z_ = other.z_; w_ = other.w_; 
  }
  SJCQuaternionf(const SJCRotateMatrixf &m) { set(m); }

  // ZXY rotation
  SJCQuaternionf(const float Z, const float X, const float Y) { set(Z,X,Y); }
  
  // 12 possible three combination
  SJCQuaternionf(const float angle1, const float angle2, 
		 const float angle3, const char* order) {
    set(angle1, angle2, angle3, order);
  }
  
  SJCQuaternionf(const SJCVector3f& source, const SJCVector3f& destination) {
    set(source, destination);
  }
  
  SJCQuaternionf(const SJCVector3f& source, const SJCVector3f& destination, 
		const SJCVector3f& rotAxis) {
      set(source, destination, rotAxis);
  }
 
 
  float x(void) const { return x_; }
  float y(void) const { return y_; }
  float z(void) const { return z_; }
  float w(void) const { return w_; }

  void    x(const float x) { x_ = x; }
  void    y(const float y) { y_ = y; }
  void    z(const float z) { z_ = z; }
  void    w(const float w) { w_ = w; }

  SJCQuaternionf&  operator=(const SJCRotateMatrixf &m) { 
    set(m); 
    return *this; 
  }
  
  SJCQuaternionf operator -() const{ return SJCQuaternionf(-x_, -y_, -z_, w_);}
  SJCQuaternionf operator +(const SJCQuaternionf& quaternion) const{
    return SJCQuaternionf(x_ + quaternion.x_, 
			  y_ + quaternion.y_, 
			  z_ + quaternion.z_, 
			  w_ + quaternion.w_).Normal();
  }
  SJCQuaternionf& operator +=(const SJCQuaternionf& quaternion){
    x_ += quaternion.x_;
    y_ += quaternion.y_;
    z_ += quaternion.z_;
    w_ += quaternion.w_;
    SJCAssert(IsUnitLength(), "Invalid quaternion not of unit length.");
    return *this;      
  }
  
  SJCQuaternionf operator -(const SJCQuaternionf& quaternion) const{
    return SJCQuaternionf(x_ - quaternion.x_, 
			  y_ - quaternion.y_, 
			  z_ - quaternion.z_, 
			  w_ - quaternion.w_).Normal();
  }

  SJCQuaternionf& operator -=(const SJCQuaternionf& quaternion){
    x_ -= quaternion.x_;
    y_ -= quaternion.y_;
    z_ -= quaternion.z_;
    w_ -= quaternion.w_;
    SJCAssert(IsUnitLength(), "Invalid quaternion not of unit length.");
    return *this;      
  }

  SJCQuaternionf operator *(float factor) const{
    return SJCQuaternionf(x_ * factor, y_ * factor, z_ * factor, w_ * factor);
  }
  
  SJCQuaternionf& operator *=(float factor){
    x_ *= factor;
    y_ *= factor;
    z_ *= factor;
    w_ *= factor;
    SJCAssert(IsUnitLength(), "Invalid quaternion not of unit length.");
    return *this;
  }
  
  SJCQuaternionf operator /(float factor) const{
    SJCAssert(fabs(factor) > SJC_EPSILON, "Divisor is zero.");
    
    factor = 1.f / factor;
    return SJCQuaternionf(x_ * factor, y_ * factor, z_ * factor, w_ * factor);
  }
  
  SJCQuaternionf& operator /=(float factor){
    SJCAssert(fabs(factor) > SJC_EPSILON, "Divisor is zero.");
    
    factor = 1.f / factor;
    x_ *= factor;
    y_ *= factor;
    z_ *= factor;
    w_ *= factor;
    
    SJCAssert(IsUnitLength(), "Invalid quaternion not of unit length.");
      
    return *this;
  }

  SJCQuaternionf operator*(const SJCQuaternionf &other) const {
    float	a = (w_ + x_)*(other.w_ + other.x_);
    float	b = (z_ - y_)*(other.y_ - other.z_);
    float	c = (w_ - x_)*(other.y_ + other.z_); 
    float	d = (y_ + z_)*(other.w_ - other.x_);
    float	e = (x_ + z_)*(other.x_ + other.y_);
    float	f = (x_ - z_)*(other.x_ - other.y_);
    float	g = (w_ + y_)*(other.w_ - other.z_);
    float	h = (w_ - y_)*(other.w_ + other.z_);
    
    return SJCQuaternionf(a - (e + f + g + h) * 0.5,
			 c + (e - f + g - h) * 0.5,
			 d + (e - f - g + h) * 0.5,
			 b + (-e - f + g + h) * 0.5);
  }
  
  void operator*=(const SJCQuaternionf &other) {
    float	a = (w_ + x_)*(other.w_ + other.x_);
    float	b = (z_ - y_)*(other.y_ - other.z_);
    float	c = (w_ - x_)*(other.y_ + other.z_); 
    float	d = (y_ + z_)*(other.w_ - other.x_);
    float	e = (x_ + z_)*(other.x_ + other.y_);
    float	f = (x_ - z_)*(other.x_ - other.y_);
    float	g = (w_ + y_)*(other.w_ - other.z_);
    float	h = (w_ - y_)*(other.w_ + other.z_);
    
    x_ = a - (e + f + g + h) * 0.5;
    y_ = c + (e - f + g - h) * 0.5;
    z_ = d + (e - f - g + h) * 0.5;
    w_ = b + (-e - f + g + h) * 0.5;
  }

  SJCVector3f operator*(const SJCVector3f &v) const {
    float mx_ = w_ * v.x() + y_ * v.z() - z_ * v.y();
    float my_ = w_ * v.y() + z_ * v.x() - x_ * v.z();
    float mz_ = w_ * v.z() + x_ * v.y() - y_ * v.x();
    float mw_ = - ( x_ * v.x() + y_ * v.y() + z_ * v.z() );
    
    float	a = (mw_ + mx_)*(w_ - x_);
    float	c = (mw_ - mx_)*(-y_ - z_); 
    float	d = (my_ + mz_)*(w_ + x_);
    float	e = (mx_ + mz_)*(-x_ - y_);
    float	f = (mx_ - mz_)*(y_ - x_);
    float	g = (mw_ + my_)*(w_ + z_);
    float	h = (mw_ - my_)*(w_ - z_);
    
    return SJCVector3f(a - (e + f + g + h) * 0.5,
		       c + (e - f + g - h) * 0.5,
		       d + (e - f - g + h) * 0.5);
  }

  void    toAngleAxis(float &angle, SJCVector3f &axis) const;

  SJCRotateMatrixf	toMatrix(void) {
    float	xx = x_ * x_;
    float	yy = y_ * y_;
    float	zz = z_ * z_;
    
    float	xy = x_ * y_;
    float	xz = x_ * z_;
    float	yz = y_ * z_;
    
    float	wx = w_ * x_;
    float	wy = w_ * y_;
    float	wz = w_ * z_;
    
    return SJCRotateMatrixf(1 - 2*yy - 2*zz, 2*xy - 2*wz, 2*xz + 2*wy,
			   2*xy + 2*wz, 1 - 2*xx - 2*zz, 2*yz - 2*wx, 
			   2*xz - 2*wy, 2*yz + 2*wx, 1 - 2*xx - 2*yy);
  }

  void set(float angle, const SJCVector3f &axis) ;

  void set(const SJCVector3f& source, const SJCVector3f& destination);
  void set(const SJCVector3f& source, const SJCVector3f& destination,
	   const SJCVector3f& rotAxis );

  void set(const SJCRotateMatrixf& m);
  // X is the exact direction of x axis and up is relative
  // Y = up % X
  // Z = X % Y
  void SetCoord(const SJCVector3f x_dir, const SJCVector3f up);
 
  void set(const float Z, const float X, const float Y);

  void set(const float w, const float x, const float y, const float z){
    w_ = w; x_ = x; y_ = y; z_ = z;
  }
  void set(const SJCQuaternionf& b){
    w_ = b.w(); x_ = b.x(); y_ = b.y(); z_ = b.z();
  }
  
  void set(const float angle1, const float angle2, const float angle3, 
	   const char* order);
  
  bool IsUnitLength() const { 
    return  fabs(x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_ - 1.f) <= SJC_EPSILON;
  }
  
  SJCQuaternionf Normal() const;
  
  
  void    identity() { x_ = y_ = z_ = 0.0; w_ = 1.0; }
  
  SJCQuaternionf inverse() const {
    return SJCQuaternionf(-x_, -y_, -z_, w_);
  }

  float dotProduct(SJCQuaternionf& d)  {
    return (x_ * d.x_ + y_ * d.y_ + z_ * d.z_ + w_ * d.w_);
    
  }
  
  void normalize();
  

  // Returns the (minimum) great-arc distance to the given quaternion.
  float distance( SJCQuaternionf& quat);

  // * Returns the logrithmic map representation of this quarternion,
  //   assuming the identity rotation is the origin.
  void toLogMap( float *result );
  
  // * Converts this rotation to ZXY euler angles; 
  //   that is, the first, second, and third elements in the given array
  //   will respectively hold the Z, X, and Y rotations
  void toEulers( float *result );

  static SJCQuaternionf Alignment(const SJCVector3f& directionA, 
				 const SJCVector3f& directionB);
  
  // The second does the same thing, but q is defined by the given axis and 
  // angle (defined in radians).
  void multiply( char axis, float angle);
  
  void mult( const SJCQuaternionf& otherQuat);
  
  // Performs spherical linear interpolation on q1 and q2 and stores the
  //  result in the final argument.  If u = 0, this returns q1; 
  // if u = 1 this returns q2.  This behaves correctly
    // if q1 == result or q2 == result.
  static void slerp( const SJCQuaternionf& q1, const SJCQuaternionf& q2, 
		     float u, SJCQuaternionf& result );
  
  // Determines which of the two equivalent quaternions represented by 
  // this rotation is closest to the given quaternion.
  void pickClosest( SJCQuaternionf& quat);
  
  // Fills in r such that b*r = a*b
  static void switchSide(SJCQuaternionf& a, SJCQuaternionf& b, 
			 SJCQuaternionf &r );
  
  void glMatrix(float rotMat[16]) ;
  
  
    //  Exponential of the quaternion.
  SJCQuaternionf Exp() const;

  //  Linearly interpolate between this quaternion and the given  quaternion 
  //  by the given factor.
  SJCQuaternionf Lerp(const SJCQuaternionf& quaternion, float factor) const {
    return (*this + (quaternion - *this) * factor).Normal();
  }
 
  // Perform spherical linear interpolation without checking if an 
  // inversion of the second quaternion would reduce spinning.
  SJCQuaternionf SlerpNoInvert(const SJCQuaternionf& quaternion, 
			      float factor) const;
 
  // Perform cubic spherical interpolation along the curve through 
  // the given quaternions by the given factor.
  SJCQuaternionf Squad(const SJCQuaternionf& quaternion2, 
		      const SJCQuaternionf& quaternion3, 
		      const SJCQuaternionf& quaternion4, 
		      float factor) const;
   
  /*   //  Calculate the control point for a spline that fits this quaternion 
   //  and the previous and next quaternions given.
   SJCQuaternionf SplineControlPoint(const SJCQuaternionf& prevQuaternion, 
   const SJCQuaternionf& nextQuaternion) const;
  */
  
  friend std::ostream& operator<<(std::ostream& o, const SJCQuaternionf& q) {
    o << "[ " << q.x_ << " " << q.y_ << " " << q.z_ << " " << q.w_ << " ]";
    return o;
  }
};

#endif

