/************************************************************************
     Main File:

     File:        SimVector4.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     The homogeneous std::vector
     
     Contructor:
		 0 paras: 0, 0, 0, 1
		 4 Real: x, y, z, w
		 array: 3 element array
		 vecter4

     Function:
		 x(), y(), z(), w(): return x, y, z, w
		 x(Real), y(Real), z(Real), w(Real): set x, y, z, w
		 [i] = return ith element

		 = (Vector4): assign the Vector4 to this
		 + (vector)/(array): std::vector addition
		 * (Vector4): dot product
		 * (scalar): std::vector * scalar
		 (scalar) * Vector4:
		 convertToVector(): return the std::vector 

    
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#ifndef SIMVECTOR4_H_
#define SIMVECTOR4_H_

#include "SJC.h"
#include "SJCVector3.h"
#include "SJCException.h"

class SJCDLL SJCVector4d {
 private:
  double d[4];
  
 public:
  SJCVector4d() { d[0] = d[1] = d[2] = d[3] = 0; }
  SJCVector4d(double x, double y, double z, double w) {
    d[0] = x; d[1] = y; d[2] = z; d[3] = w; }
  SJCVector4d(const SJCVector3d v) {
    d[0] = v.x(); d[1] = v.y(); d[2] = v.z(); d[3] = 1.0; }
  
  SJCVector3d convertToVector() const {
    if ( d[3] == 0 )
      throw new SJCException("SJCVector4d::toVector: w_ is zero");
    return SJCVector3d(d[0] / d[3], d[1] / d[3], d[2] / d[3]);
  }

  double x(void) const { return d[0]; }
  double y(void) const { return d[1]; }
  double z(void) const { return d[2]; }
  double w(void) const { return d[3]; }

  void x(double v) { d[0] = v; }
  void y(double v) { d[1] = v; }
  void z(double v) { d[2] = v; }
  void w(double v) { d[2] = v; }

  double& operator[](const int i) { return d[i]; }
  
  SJCVector4d& operator=(const SJCVector4d& v) {
    d[0] = v.d[0]; d[1] = v.d[1]; d[2] = v.d[2]; d[3] = v.d[3];
    return *this;
  }

  SJCVector4d operator+(const SJCVector4d &b) const {
    return SJCVector4d( d[0] + b.d[0], d[1] + b.d[1],
		       d[2] + b.d[2], d[3] + b.d[3] );
  }
  
  SJCVector4d operator*(double v) const {
    return SJCVector4d(d[0] * v, d[1] * v, d[2] * v, d[3] * v);
  }

  friend SJCVector4d operator*(double v, const SJCVector4d &b) {
    return b * v;
  }
  double *GetPointer(void){return d;}
  const double*    get(void) const { return d; }

  double operator*(SJCVector3d v) const {
    return d[0] * v.x() + d[1] * v.y() + d[2] * v.z() + d[3];
  }
};


class SJCDLL SJCVector4f {
 private:
  float d[4];
  
 public:
  SJCVector4f() { d[0] = d[1] = d[2] = d[3] = 0; }
  SJCVector4f(float x, float y, float z, float w) {
    d[0] = x; d[1] = y; d[2] = z; d[3] = w; }
  SJCVector4f(const SJCVector3f v) {
    d[0] = v.x(); d[1] = v.y(); d[2] = v.z(); d[3] = 1.0; }
  
  SJCVector3f convertToVector() const {
    if ( d[3] == 0 )
      throw new SJCException("SJCVector4f::toVector: w_ is zero");
    return SJCVector3f(d[0] / d[3], d[1] / d[3], d[2] / d[3]);
  }
  
  float x() const { return d[0]; }
  float y() const { return d[1]; }
  float z() const { return d[2]; }
  float w() const { return d[3]; }
  
  void x(float v) { d[0] = v; }
  void y(float v) { d[1] = v; }
  void z(float v) { d[2] = v; }
  void w(float v) { d[2] = v; }
  
  float& operator[](const int i) { return d[i]; }

  SJCVector4f& operator=(const SJCVector4f& v) {
    d[0] = v.d[0]; d[1] = v.d[1]; d[2] = v.d[2]; d[3] = v.d[3];
    return *this;
  }

  SJCVector4f operator+(const SJCVector4f &b) const {
    return SJCVector4f( d[0] + b.d[0], d[1] + b.d[1],
			d[2] + b.d[2], d[3] + b.d[3] );
  }
  float *GetPointer(void){return d;}
  const float*    get(void) const { return d; }

  SJCVector4f operator*(float v) const {
    return SJCVector4f(d[0] * v, d[1] * v, d[2] * v, d[3] * v);
  }
  friend SJCVector4f operator*(float v, const SJCVector4f &b) {
    return b * v;
  }


  float operator*(SJCVector3f v) const {
    return d[0] * v.x() + d[1] * v.y() + d[2] * v.z() + d[3];
  }
};

#endif

