/************************************************************************
     Main File:

     File:        SJCVector3d.h

     Author:     
                  Steven Chenney, schenney@cs.wisc.edu
     Modifierr:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
  
     Comment:     The normal 3 std::vector
     
     Contructor:
		 0 paras: 0, 0, 0
		 3 double: x, y, z
		 array: 3 element array
		 vecter

     Function:
		 x(), y(), z(): return x, y, z
		 x(Real), y(Real), z(Real): set x, y, z
		 [i] = return ith element
		 
		 = (vector): assign the std::vector to this
		 == (vector): equality
		 + (vector)/(array): std::vector addition
		 += (vector)/(array)
		 - (vector)/(array): std::vector -
		 -= (vector)/(array)
		 * (scalar): std::vector scalar multiplication
		 *= (scalar)
		 / (scalar): std::vector scalar division
		 /= (scalar)
		 * (vector): dot product
		 *= (vector)
		 % (vector): cross product
		 
		 set(x, y, z)
		 normal(): return normal std::vector of this
		 normalize(): let this become unit std::vector
		 length(): return norm 
		 isZero: norm(this) = 0
		 nearZero: norm(this) ~ 0
		 negate:this * -1
		 isBound(vector, std::vector): true is bounded above and below by
					  given std::vectors
		 setMax(vector): set the maximum value of each component
		 setMin(vector): set the minimum value of each component
                 distance(vector): return the distance between this and pt
                 extractCoords(array):return the array
                 parallelComponent(vector): return the component in std::vector 
                                            direction
                 perpendicularComponent(vector): return the component 
                                                 perpendicular to std::vector 
                                                 direction
                 truncateLength (Real): truncate to desire length


   
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#ifndef SJCVECTOR3_H_
#define SJCVECTOR3_H_

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "SJC.h"
#include "SJCException.h"


#include <iostream>


class SJCDLL SJCVector3d {
  protected:
    double d[3];

  public:
    SJCVector3d(void) { d[0] = d[1] = d[2] = 0.0; };
    SJCVector3d(const double c) { d[0] = d[1] = d[2] = c; };
    SJCVector3d(const double x, const double y, const double z) {
      d[0] = x; d[1] = y; d[2] = z; };
    SJCVector3d(const SJCVector3d& v){
      d[0] = v.d[0]; d[1] = v.d[1]; d[2] = v.d[2]; };
    SJCVector3d(const double *v){
      d[0] = v[0]; d[1] = v[1]; d[2] = v[2]; };

    double x(void) const { return d[0]; }
    double y(void) const { return d[1]; }
    double z(void) const { return d[2]; }

    void x(double v) { d[0] = v; }
    void y(double v) { d[1] = v; }
    void z(double v) { d[2] = v; }

    void    set(const double x, const double y, const double z) {
      d[0] = x; d[1] = y; d[2] = z;
    };
    const double*    get(void) const { return d; }

    // Copies the coordinates to the given  std::vector
    void extractCoords( double *v ) const {
      v[0] = d[0]; v[1] = d[1]; v[2] = d[2];
    } 

    double& operator[](const int i) { return d[i]; }	// No error checking!

    SJCVector3d operator+(const SJCVector3d &b) const {
      return SJCVector3d(d[0] + b.d[0], d[1] + b.d[1], d[2] + b.d[2]);
    }
    void operator+=(const SJCVector3d &b) {
      d[0] += b.d[0]; d[1] += b.d[1]; d[2] += b.d[2];
    }
    SJCVector3d operator+(const double b) const {
      return SJCVector3d(d[0] + b, d[1] + b, d[2] + b);
    }
    void operator+=(const double b) {
      d[0] += b; d[1] += b; d[2] += b;
    }

    SJCVector3d operator-(const SJCVector3d &b) const {
      return SJCVector3d(d[0] - b.d[0], d[1] - b.d[1], d[2] - b.d[2]);
    }
    void operator-=(const SJCVector3d &b) {
      d[0] -= b.d[0]; d[1] -= b.d[1]; d[2] -= b.d[2];
    }

    SJCVector3d operator-(const double b) const {
      return SJCVector3d(d[0] - b, d[1] - b, d[2] - b);
    }
    void operator-=(const double b) {
      d[0] -= b; d[1] -= b; d[2] -= b;
    }
    
    SJCVector3d operator-(const double* b) const {
      return SJCVector3d(d[0] - b[0], d[1] - b[1], d[2] - b[2]);
    }
    void operator-=(const double* b) {
      d[0] -= b[0]; d[1] -= b[1]; d[2] -= b[2];
    }

    SJCVector3d operator-() const { return SJCVector3d(-d[0], -d[1], -d[2]); }

    SJCVector3d operator*(const double x) const {
      return SJCVector3d(d[0] * x, d[1] * x, d[2] * x);
    }
    friend SJCVector3d operator*(const double x, const SJCVector3d &v) {
      return v * x;
    }
    void operator*=(double x) {
      d[0] *= x; d[1] *= x; d[2] *= x;
    }

    SJCVector3d operator/(const double x) const {
      return SJCVector3d(d[0] / x, d[1] / x, d[2] / x);
    }
    void operator/=(double x) {
      d[0] /= x; d[1] /= x; d[2] /= x;
    }

    double operator*(const SJCVector3d &b) const {
      return d[0] * b.d[0] + d[1] * b.d[1] + d[2] * b.d[2];
    }

    SJCVector3d operator%(const SJCVector3d &b) const {
      return SJCVector3d(d[1] * b.d[2] - d[2] * b.d[1],
		       d[2] * b.d[0] - d[0] * b.d[2],
		       d[0] * b.d[1] - d[1] * b.d[0]);
    }

    static
    SJCVector3d	lerp(const SJCVector3d &a, const SJCVector3d &b, 
		     const double u)    {
      return SJCVector3d((1-u)*a.d[0]+u*b.d[0],
		         (1-u)*a.d[1]+u*b.d[1],
		         (1-u)*a.d[2]+u*b.d[2]);
    }

    void negate(void) { d[0] = -d[0]; d[1] = -d[1]; d[2] = -d[2]; }
    void zero(void) { d[0] = 0.0; d[1] = 0.0; d[2] = 0.0; }

    SJCVector3d& operator=(const SJCVector3d &v) {
      d[0] = v.d[0]; d[1] = v.d[1]; d[2] = v.d[2]; return *this;
    }

    bool operator==(const SJCVector3d &b) const {
      return d[0] == b.d[0] && d[1] == b.d[1] && d[2] == b.d[2];
    }
    bool operator!=(const SJCVector3d &b) const {
      return d[0] != b.d[0] || d[1] != b.d[1] || d[2] != b.d[2];
    }

    bool nearZero(void) const {
      return d[0] < DBL_EPSILON && d[0] > -DBL_EPSILON &&
	     d[1] < DBL_EPSILON && d[1] > -DBL_EPSILON &&
	     d[2] < DBL_EPSILON && d[2] > -DBL_EPSILON;
    }

    bool isZero(void) const {
      return nearZero() && d[0]*d[0] + d[1]*d[1] +d[2]*d[2] < DBL_EPSILON;
    }

    double length(void) const { 
      return sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]); }
    double lengthSquare(void) const { 
      return (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);}

    void normalize(void) {
      double	l = length();
      if ( l == 0.0 )
	throw new SJCException("Vector::normalize: Zero length std::vector");
      d[0] /= l; d[1] /= l; d[2] /= l;
    }
    SJCVector3d normal(void) const {
	SJCVector3d n(*this); n.normalize(); return n;
    };

    // The distance between this and pt in 3D
    double distance( SJCVector3d& pt) {
      return( sqrt((pt.d[0] - d[0])*(pt.d[0] - d[0]) + 
		   (pt.d[1] - d[1])*(pt.d[1] - d[1]) +
		   (pt.d[2] - d[2])*(pt.d[2] - d[2])));
    }

    // Whether this std::vector is bounding inside the limit
    bool isBound(SJCVector3d& lower, SJCVector3d& upper){
      if( d[0] >= lower[0] && d[1] >= lower[1] && d[2] >= lower[2] &&
	  d[0] <= upper[0] && d[1] <= upper[1] && d[2] <= upper[2])
	return true;
      else
	return false;
    }

    // Return component of std::vector parallel to a unit basis std::vector
    // * (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))
    SJCVector3d parallelComponent (const SJCVector3d& unitBasis) const  {
      const double projection = *this * unitBasis;
      return projection * unitBasis;
    }

    // Return component of std::vector perpendicular to a unit basis std::vector
    // * (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))
    SJCVector3d perpendicularComponent (const SJCVector3d& unitBasis) const  {
      return (*this) - parallelComponent (unitBasis);
    }

    // Clamps the length of a given std::vector to maxLength.  If the std::vector is
    // shorter its value is returned unaltered, if the std::vector is longer
    // the value returned has length of maxLength and is parallel to the
    // original input.
    SJCVector3d truncateLength (const double maxLength) const  {
      const double maxLengthSquared = maxLength * maxLength;
      const double vecLengthSquared = this->lengthSquare ();
      if (vecLengthSquared <= maxLengthSquared)
	return *this;
      else
	return (*this) * (maxLength / sqrt (vecLengthSquared));
    }


    friend std::ostream& operator<<(std::ostream&o, const SJCVector3d&v) {
	o << "[ " << v.d[0] << " " << v.d[1] << " " << v.d[2] << " ]";
	return o;
    }

    int read(std::istream &i) {
	i >> d[0];
	i >> d[1];
	i >> d[2];
	return i.good();
    }

    bool fread(FILE *f) {
	return fscanf(f, "%lg %lg %lg", &(d[0]), &(d[1]), &(d[2])) == 3;
    }

    bool sread(char *s) {
	return sscanf(s, "%lg %lg %lg", &(d[0]), &(d[1]), &(d[2])) == 3;
    }

    bool Is_Bound(SJCVector3d& lower, SJCVector3d& upper){
      if( d[0] >= lower[0] && d[1] >= lower[1] && d[2] >= lower[2] &&
	  d[0] <= upper[0] && d[1] <= upper[1] && d[2] <= upper[2])
	return true;
      else
	return false;
    }
    SJCVector3d ComponentWiseMultiply(SJCVector3d &mul) {
      return SJCVector3d(d[0] * mul.d[0], d[1] * mul.d[1], d[2] * mul.d[2]);
    }
    

};



class SJCDLL SJCVector3f {
  protected:
    float   d[3];

  public:
    SJCVector3f() { d[0] = d[1] = d[2] = 0.0; };
    SJCVector3f(const float x, const float y, const float z) {
	d[0] = x; d[1] = y; d[2] = z; };
    SJCVector3f(const float c) { d[0] = d[1] = d[2] = c; };
    SJCVector3f(const SJCVector3f& v){
	d[0] = v.d[0]; d[1] = v.d[1]; d[2] = v.d[2]; };
    SJCVector3f(const float *v){
      d[0] = v[0]; d[1] = v[1]; d[2] = v[2]; };

    float   x() const { return d[0]; }
    float   y() const { return d[1]; }
    float   z() const { return d[2]; }

    void    x(float v) { d[0] = v; }
    void    y(float v) { d[1] = v; }
    void    z(float v) { d[2] = v; }

    void    set(const float x, const float y, const float z) {
	d[0] = x; d[1] = y; d[2] = z;
    };
    const float*    get(void) const { return d; }

    // Copies the coordinates to the given  std::vector
    void extractCoords( double *v ) const {
      v[0] = d[0]; v[1] = d[1]; v[2] = d[2];
    } 

    float&  operator[](const int i) { return d[i]; }	// No error checking!

    SJCVector3f operator+(const SJCVector3f &b) const {
	return SJCVector3f(d[0] + b.d[0], d[1] + b.d[1], d[2] + b.d[2]);
    }
    void operator+=(const SJCVector3f &b) {
	d[0] += b.d[0]; d[1] += b.d[1]; d[2] += b.d[2];
    }
    SJCVector3f operator+(const float b) const {
	return SJCVector3f(d[0] + b, d[1] + b, d[2] + b);
    }
    void operator+=(const float b) {
	d[0] += b; d[1] += b; d[2] += b;
    }

    SJCVector3f operator-(const SJCVector3f &b) const {
	return SJCVector3f(d[0] - b.d[0], d[1] - b.d[1], d[2] - b.d[2]);
    }
    void operator-=(const SJCVector3f &b) {
	d[0] -= b.d[0]; d[1] -= b.d[1]; d[2] -= b.d[2];
    }
    SJCVector3f operator-(const float b) const {
	return SJCVector3f(d[0] - b, d[1] - b, d[2] - b);
    }
    void operator-=(const float b) {
	d[0] -= b; d[1] -= b; d[2] -= b;
    }
    SJCVector3f operator-() const { return SJCVector3f(-d[0], -d[1], -d[2]); }

    SJCVector3f operator*(const float x) const {
	return SJCVector3f(d[0] * x, d[1] * x, d[2] * x);
    }
    friend SJCVector3f operator*(const float x, const SJCVector3f &v) {
	return v * x;
    }
    void operator*=(float x) {
	d[0] *= x; d[1] *= x; d[2] *= x;
    }

    SJCVector3f operator/(const float x) const {
	return SJCVector3f(d[0] / x, d[1] / x, d[2] / x);
    }
    void operator/=(float x) {
	d[0] /= x; d[1] /= x; d[2] /= x;
    }

    double operator*(const SJCVector3f &b) const {
    	return d[0] * b.d[0] + d[1] * b.d[1] + d[2] * b.d[2];
    }

    SJCVector3f operator%(const SJCVector3f &b) const {
	return SJCVector3f(d[1] * b.d[2] - d[2] * b.d[1],
			 d[2] * b.d[0] - d[0] * b.d[2],
			 d[0] * b.d[1] - d[1] * b.d[0]);
    }

    static
    SJCVector3f	lerp(const SJCVector3f &a, 
		     const SJCVector3f &b, const float u) 
    {
	return SJCVector3f((1-u)*a.d[0]+u*b.d[0],
			   (1-u)*a.d[1]+u*b.d[1],
			   (1-u)*a.d[2]+u*b.d[2]);
    }

    void negate(void) { d[0] = -d[0]; d[1] = -d[1]; d[2] = -d[2]; }
    void zero(void) { d[0] = 0.0; d[1] = 0.0; d[2] = 0.0; }

    SJCVector3f& operator=(const SJCVector3f &v) {
    	d[0] = v.d[0]; d[1] = v.d[1]; d[2] = v.d[2]; return *this;
    }
  
 
    bool operator==(const SJCVector3f &b) const {
	return d[0] == b.d[0] && d[1] == b.d[1] && d[2] == b.d[2];
    }
    bool nearZero(void) const {
	return d[0] < FLT_EPSILON && d[0] > -FLT_EPSILON &&
	       d[1] < FLT_EPSILON && d[1] > -FLT_EPSILON &&
	       d[2] < FLT_EPSILON && d[2] > -FLT_EPSILON;
    }
    bool isZero(void) const {
	return nearZero() && d[0]*d[0] + d[1]*d[1] +d[2]*d[2] < DBL_EPSILON;
    }

    float length() const { return sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]); }
    float lengthSquare(void) const { 
      return (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);}

    void normalize() {
	double	l = length();
	if ( l == 0.0 )
	    throw new SJCException("Vectorf::normalize: Zero length std::vector");
	d[0] /= l; d[1] /= l; d[2] /= l;
    }
    SJCVector3f normal() const {
	SJCVector3f n(*this); n.normalize(); return n;
    };

    // The distance between this and pt in 3D
    double distance( SJCVector3f& pt) {
      return( sqrt((pt.d[0] - d[0])*(pt.d[0] - d[0]) + 
		   (pt.d[1] - d[1])*(pt.d[1] - d[1]) +
		   (pt.d[2] - d[2])*(pt.d[2] - d[2])));
    }

    // Whether this std::vector is bounding inside the limit
    bool isBound(SJCVector3f& lower, SJCVector3f& upper){
      if( d[0] >= lower[0] && d[1] >= lower[1] && d[2] >= lower[2] &&
	  d[0] <= upper[0] && d[1] <= upper[1] && d[2] <= upper[2])
	return true;
      else
	return false;
    }

    // Return component of std::vector parallel to a unit basis std::vector
    // * (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))
    SJCVector3f parallelComponent (const SJCVector3f& unitBasis) const  {
      const double projection = *this * unitBasis;
      return projection * unitBasis;
    }

    // Return component of std::vector perpendicular to a unit basis std::vector
    // * (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))
    SJCVector3f perpendicularComponent (const SJCVector3f& unitBasis) const  {
      return (*this) - parallelComponent (unitBasis);
    }

    // Clamps the length of a given std::vector to maxLength.  If the std::vector is
    // shorter its value is returned unaltered, if the std::vector is longer
    // the value returned has length of maxLength and is parallel to the
    // original input.
    SJCVector3f truncateLength (const float maxLength) const  {
      const float maxLengthSquared = maxLength * maxLength;
      const float vecLengthSquared = this->lengthSquare ();
      if (vecLengthSquared <= maxLengthSquared)
	return *this;
      else
	return (*this) * (maxLength / sqrt (vecLengthSquared));
    }

    friend std::ostream& operator<<(std::ostream&o, const SJCVector3f&v) {
	o << "[ " << v.d[0] << " " << v.d[1] << " " << v.d[2] << " ]";
	return o;
    }

    int read(std::istream &i) {
	i >> d[0];
	i >> d[1];
	i >> d[2];
	return i.good();
    }

    bool fread(FILE *f) {
	return fscanf(f, "%g %g %g", &(d[0]), &(d[1]), &(d[2])) == 3;
    }

    bool sread(char *s) {
	return sscanf(s, "%g %g %g", &(d[0]), &(d[1]), &(d[2])) == 3;
    }
    
    bool Is_Bound(SJCVector3f& lower, SJCVector3f& upper){
      if( d[0] >= lower[0] && d[1] >= lower[1] && d[2] >= lower[2] &&
	  d[0] <= upper[0] && d[1] <= upper[1] && d[2] <= upper[2])
	return true;
      else
	return false;
    }
    SJCVector3f ComponentWiseMultiply(SJCVector3f &mul) {
      return SJCVector3f(d[0] * mul.d[0], d[1] * mul.d[1], d[2] * mul.d[2]);
    }
    
};


#endif

