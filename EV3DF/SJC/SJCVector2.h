
/************************************************************************
     Main File:

     File:        SJCVector2.h

     Author:     
                  Steven Chenney, schenney@cs.wisc.edu
     Modifier:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
 
     Comment:     The normal 2D std::vector
     
     Contructor:
		 0 paras: 0, 0, 0
		 3 Real: x, y, z
		 array: 3 element array
		 vecter

     Function:
		 x(), y(), z(): return x, y, z
		 x(Real), y(Real), z(Real): set x, y, z
		 [i] = return ith element
		 
		 = (vector): assign the std::vector to this
		 == (vector): equality
		 != (vector): equality
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
		 setMax(vector): set the maximum value of each component
		 setMin(vector): set the minimum value of each component
                 distance(vector): return the distance between this and pt
                 extractCoords(array):return the array

   
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#ifndef SJCVECTOR2_H_
#define SJCVECTOR2_H_

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "SJC.h"
#include "SJCException.h"

#include <iostream>


class SJCDLL SJCVector2d {
 protected:
  double d[2];

 public:
  // Constructors
  // The first sets each element to zero; the remaining ones copy the given
  //  coordinate values.
  SJCVector2d(void) { d[0] = d[1] = 0.0; }
  SJCVector2d(const double x, const double y) { d[0] = x; d[1] = y; }
  SJCVector2d(const double *values){ d[0] = values[0]; d[1] = values[1]; };
  SJCVector2d(const SJCVector2d& v){ d[0] = v.d[0]; d[1] = v.d[1]; }

  double x(void) const { return d[0]; }
  double y(void) const { return d[1]; }

  void x(double v) { d[0] = v; }
  void y(double v) { d[1] = v; }
  
  const double* get(void) const { return d; }
  
  void set(const double x, const double y) { d[0] = x; d[1] = y; }
  void set(const double* v) { d[0] = v[0]; d[1] = v[1];  };

  double& operator[](const int i) { return d[i]; }	// No error checking!

  SJCVector2d operator+(const SJCVector2d &b) const {
    return SJCVector2d(d[0] + b.d[0], d[1] + b.d[1]);
  }
  void operator+=(const SJCVector2d &b) {
    d[0] += b.d[0]; d[1] += b.d[1];
  }

  SJCVector2d operator+(const double b) const {
    return SJCVector2d(d[0] + b, d[1] + b);
  }
  void operator+=(const double b) {
    d[0] += b; d[1] += b;
  }

  SJCVector2d operator+(const double* b) const {
    return SJCVector2d(d[0] + b[0], d[1] + b[1]);
  }
  void operator+=(const double* b) {
    d[0] += b[0]; d[1] += b[1];
  }

  SJCVector2d operator-(const SJCVector2d &b) const {
    return SJCVector2d(d[0] - b.d[0], d[1] - b.d[1]);
  }
  void operator-=(const SJCVector2d &b) {
    d[0] -= b.d[0]; d[1] -= b.d[1];
  }

  SJCVector2d operator-(const double b) const {
    return SJCVector2d(d[0] - b, d[1] - b);
  }
  void operator-=(const double b) {
    d[0] -= b; d[1] -= b;
  }

  SJCVector2d operator-(const double* b) const {
    return SJCVector2d(d[0] - b[0], d[1] - b[1]);
  }
  void operator-=(const double* b) {
    d[0] -= b[0]; d[1] -= b[1];
  }

  SJCVector2d operator-() const { return SJCVector2d(-d[0], -d[1]); }

  SJCVector2d operator*(const double x) const {
    return SJCVector2d(d[0] * x, d[1] * x);
  }
  friend SJCVector2d operator*(const double x, const SJCVector2d &v) {
    return v * x;
  }
  void operator*=(double x) {
    d[0] *= x; d[1] *= x;
  }

  SJCVector2d operator/(const double x) const {
    return SJCVector2d(d[0] / x, d[1] / x);
  }
  void operator/=(double x) {
    d[0] /= x; d[1] /= x;
  }
  
  double operator*(const SJCVector2d &b) const {
    return d[0] * b.d[0] + d[1] * b.d[1];
  }

  double operator%(const SJCVector2d &b) const {
    return d[0] * b.d[1] - d[1] * b.d[0];
  }

  static
    SJCVector2d lerp(const SJCVector2d &a, const SJCVector2d &b, const double u) {
    return SJCVector2d((1-u)*a.d[0]+u*b.d[0], (1-u)*a.d[1]+u*b.d[1]);
  }
  
  void negate(void) { d[0] = -d[0]; d[1] = -d[1]; }
  void zero(void) { d[0] = 0.0; d[1] = 0.0; }

  SJCVector2d& operator=(const SJCVector2d &v) {
    d[0] = v.d[0]; d[1] = v.d[1]; return *this;
  }

  bool operator==(const SJCVector2d &b) const {
    return d[0] == b.d[0] && d[1] == b.d[1];
  }

  bool operator!=(const SJCVector2d &b) const {
    return d[0] != b.d[0] || d[1] != b.d[1];
  }

  bool nearZero(void) const {
    return d[0] < DBL_EPSILON && d[0] > -DBL_EPSILON &&
           d[1] < DBL_EPSILON && d[1] > -DBL_EPSILON;
  }
  bool isZero(void) const {
    return nearZero() && d[0]*d[0] + d[1]*d[1] < DBL_EPSILON;
  }

  double length(void) const { return sqrt(d[0]*d[0] + d[1]*d[1]); }

  void normalize(void) {
    double	l = length();
    if ( l == 0.0 )
      throw new SJCException("Vector::normalize: Zero length std::vector");
    d[0] /= l; d[1] /= l;
  }
  SJCVector2d normal() const {
    SJCVector2d n(*this); n.normalize(); return n;
  };

  // The distance between this and pt in 2D
  double distance( SJCVector2d& pt) {
    return( sqrt((pt.d[0] - d[0])*(pt.d[0] - d[0]) + 
		 (pt.d[1] - d[1])*(pt.d[1] - d[1])));
  }

  // Copies the coordinates to the given  std::vector
  void extractCoords( double *vec ){vec[0] = d[0]; vec[1] = d[1];}; 

  friend std::ostream& operator<<(std::ostream&o, const SJCVector2d&v) {
    o << "[ " << v.d[0] << " " << v.d[1] << " ]";
    return o;
  }

  int read(std::istream &i) {
    i >> d[0];
    i >> d[1];
    return i.good();
  }
  
  bool fread(FILE *f) {
    return fscanf(f, "%lg %lg", &(d[0]), &(d[1])) == 2;
  }
  
  bool sread(char *s) {
    return sscanf(s, "%lg %lg", &(d[0]), &(d[1])) == 2;
  }
};


class SJCDLL SJCVector2f {
  protected:
    float d[2];

  public:
    SJCVector2f() { d[0] = d[1] = 0.0; }
    SJCVector2f(const float x, const float y) { d[0] = x; d[1] = y; }
    SJCVector2f(const SJCVector2f& v){ d[0] = v.d[0]; d[1] = v.d[1]; }

    float x() const { return d[0]; }
    float y() const { return d[1]; }

    void x(float v) { d[0] = v; }
    void y(float v) { d[1] = v; }

    const float*    get(void) const { return d; }

    void    set(const float x, const float y) { d[0] = x; d[1] = y; }

    float& operator[](const int i) { return d[i]; }	// No error checking!

    SJCVector2f operator+(const SJCVector2f &b) const {
	return SJCVector2f(d[0] + b.d[0], d[1] + b.d[1]);
    }
    void operator+=(const SJCVector2f &b) {
	d[0] += b.d[0]; d[1] += b.d[1];
    }
    SJCVector2f operator+(const float b) const {
	return SJCVector2f(d[0] + b, d[1] + b);
    }
    void operator+=(const float b) {
	d[0] += b; d[1] += b;
    }

    SJCVector2f operator-(const SJCVector2f &b) const {
	return SJCVector2f(d[0] - b.d[0], d[1] - b.d[1]);
    }
    void operator-=(const SJCVector2f &b) {
	d[0] -= b.d[0]; d[1] -= b.d[1];
    }
    SJCVector2f operator-(const float b) const {
	return SJCVector2f(d[0] - b, d[1] - b);
    }
    void operator-=(const float b) {
	d[0] -= b; d[1] -= b;
    }
    SJCVector2f operator-() const { return SJCVector2f(-d[0], -d[1]); }

    SJCVector2f operator*(const float x) const {
	return SJCVector2f(d[0] * x, d[1] * x);
    }
    friend SJCVector2f operator*(const float x, const SJCVector2f &v) {
	return v * x;
    }
    void operator*=(float x) {
	d[0] *= x; d[1] *= x;
    }

    SJCVector2f operator/(const float x) const {
	return SJCVector2f(d[0] / x, d[1] / x);
    }
    void operator/=(float x) {
	d[0] /= x; d[1] /= x;
    }

    float operator*(const SJCVector2f &b) const {
    	return d[0] * b.d[0] + d[1] * b.d[1];
    }

    float operator%(const SJCVector2f &b) const {
	return d[0] * b.d[1] - d[1] * b.d[0];
    }

    static
    SJCVector2f	lerp(const SJCVector2f &a, const SJCVector2f &b, const float u){
	return SJCVector2f((1-u)*a.d[0]+u*b.d[0], (1-u)*a.d[1]+u*b.d[1]);
    }

    void negate(void) { d[0] = -d[0]; d[1] = -d[1]; }
    void zero(void) { d[0] = 0.0; d[1] = 0.0; }

    SJCVector2f& operator=(const SJCVector2f &v) {
    	d[0] = v.d[0]; d[1] = v.d[1]; return *this;
    }

    bool operator==(const SJCVector2f &b) const {
	return d[0] == b.d[0] && d[1] == b.d[1];
    }
    bool operator!=(const SJCVector2f &b) const {
	return d[0] != b.d[0] || d[1] != b.d[1];
    }
    bool nearZero(void) const {
	return d[0] < FLT_EPSILON && d[0] > -FLT_EPSILON &&
	       d[1] < FLT_EPSILON && d[1] > -FLT_EPSILON;
    }
    bool isZero(void) const {
	return nearZero() && d[0]*d[0] + d[1]*d[1] < FLT_EPSILON;
    }

    double length() const { return sqrt(d[0]*d[0] + d[1]*d[1]); }
    void normalize() {
	double	l = length();
	if ( l == 0.0 )
	    throw new SJCException("Vector::normalize: Zero length std::vector");
	d[0] /= l; d[1] /= l;
    }
    SJCVector2f normal() const {
	SJCVector2f n(*this); n.normalize(); return n;
    };

    friend std::ostream& operator<<(std::ostream&o, const SJCVector2f&v) {
	o << "[ " << v.d[0] << " " << v.d[1] << " ]";
	return o;
    }

    int read(std::istream &i) {
	i >> d[0];
	i >> d[1];
	return i.good();
    }

    void Write(std::ostream &o) const {
	o.write((char*)&(d[0]), sizeof(float));
	o.write((char*)&(d[1]), sizeof(float));
    }

    void Read(std::istream &f) {
	f.read((char*)&(d[0]), sizeof(float));
	f.read((char*)&(d[1]), sizeof(float));
    }

    bool fread(FILE *f) {
	return fscanf(f, "%g %g", &(d[0]), &(d[1])) == 2;
    }

    bool sread(char *s) {
	return sscanf(s, "%g %g", &(d[0]), &(d[1])) == 2;
    }
};



#endif


