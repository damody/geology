/************************************************************************
     Main File:

     File:        SJCRotateMatrix.h

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

#ifndef SJCROTATEMATRIX_H_
#define SJCROTATEMATRIX_H_

#include <stdio.h>
#include <string.h>

#include "SJC.h"
#include "SJCVector3.h"

class SJCDLL SJCRotateMatrixd {
 public:

  SJCRotateMatrixd(void) {
    data[0][0] = data[1][1] = data[2][2] = 1.0;
    data[0][1] = data[0][2] = data[1][0] = data[1][2] =
    data[2][0] = data[2][1] = 0.0; }
 
  SJCRotateMatrixd(const double x00, const double x01, const double x02,
		   const double x10, const double x11, const double x12,
		   const double x20, const double x21, const double x22) {
    data[0][0] = x00; data[0][1] = x01; data[0][2] = x02;
    data[1][0] = x10; data[1][1] = x11; data[1][2] = x12;
    data[2][0] = x20; data[2][1] = x21; data[2][2] = x22;
  }

  SJCRotateMatrixd(const double angle, const SJCVector3d& axis);

  SJCRotateMatrixd(const SJCVector3d x, const SJCVector3d y, 
		   const SJCVector3d z)
  {
    data[0][0] = x.x(); data[0][1] = x.y(); data[0][2] = x.z();
    data[1][0] = y.x(); data[1][1] = y.y(); data[1][2] = y.z();
    data[2][0] = z.x(); data[2][1] = z.y(); data[2][2] = z.z();
  }
  SJCRotateMatrixd( const SJCRotateMatrixd &other ) {
    memcpy( data, other.data, 9*sizeof(double) );
  }

  double& operator()(const int i, const int j) { return data[i][j]; }

  SJCVector3d operator[](const int i) const {
    return SJCVector3d(data[i][0], data[i][1], data[i][2]);
  }

  SJCVector3d x() const {
    return SJCVector3d(data[0][0], data[0][1], data[0][2]);
  }
  SJCVector3d y() const {
    return SJCVector3d(data[1][0], data[1][1], data[1][2]);
  }
  SJCVector3d z() const {
    return SJCVector3d(data[2][0], data[2][1], data[2][2]);
  }

  SJCRotateMatrixd &operator=(const SJCRotateMatrixd &other) {
    data[0][0] = other.data[0][0]; data[0][1] = other.data[0][1];
    data[0][2] = other.data[0][2];
    data[1][0] = other.data[1][0]; data[1][1] = other.data[1][1];
    data[1][2] = other.data[1][2];
    data[2][0] = other.data[2][0]; data[2][1] = other.data[2][1];
    data[2][2] = other.data[2][2];
    return *this;
  }
  SJCRotateMatrixd toIdentity(void) {
    data[0][0] = data[1][1] = 1.0;
    data[0][1] = data[0][2] =
    data[1][0] = data[1][2] =
    data[2][0] = data[2][1] = 0.0;
    return *this;
  }
  
  SJCRotateMatrixd operator*(const SJCRotateMatrixd &b) const {
    return SJCRotateMatrixd(  data[0][0] * b.data[0][0]
			   + data[0][1] * b.data[1][0]
			   + data[0][2] * b.data[2][0],
			     data[0][0] * b.data[0][1]
			   + data[0][1] * b.data[1][1]
			   + data[0][2] * b.data[2][1],
			     data[0][0] * b.data[0][2]
			   + data[0][1] * b.data[1][2]
			   + data[0][2] * b.data[2][2],
			     data[1][0] * b.data[0][0]
			   + data[1][1] * b.data[1][0]
			   + data[1][2] * b.data[2][0],
			     data[1][0] * b.data[0][1]
			   + data[1][1] * b.data[1][1]
			   + data[1][2] * b.data[2][1],
			     data[1][0] * b.data[0][2]
			   + data[1][1] * b.data[1][2]
			   + data[1][2] * b.data[2][2],
			     data[2][0] * b.data[0][0]
			   + data[2][1] * b.data[1][0]
			   + data[2][2] * b.data[2][0],
			     data[2][0] * b.data[0][1]
			   + data[2][1] * b.data[1][1]
			   + data[2][2] * b.data[2][1],
			     data[2][0] * b.data[0][2]
			   + data[2][1] * b.data[1][2]
			   + data[2][2] * b.data[2][2]);
    }

  SJCVector3d operator*(const SJCVector3d &v) const {
    return SJCVector3d(  data[0][0] * v.x() + data[0][1] * v.y()
		     + data[0][2] * v.z(),
		       data[1][0] * v.x() + data[1][1] * v.y()
		     + data[1][2] * v.z(),
		       data[2][0] * v.x() + data[2][1] * v.y()
		     + data[2][2] * v.z());
  }

  SJCRotateMatrixd inverse(void) const {
    return SJCRotateMatrixd(data[0][0], data[1][0], data[2][0],
			   data[0][1], data[1][1], data[2][1], 
			   data[0][2], data[1][2], data[2][2]);
  }

  void glForm(double vals[16]) const {
    vals[0]=data[0][0];vals[1]=data[1][0];vals[2]=data[2][0];vals[3]=0.0;
    vals[4]=data[0][1];vals[5]=data[1][1];vals[6]=data[2][1];vals[7]=0.0;
    vals[8]=data[0][2];vals[9]=data[1][2];vals[10]=data[2][2];vals[11]=0.0;
    vals[12] = vals[13] = vals[14] = vals[15] = 1.0;
  }

  void toAngleAxis(double &angle, SJCVector3d &axis) const;
    
  static SJCRotateMatrixd rotateMatrix(const double angle,  
				      const SJCVector3d& axis);


  double* extractEuler() ;

  void   extractEuler(double* result);

  void   calcEuler( double &xAngle, double &yAngle, double &zAngle ) ;

  friend std::ostream& operator<<(std::ostream& o, const SJCRotateMatrixd& m);
  
 private:
  double data[3][3];  // [row][column]
  
  static SJCRotateMatrixd	outerProduct(const SJCVector3d &v1,
					     const SJCVector3d &v2);

};

class SJCDLL SJCRotateMatrixf {
 public:

  SJCRotateMatrixf(void) {
    data[0][0] = data[1][1] = data[2][2] = 1.0;
    data[0][1] = data[0][2] = data[1][0] = data[1][2] =
    data[2][0] = data[2][1] = 0.0; }
 
  SJCRotateMatrixf(const float x00, const float x01, const float x02,
	    const float x10, const float x11, const float x12,
	    const float x20, const float x21, const float x22) {
    data[0][0] = x00; data[0][1] = x01; data[0][2] = x02;
    data[1][0] = x10; data[1][1] = x11; data[1][2] = x12;
    data[2][0] = x20; data[2][1] = x21; data[2][2] = x22;
  }

  SJCRotateMatrixf(const float angle, const SJCVector3f& axis);

  SJCRotateMatrixf(const SJCVector3f x, const SJCVector3f y, 
		   const SJCVector3f z) {
    data[0][0] = x.x(); data[0][1] = x.y(); data[0][2] = x.z();
    data[1][0] = y.x(); data[1][1] = y.y(); data[1][2] = y.z();
    data[2][0] = z.x(); data[2][1] = z.y(); data[2][2] = z.z();
  }
  SJCRotateMatrixf( const SJCRotateMatrixf &other ) {
    memcpy( data, other.data, 9*sizeof(float) );
  }

  float& operator()(const int i, const int j) { return data[i][j]; }

  SJCVector3f operator[](const int i) const {
    return SJCVector3f(data[i][0], data[i][1], data[i][2]);
  }

  SJCVector3f x() const {
    return SJCVector3f(data[0][0], data[0][1], data[0][2]);
  }
  SJCVector3f y() const {
    return SJCVector3f(data[1][0], data[1][1], data[1][2]);
  }
  SJCVector3f z() const {
    return SJCVector3f(data[2][0], data[2][1], data[2][2]);
  }

  SJCRotateMatrixf &operator=(const SJCRotateMatrixf &other) {
    data[0][0] = other.data[0][0]; data[0][1] = other.data[0][1];
    data[0][2] = other.data[0][2];
    data[1][0] = other.data[1][0]; data[1][1] = other.data[1][1];
    data[1][2] = other.data[1][2];
    data[2][0] = other.data[2][0]; data[2][1] = other.data[2][1];
    data[2][2] = other.data[2][2];
    return *this;
  }
  SJCRotateMatrixf toIdentity(void) {
    data[0][0] = data[1][1] = 1.0;
    data[0][1] = data[0][2] =
    data[1][0] = data[1][2] =
    data[2][0] = data[2][1] = 0.0;
    return *this;
  }
  
  SJCRotateMatrixf operator*(const SJCRotateMatrixf &b) const {
    return SJCRotateMatrixf(  data[0][0] * b.data[0][0]
			   + data[0][1] * b.data[1][0]
			   + data[0][2] * b.data[2][0],
			     data[0][0] * b.data[0][1]
			   + data[0][1] * b.data[1][1]
			   + data[0][2] * b.data[2][1],
			     data[0][0] * b.data[0][2]
			   + data[0][1] * b.data[1][2]
			   + data[0][2] * b.data[2][2],
			     data[1][0] * b.data[0][0]
			   + data[1][1] * b.data[1][0]
			   + data[1][2] * b.data[2][0],
			     data[1][0] * b.data[0][1]
			   + data[1][1] * b.data[1][1]
			   + data[1][2] * b.data[2][1],
			     data[1][0] * b.data[0][2]
			   + data[1][1] * b.data[1][2]
			   + data[1][2] * b.data[2][2],
			     data[2][0] * b.data[0][0]
			   + data[2][1] * b.data[1][0]
			   + data[2][2] * b.data[2][0],
			     data[2][0] * b.data[0][1]
			   + data[2][1] * b.data[1][1]
			   + data[2][2] * b.data[2][1],
			     data[2][0] * b.data[0][2]
			   + data[2][1] * b.data[1][2]
			   + data[2][2] * b.data[2][2]);
    }

  SJCVector3f operator*(const SJCVector3f &v) const {
    return SJCVector3f(  data[0][0] * v.x() + data[0][1] * v.y()
		     + data[0][2] * v.z(),
		       data[1][0] * v.x() + data[1][1] * v.y()
		     + data[1][2] * v.z(),
		       data[2][0] * v.x() + data[2][1] * v.y()
		     + data[2][2] * v.z());
  }

  SJCRotateMatrixf inverse(void) const {
    return SJCRotateMatrixf(data[0][0], data[1][0], data[2][0],
			   data[0][1], data[1][1], data[2][1], 
			   data[0][2], data[1][2], data[2][2]);
  }

  void glForm(float vals[16]) const {
    vals[0]=data[0][0];vals[1]=data[1][0];vals[2]=data[2][0];vals[3]=0.0;
    vals[4]=data[0][1];vals[5]=data[1][1];vals[6]=data[2][1];vals[7]=0.0;
    vals[8]=data[0][2];vals[9]=data[1][2];vals[10]=data[2][2];vals[11]=0.0;
    vals[12] = vals[13] = vals[14] = vals[15] = 1.0;
  }
  
  void toAngleAxis(float &angle, SJCVector3f &axis) const;
    
  static SJCRotateMatrixf rotateMatrix(const float angle,  
				      const SJCVector3f& axis);


  float* extractEuler() ;

  void   extractEuler(float* result);

  void   calcEuler( float &xAngle, float &yAngle, float &zAngle ) ;

  friend std::ostream& operator<<(std::ostream& o, const SJCRotateMatrixf& m);
  
 private:
  float data[3][3];  // [row][column]
  
  static SJCRotateMatrixf	outerProduct(const SJCVector3f &v1,
					     const SJCVector3f &v2);

};

#endif

