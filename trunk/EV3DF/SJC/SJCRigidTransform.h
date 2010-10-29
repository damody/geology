/************************************************************************
     Main File:

     File:        SimTransform.h

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

     Accessory:   LU_Decomposition, LU_Back_Substd
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#ifndef SJCRIGIDTRANSFORM_H_
#define SJCRIGIDTRANSFORM_H_

#include "SJC.h"
#include "SJCConstants.h"

#include <stdio.h>
#include <string.h>
#include "SJCVector3.h"
#include "SJCVector4.h"
#include "SJCQuaternion.h"
#include "SJCRotateMatrix.h"

class SJCDLL SJCRigidTransformd {
 private:
  double data[3][4];  // [row][column]
  
  static SJCRigidTransformd	outerProduct(const SJCVector3d &v1,
					     const SJCVector3d &v2);
  
 public:
  
  SJCRigidTransformd() {
    data[0][0] = data[1][1] = data[2][2] = 1.0;
    data[0][1] = data[0][2] = data[0][3] =
      data[1][0] = data[1][2] = data[1][3] =
      data[2][0] = data[2][1] = data[2][3] = 0.0; }
  SJCRigidTransformd(const double x00, const double x01,
		    const double x02, const double x03,
		    const double x10, const double x11,
		    const double x12, const double x13,
		    const double x20, const double x21,
		    const double x22, const double x23) {
    data[0][0] = x00; data[0][1] = x01; data[0][2] = x02; data[0][3] = x03;
    data[1][0] = x10; data[1][1] = x11; data[1][2] = x12; data[1][3] = x13;
    data[2][0] = x20; data[2][1] = x21; data[2][2] = x22; data[2][3] = x23;
  }
  SJCRigidTransformd( const SJCRigidTransformd &other ) {
    memcpy( data, other.data, 12*sizeof(double) );
  }
  SJCRigidTransformd(const SJCQuaterniond &q, const SJCVector3d &v);

  double& operator()(const int i, const int j) { return data[i][j]; }
  const double& operator()(const int i, const int j) const { 
    return data[i][j]; 
  }
  
  SJCVector4d operator[](const int i) const {
    return SJCVector4d(data[i][0], data[i][1], data[i][2], data[i][3]);
  }

  SJCVector4d x() const {
    return SJCVector4d(data[0][0], data[0][1], data[0][2], data[0][3]);
  }
  SJCVector4d y() const {
    return SJCVector4d(data[1][0], data[1][1], data[1][2], data[1][3]);
  }
  SJCVector4d z() const {
    return SJCVector4d(data[2][0], data[2][1], data[2][2], data[2][3]);
  }
  void x(const SJCVector4d &v) {
    data[0][0] = v.x(); data[0][1] = v.y();
    data[0][2] = v.z(); data[0][3] = v.w();
  }
  void y(const SJCVector4d &v) {
    data[1][0] = v.x(); data[1][1] = v.y();
    data[1][2] = v.z(); data[1][3] = v.w();
  }
  void z(const SJCVector4d &v) {
    data[2][0] = v.x(); data[2][1] = v.y();
    data[2][2] = v.z(); data[2][3] = v.w();
  }
  
  SJCRigidTransformd &operator=(const SJCRigidTransformd &other) {
    memcpy( data, other.data, 12*sizeof(double) );
    return *this;
  }
  SJCRigidTransformd toIdentity(void) {
    data[0][0] = data[1][1] = data[2][2] = 1.0;
    data[0][1] = data[0][2] = data[0][3] =
      data[1][0] = data[1][2] = data[1][3] =
      data[2][0] = data[2][1] = data[2][3] = 0.0;
    return *this;
  }
  
  SJCRigidTransformd operator*(const SJCRigidTransformd &b) const {
    return SJCRigidTransformd(
	    	         data[0][0] * b.data[0][0] + data[0][1] * b.data[1][0]
		       + data[0][2] * b.data[2][0],
			 data[0][0] * b.data[0][1] + data[0][1] * b.data[1][1]
		       + data[0][2] * b.data[2][1],
			 data[0][0] * b.data[0][2] + data[0][1] * b.data[1][2]
		       + data[0][2] * b.data[2][2],
			 data[0][0] * b.data[0][3] + data[0][1] * b.data[1][3]
		       + data[0][2] * b.data[2][3] + data[0][3],
			 data[1][0] * b.data[0][0] + data[1][1] * b.data[1][0]
		       + data[1][2] * b.data[2][0],
			 data[1][0] * b.data[0][1] + data[1][1] * b.data[1][1]
		       + data[1][2] * b.data[2][1],
			 data[1][0] * b.data[0][2] + data[1][1] * b.data[1][2]
		       + data[1][2] * b.data[2][2],
			 data[1][0] * b.data[0][3] + data[1][1] * b.data[1][3]
		       + data[1][2] * b.data[2][3] + data[1][3],
			 data[2][0] * b.data[0][0] + data[2][1] * b.data[1][0]
		       + data[2][2] * b.data[2][0],
			 data[2][0] * b.data[0][1] + data[2][1] * b.data[1][1]
		       + data[2][2] * b.data[2][1],
			 data[2][0] * b.data[0][2] + data[2][1] * b.data[1][2]
		       + data[2][2] * b.data[2][2],
			 data[2][0] * b.data[0][3] + data[2][1] * b.data[1][3]
		       + data[2][2] * b.data[2][3] + data[2][3]);
  }
  
  SJCVector3d operator*(const SJCVector3d &v) const {
    return SJCVector3d(  data[0][0] * v.x() + data[0][1] * v.y()
		     + data[0][2] * v.z() + data[0][3],
		       data[1][0] * v.x() + data[1][1] * v.y()
		     + data[1][2] * v.z() + data[1][3],
		       data[2][0] * v.x() + data[2][1] * v.y()
		     + data[2][2] * v.z() + data[2][3]);
    }
  SJCVector3d operator%(const SJCVector3d &v) const {
    return SJCVector3d(data[0][0]*v.x() + data[0][1]*v.y() + data[0][2]*v.z(),
		     data[1][0]*v.x() + data[1][1]*v.y() + data[1][2]*v.z(),
		     data[2][0]*v.x() + data[2][1]*v.y() + data[2][2]*v.z());
  }
  SJCVector4d operator*(const SJCVector4d &v) const {
    return SJCVector4d(  data[0][0] * v.x() + data[0][1] * v.y()
		      + data[0][2] * v.z() + data[0][3] * v.w(),
		        data[1][0] * v.x() + data[1][1] * v.y()
		      + data[1][2] * v.z() + data[1][3] * v.w(),
		        data[2][0] * v.x() + data[2][1] * v.y()
		      + data[2][2] * v.z() + data[2][3] * v.w(), v.w());
    }

  SJCRigidTransformd inverse() const {
    return SJCRigidTransformd(data[0][0], data[1][0], data[2][0],
			    -data[0][0] * data[0][3] - data[1][0] * data[1][3]
			    -data[2][0] * data[2][3],
			     data[0][1], data[1][1], data[2][1], 
			    -data[0][1] * data[0][3] - data[1][1] * data[1][3]
			    -data[2][1] * data[2][3],
			     data[0][2], data[1][2], data[2][2], 
			    -data[0][2] * data[0][3] - data[1][2] * data[1][3]
			    -data[2][2] * data[2][3]);
  }

  void    toComponents(double &angle, SJCVector3d &axis,
		       SJCVector3d &tran) const;

  void   invert(void) ;

  friend std::ostream& operator<<(std::ostream& o, const SJCRigidTransformd& m) ;
 

  static SJCRigidTransformd	rotateTransform(const double angle,
					        const SJCVector3d& axis);

  static SJCRigidTransformd	translateTransform(const SJCVector3d t) {
    return SJCRigidTransformd(1., 0., 0., t.x(),
			     0., 1., 0., t.y(),
			     0., 0., 1., t.z());
  }
  static SJCRigidTransformd	scaleTransform(const SJCVector3d s) {
    return SJCRigidTransformd(s.x(), 0., 0., 0.,
			      0., s.y(), 0., 0.,
			      0., 0., s.z(), 0.);
  }
  

};

class SJCDLL SJCRigidTransformf {
 private:
  float data[3][4];  // [row][column]
  
  static SJCRigidTransformf	outerProduct(const SJCVector3f &v1,
					     const SJCVector3f &v2);
  
 public:
  
  SJCRigidTransformf() {
    data[0][0] = data[1][1] = data[2][2] = 1.0;
    data[0][1] = data[0][2] = data[0][3] =
      data[1][0] = data[1][2] = data[1][3] =
      data[2][0] = data[2][1] = data[2][3] = 0.0; }
  SJCRigidTransformf(const float x00, const float x01,
		    const float x02, const float x03,
		    const float x10, const float x11,
		    const float x12, const float x13,
		    const float x20, const float x21,
		    const float x22, const float x23) {
    data[0][0] = x00; data[0][1] = x01; data[0][2] = x02; data[0][3] = x03;
    data[1][0] = x10; data[1][1] = x11; data[1][2] = x12; data[1][3] = x13;
    data[2][0] = x20; data[2][1] = x21; data[2][2] = x22; data[2][3] = x23;
  }
  SJCRigidTransformf( const SJCRigidTransformf &other ) {
    memcpy( data, other.data, 12*sizeof(float) );
  }
  SJCRigidTransformf(const SJCQuaternionf &q, const SJCVector3f &v);

  float& operator()(const int i, const int j) { return data[i][j]; }
  const float& operator()(const int i, const int j) const { 
    return data[i][j]; 
  }
  
  SJCVector4d operator[](const int i) const {
    return SJCVector4d(data[i][0], data[i][1], data[i][2], data[i][3]);
  }

  SJCVector4d x() const {
    return SJCVector4d(data[0][0], data[0][1], data[0][2], data[0][3]);
  }
  SJCVector4d y() const {
    return SJCVector4d(data[1][0], data[1][1], data[1][2], data[1][3]);
  }
  SJCVector4d z() const {
    return SJCVector4d(data[2][0], data[2][1], data[2][2], data[2][3]);
  }
  void x(const SJCVector4d &v) {
    data[0][0] = v.x(); data[0][1] = v.y();
    data[0][2] = v.z(); data[0][3] = v.w();
  }
  void y(const SJCVector4d &v) {
    data[1][0] = v.x(); data[1][1] = v.y();
    data[1][2] = v.z(); data[1][3] = v.w();
  }
  void z(const SJCVector4d &v) {
    data[2][0] = v.x(); data[2][1] = v.y();
    data[2][2] = v.z(); data[2][3] = v.w();
  }
  
  SJCRigidTransformf &operator=(const SJCRigidTransformf &other) {
    memcpy( data, other.data, 12*sizeof(float) );
    return *this;
  }
  SJCRigidTransformf toIdentity(void) {
    data[0][0] = data[1][1] = data[2][2] = 1.0;
    data[0][1] = data[0][2] = data[0][3] =
      data[1][0] = data[1][2] = data[1][3] =
      data[2][0] = data[2][1] = data[2][3] = 0.0;
    return *this;
  }
  
  SJCRigidTransformf operator*(const SJCRigidTransformf &b) const {
    return SJCRigidTransformf(
	    	         data[0][0] * b.data[0][0] + data[0][1] * b.data[1][0]
		       + data[0][2] * b.data[2][0],
			 data[0][0] * b.data[0][1] + data[0][1] * b.data[1][1]
		       + data[0][2] * b.data[2][1],
			 data[0][0] * b.data[0][2] + data[0][1] * b.data[1][2]
		       + data[0][2] * b.data[2][2],
			 data[0][0] * b.data[0][3] + data[0][1] * b.data[1][3]
		       + data[0][2] * b.data[2][3] + data[0][3],
			 data[1][0] * b.data[0][0] + data[1][1] * b.data[1][0]
		       + data[1][2] * b.data[2][0],
			 data[1][0] * b.data[0][1] + data[1][1] * b.data[1][1]
		       + data[1][2] * b.data[2][1],
			 data[1][0] * b.data[0][2] + data[1][1] * b.data[1][2]
		       + data[1][2] * b.data[2][2],
			 data[1][0] * b.data[0][3] + data[1][1] * b.data[1][3]
		       + data[1][2] * b.data[2][3] + data[1][3],
			 data[2][0] * b.data[0][0] + data[2][1] * b.data[1][0]
		       + data[2][2] * b.data[2][0],
			 data[2][0] * b.data[0][1] + data[2][1] * b.data[1][1]
		       + data[2][2] * b.data[2][1],
			 data[2][0] * b.data[0][2] + data[2][1] * b.data[1][2]
		       + data[2][2] * b.data[2][2],
			 data[2][0] * b.data[0][3] + data[2][1] * b.data[1][3]
		       + data[2][2] * b.data[2][3] + data[2][3]);
  }
  
  SJCVector3f operator*(const SJCVector3f &v) const {
    return SJCVector3f(  data[0][0] * v.x() + data[0][1] * v.y()
		     + data[0][2] * v.z() + data[0][3],
		       data[1][0] * v.x() + data[1][1] * v.y()
		     + data[1][2] * v.z() + data[1][3],
		       data[2][0] * v.x() + data[2][1] * v.y()
		     + data[2][2] * v.z() + data[2][3]);
    }
  SJCVector3f operator%(const SJCVector3f &v) const {
    return SJCVector3f(data[0][0]*v.x() + data[0][1]*v.y() + data[0][2]*v.z(),
		     data[1][0]*v.x() + data[1][1]*v.y() + data[1][2]*v.z(),
		     data[2][0]*v.x() + data[2][1]*v.y() + data[2][2]*v.z());
  }
  SJCVector4d operator*(const SJCVector4d &v) const {
    return SJCVector4d(  data[0][0] * v.x() + data[0][1] * v.y()
		      + data[0][2] * v.z() + data[0][3] * v.w(),
		        data[1][0] * v.x() + data[1][1] * v.y()
		      + data[1][2] * v.z() + data[1][3] * v.w(),
		        data[2][0] * v.x() + data[2][1] * v.y()
		      + data[2][2] * v.z() + data[2][3] * v.w(), v.w());
    }

  SJCRigidTransformf inverse() const {
    return SJCRigidTransformf(data[0][0], data[1][0], data[2][0],
			    -data[0][0] * data[0][3] - data[1][0] * data[1][3]
			    -data[2][0] * data[2][3],
			     data[0][1], data[1][1], data[2][1], 
			    -data[0][1] * data[0][3] - data[1][1] * data[1][3]
			    -data[2][1] * data[2][3],
			     data[0][2], data[1][2], data[2][2], 
			    -data[0][2] * data[0][3] - data[1][2] * data[1][3]
			    -data[2][2] * data[2][3]);
  }

  void    toComponents(float &angle, SJCVector3f &axis,
		       SJCVector3f &tran) const;

  void   invert(void) ;

  friend std::ostream& operator<<(std::ostream& o, const SJCRigidTransformf& m) ;
 

  static SJCRigidTransformf	rotateTransform(const float angle,
					        const SJCVector3f& axis);

  static SJCRigidTransformf	translateTransform(const SJCVector3f t) {
    return SJCRigidTransformf(1., 0., 0., t.x(),
			     0., 1., 0., t.y(),
			     0., 0., 1., t.z());
  }
  static SJCRigidTransformf	scaleTransform(const SJCVector3f s) {
    return SJCRigidTransformf(s.x(), 0., 0., 0.,
			      0., s.y(), 0., 0.,
			      0., 0., s.z(), 0.);
  }
  

};


#endif

