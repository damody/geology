/*
** $Header: /p/graphics/CVS/yu-chi/SJCMatrix44.h,v 1.1.1.1 2006/04/25 20:21:43 yu-chi Exp $
**
** (c) 2002 - 2005 Stephen Chenney
**
** $Log: SJCMatrix44.h,v $
** Revision 1.1.1.1  2006/04/25 20:21:43  yu-chi
**
**
** Revision 1.2  2005/08/31 16:42:04  schenney
** Renamed the std::vector classes for greater consistency.
**
** Revision 1.1.1.1  2005/08/29 20:24:14  schenney
** New libSJC source repository.
**
*/


#ifndef SJCMATRIX44_H_
#define SJCMATRIX44_H_

#include <stdio.h>
#include <string.h>

#include "SJC.h"
#include "SJCVector3.h"
#include "SJCVector4.h"

class SJCDLL SJCMatrix44 {
  public:

    SJCMatrix44() {
	data[0][0] = data[1][1] = data[2][2] = data[3][3] = 1.0;
	data[0][1] = data[0][2] = data[0][3] =
	data[1][0] = data[1][2] = data[1][3] =
	data[2][0] = data[2][1] = data[2][3] =
	data[3][0] = data[3][1] = data[3][2] = 0.0; }
    SJCMatrix44(const double x00, const double x01,
                const double x02, const double x03,
	        const double x10, const double x11,
	        const double x12, const double x13,
	        const double x20, const double x21,
	        const double x22, const double x23,
	        const double x30, const double x31,
	        const double x32, const double x33) {
	data[0][0] = x00; data[0][1] = x01; data[0][2] = x02; data[0][3] = x03;
	data[1][0] = x10; data[1][1] = x11; data[1][2] = x12; data[1][3] = x13;
	data[2][0] = x20; data[2][1] = x21; data[2][2] = x22; data[2][3] = x23;
	data[3][0] = x30; data[3][1] = x31; data[3][2] = x32; data[3][3] = x33;
    }
    SJCMatrix44(const SJCVector4d x, const SJCVector4d y, const SJCVector4d z,
		const SJCVector4d w) {
	data[0][0] = x.x(); data[0][1] = x.y();
	data[0][2] = x.z(); data[0][3] = x.w();
	data[1][0] = y.x(); data[1][1] = y.y();
	data[1][2] = y.z(); data[1][3] = x.w();
	data[2][0] = z.x(); data[2][1] = z.y();
	data[2][2] = z.z(); data[2][3] = x.w();
	data[3][0] = z.x(); data[3][1] = z.y();
	data[3][2] = z.z(); data[3][3] = x.w();
    }
    SJCMatrix44( const SJCMatrix44 &other ) {
	memcpy( data, other.data, 16*sizeof(double) );
    }

    double& operator()(const int i, const int j) { return data[i][j]; }

    SJCVector4d operator[](const int i) const {
	return SJCVector4d(data[i][0], data[i][1], data[i][2], data[i][3]);
    }

    SJCMatrix44 &operator=(const SJCMatrix44 &other) {
	memcpy( data, other.data, 16*sizeof(double) );
	return *this;
    }

    SJCMatrix44 operator*(const SJCMatrix44 &b) const {
	return SJCMatrix44(data[0][0] * b.data[0][0]
	    		 + data[0][1] * b.data[1][0]
	    		 + data[0][2] * b.data[2][0]
	    		 + data[0][3] * b.data[3][0],
			   data[0][0] * b.data[0][1]
			 + data[0][1] * b.data[1][1]
			 + data[0][2] * b.data[2][1]
			 + data[0][3] * b.data[3][1],
			   data[0][0] * b.data[0][2]
			 + data[0][1] * b.data[1][2]
			 + data[0][2] * b.data[2][2]
			 + data[0][3] * b.data[3][2],
			   data[0][0] * b.data[0][3]
			 + data[0][1] * b.data[1][3]
			 + data[0][2] * b.data[2][3]
			 + data[0][3] * b.data[3][3],
			   data[1][0] * b.data[0][0]
	    		 + data[1][1] * b.data[1][0]
	    		 + data[1][2] * b.data[2][0]
	    		 + data[1][3] * b.data[3][0],
			   data[1][0] * b.data[0][1]
			 + data[1][1] * b.data[1][1]
			 + data[1][2] * b.data[2][1]
			 + data[1][3] * b.data[3][1],
			   data[1][0] * b.data[0][2]
			 + data[1][1] * b.data[1][2]
			 + data[1][2] * b.data[2][2]
			 + data[1][3] * b.data[3][2],
			   data[1][0] * b.data[0][3]
			 + data[1][1] * b.data[1][3]
			 + data[1][2] * b.data[2][3]
			 + data[1][3] * b.data[3][3],
			   data[2][0] * b.data[0][0]
	    		 + data[2][1] * b.data[1][0]
	    		 + data[2][2] * b.data[2][0]
	    		 + data[2][3] * b.data[3][0],
			   data[2][0] * b.data[0][1]
			 + data[2][1] * b.data[1][1]
			 + data[2][2] * b.data[2][1]
			 + data[2][3] * b.data[3][1],
			   data[2][0] * b.data[0][2]
			 + data[2][1] * b.data[1][2]
			 + data[2][2] * b.data[2][2]
			 + data[2][3] * b.data[3][2],
			   data[2][0] * b.data[0][3]
			 + data[2][1] * b.data[1][3]
			 + data[2][2] * b.data[2][3]
			 + data[2][3] * b.data[3][3],
			   data[3][0] * b.data[0][0]
	    		 + data[3][1] * b.data[1][0]
	    		 + data[3][2] * b.data[2][0]
	    		 + data[3][3] * b.data[3][0],
			   data[3][0] * b.data[0][1]
			 + data[3][1] * b.data[1][1]
			 + data[3][2] * b.data[2][1]
			 + data[3][3] * b.data[3][1],
			   data[3][0] * b.data[0][2]
			 + data[3][1] * b.data[1][2]
			 + data[3][2] * b.data[2][2]
			 + data[3][3] * b.data[3][2],
			   data[3][0] * b.data[0][3]
			 + data[3][1] * b.data[1][3]
			 + data[3][2] * b.data[2][3]
			 + data[3][3] * b.data[3][3]);
    }

    SJCVector4d operator*(const SJCVector4d &v) const {
	return SJCVector4d(data[0][0] * v.x() + data[0][1] * v.y()
		        + data[0][2] * v.z() + data[0][3] * v.w(),
		          data[1][0] * v.x() + data[1][1] * v.y()
		        + data[1][2] * v.z() + data[1][3] * v.w(),
		          data[2][0] * v.x() + data[2][1] * v.y()
		        + data[2][2] * v.z() + data[2][3] * v.w(),
		          data[3][0] * v.x() + data[3][1] * v.y()
		        + data[3][2] * v.z() + data[3][3] * v.w());
    }
    SJCVector4f operator*(const SJCVector4f &v) const {
	return SJCVector4f(data[0][0] * v.x() + data[0][1] * v.y()
		         + data[0][2] * v.z() + data[0][3] * v.w(),
		           data[1][0] * v.x() + data[1][1] * v.y()
		         + data[1][2] * v.z() + data[1][3] * v.w(),
		           data[2][0] * v.x() + data[2][1] * v.y()
		         + data[2][2] * v.z() + data[2][3] * v.w(),
		           data[3][0] * v.x() + data[3][1] * v.y()
		         + data[3][2] * v.z() + data[3][3] * v.w());
    }

    void glForm(double vals[16]) const {
	vals[0]=data[0][0];  vals[1]=data[1][0];
	vals[2]=data[2][0];  vals[3]=data[3][0];
	vals[4]=data[0][1];  vals[5]=data[1][1];
	vals[6]=data[2][1];  vals[7]=data[3][1];
	vals[8]=data[0][2];  vals[9]=data[1][2];
	vals[10]=data[2][2]; vals[11]=data[3][2];
	vals[12]=data[0][3]; vals[13]=data[1][3];
	vals[14]=data[2][3]; vals[15]=data[3][3];
    }

    void glForm(float vals[16]) const {
	vals[0]=data[0][0];  vals[1]=data[1][0];
	vals[2]=data[2][0];  vals[3]=data[3][0];
	vals[4]=data[0][1];  vals[5]=data[1][1];
	vals[6]=data[2][1];  vals[7]=data[3][1];
	vals[8]=data[0][2];  vals[9]=data[1][2];
	vals[10]=data[2][2]; vals[11]=data[3][2];
	vals[12]=data[0][3]; vals[13]=data[1][3];
	vals[14]=data[2][3]; vals[15]=data[3][3];
    }

    static SJCMatrix44	Rotation(const double angle, const SJCVector3d& axis);
    static SJCMatrix44	Rotation(const float angle, const SJCVector3f& axis);
    static SJCMatrix44	Scale(const SJCVector3d& s);
    static SJCMatrix44	Scale(const SJCVector3f& s);
    static SJCMatrix44	Translation(const SJCVector3d& t);
    static SJCMatrix44	Translation(const SJCVector3f& t);

    friend std::ostream& operator<<(std::ostream& o, const SJCMatrix44& m);

  private:
    double data[4][4];  // [row][column]
};


#endif

