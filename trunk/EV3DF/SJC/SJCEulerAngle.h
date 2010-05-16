/************************************************************************
     Main File:   main.cpp
  
     File:        SJCEulerAngle.h
  
     Author:
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
       
     Comment:     EulerAngle operation
                                                 

     Functions:   
                  1. Eul_: set up the euler rotation
                  2. Eul_FromHMatrix: set up euler angles from matrix
                  3. Eul_FromQuat: set up euler from quaternion
                             
     Compiler:    g++
 

     Platform:    Linux
*************************************************************************/

#ifndef _SJC_EULERANGLE_H
#define _SJC_EULERANGLE_H

#include <SJC/SJC.h>

#include <SJC/SJCQuaternion.h>


//****************************************************************************
//
// * Definition for Euler Angle
//
//****************************************************************************

enum QuatPart { 
  X, 
  Y, 
  Z, 
  W
};

typedef float HMatrix[4][4];   /* Right-handed, for column vectors */

/**** EulerAngles.h - Support for 24 angle schemes ****/
/* Ken Shoemake, 1993 */

/*** Order type constants, constructors, extractors ***
  There are 24 possible conventions, designated by:    
  o EulAxI = axis used initially		    
  o EulPar = parity of axis permutation		    
  o EulRep = repetition of initial axis as last	   
  o EulFrm = frame from which axes are taken	    
  Axes I,J,K will be a permutation of X,Y,Z.	    
  Axis H will be either I or K, depending on EulRep.   
  Frame S takes axes from initial static frame.	    
  If ord = (AxI=X, Par=Even, Rep=No, Frm=S), then	    
  {a,b,c,ord} means Rz(c)Ry(b)Rx(a), where Rz(c)v	    
  rotates v around Z by c radians.			    */

#define EulFrmS	     0
#define EulFrmR	     1
#define EulFrm(ord)  ((unsigned)(ord)&1)

#define EulRepNo     0
#define EulRepYes    1
#define EulRep(ord)  (((unsigned)(ord)>>1)&1)

#define EulParEven   0
#define EulParOdd    1
#define EulPar(ord)  (((unsigned)(ord)>>2)&1)

#define EulSafe	     "\000\001\002\000"
#define EulNext	     "\001\002\000\001"

#define EulAxI(ord)  ((int)(EulSafe[(((unsigned)(ord)>>3)&3)]))
#define EulAxJ(ord)  ((int)(EulNext[EulAxI(ord)+(EulPar(ord)==EulParOdd)]))
#define EulAxK(ord)  ((int)(EulNext[EulAxI(ord)+(EulPar(ord)!=EulParOdd)]))
#define EulAxH(ord)  ((EulRep(ord)==EulRepNo)?EulAxK(ord):EulAxI(ord))

// EulGetOrd unpacks all useful information about order simultaneously. 
#define EulGetOrd(ord,i,j,k,h,n,s,f) {unsigned o=ord;f=o&1;o>>=1;s=o&1;o>>=1;\
    n=o&1;o>>=1;i=EulSafe[o&3];j=EulNext[i+n];k=EulNext[i+1-n];h=s?k:i;}

// EulOrd creates an order value between 0 and 23 from 4-tuple choices. 
#define EulOrd(i,p,r,f)	   (((((((i)<<1)+(p))<<1)+(r))<<1)+(f))

// Static axes 
#define EulOrdXYZs    EulOrd(X, EulParEven, EulRepNo,  EulFrmS)
#define EulOrdXYXs    EulOrd(X, EulParEven, EulRepYes, EulFrmS)
#define EulOrdXZYs    EulOrd(X, EulParOdd,  EulRepNo,  EulFrmS)
#define EulOrdXZXs    EulOrd(X, EulParOdd,  EulRepYes, EulFrmS)
#define EulOrdYZXs    EulOrd(Y, EulParEven, EulRepNo,  EulFrmS)
#define EulOrdYZYs    EulOrd(Y, EulParEven, EulRepYes, EulFrmS)
#define EulOrdYXZs    EulOrd(Y, EulParOdd,  EulRepNo,  EulFrmS)
#define EulOrdYXYs    EulOrd(Y, EulParOdd,  EulRepYes, EulFrmS)
#define EulOrdZXYs    EulOrd(Z, EulParEven, EulRepNo,  EulFrmS)
#define EulOrdZXZs    EulOrd(Z, EulParEven, EulRepYes, EulFrmS)
#define EulOrdZYXs    EulOrd(Z, EulParOdd,  EulRepNo,  EulFrmS)
#define EulOrdZYZs    EulOrd(Z, EulParOdd,  EulRepYes, EulFrmS)

// Rotating axes 
#define EulOrdZYXr    EulOrd(X, EulParEven, EulRepNo,  EulFrmR)
#define EulOrdXYXr    EulOrd(X, EulParEven, EulRepYes, EulFrmR)
#define EulOrdYZXr    EulOrd(X, EulParOdd,  EulRepNo,  EulFrmR)
#define EulOrdXZXr    EulOrd(X, EulParOdd,  EulRepYes, EulFrmR)
#define EulOrdXZYr    EulOrd(Y, EulParEven, EulRepNo,  EulFrmR)
#define EulOrdYZYr    EulOrd(Y, EulParEven, EulRepYes, EulFrmR)
#define EulOrdZXYr    EulOrd(Y, EulParOdd,  EulRepNo,  EulFrmR)
#define EulOrdYXYr    EulOrd(Y, EulParOdd,  EulRepYes, EulFrmR)
#define EulOrdYXZr    EulOrd(Z, EulParEven, EulRepNo,  EulFrmR)
#define EulOrdZXZr    EulOrd(Z, EulParEven, EulRepYes, EulFrmR)
#define EulOrdXYZr    EulOrd(Z, EulParOdd,  EulRepNo,  EulFrmR)
#define EulOrdZYZr    EulOrd(Z, EulParOdd,  EulRepYes, EulFrmR)

class SJCEulerAngle{
 public:
  float x_, y_, z_, w_;

 public:
  SJCEulerAngle(void){}
  ~SJCEulerAngle(void){}

  static SJCEulerAngle Eul_(float ai, float aj, float ah, int order){
    SJCEulerAngle ea;
    ea.x_ = ai; 
    ea.y_ = aj; 
    ea.z_ = ah;
    ea.w_ = order;
    return (ea);
  }
  static SJCEulerAngle Eul_FromHMatrix(HMatrix M, int order){
    SJCEulerAngle ea;
    int i, j, k, h, n, s, f;
    EulGetOrd(order, i, j, k, h, n, s, f);
    if (s==EulRepYes) {
      float sy = sqrt(M[i][j]*M[i][j] + M[i][k]*M[i][k]);
      if (sy > 16*SJC_EPSILON) {
	ea.x_ = atan2((double)M[i][j], (double)M[i][k]);
	ea.y_ = atan2((double)sy, (double)M[i][i]);
	ea.z_ = atan2((double)M[j][i], (double)-M[k][i]);
      } 
      else {
	ea.x_ = atan2((double)-M[j][k], (double)M[j][j]);
	ea.y_ = atan2((double)sy,(double) M[i][i]);
	ea.z_ = 0;
      }
    } 
    else {
      float cy = sqrt((double)(M[i][i] * M[i][i] + M[j][i] * M[j][i]));
      if (cy > 16*SJC_EPSILON) {
	ea.x_ = atan2((double)M[k][j], (double)M[k][k]);
	ea.y_ = atan2((double)-M[k][i], (double)cy);
	ea.z_ = atan2((double)M[j][i], (double)M[i][i]);
      } 
      else {
	ea.x_ = atan2((double)-M[j][k], (double)M[j][j]);
	ea.y_ = atan2((double)-M[k][i], (double)cy);
	ea.z_ = 0;
      }
    }
    if (n==EulParOdd) {
      ea.x_ = -ea.x_; 
      ea.y_ = -ea.y_; 
      ea.z_ = -ea.z_;
    }
    if (f == EulFrmR) {
      float t = ea.x_; 
      ea.x_   = ea.z_; 
      ea.z_   = t;
    }
    ea.w_ = order;
    return (ea);
  }

  // Convert quaternion to Euler angles (in radians). 
  static SJCEulerAngle Eul_FromQuat(SJCQuaternionf q, int order) {
    HMatrix M;
    float Nq = q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w();
    float s = (Nq > 0.0) ? (2.0 / Nq) : 0.0;
    float xs = q.x() * s,   ys = q.y() * s,  zs = q.z() * s;
    float wx = q.w() * xs,  wy = q.w() * ys, wz = q.w() * zs;
    float xx = q.x() * xs,  xy = q.x() * ys, xz = q.x() * zs;
    float yy = q.y() * ys,  yz = q.y() * zs, zz = q.z() * zs;

    M[X][X] = 1.0 - (yy + zz); M[X][Y] = xy - wz; M[X][Z] = xz + wy;
    M[Y][X] = xy + wz; M[Y][Y] = 1.0 - (xx + zz); M[Y][Z] = yz - wx;
    M[Z][X] = xz - wy; M[Z][Y] = yz + wx; M[Z][Z] = 1.0 - (xx + yy);
    M[W][X] = M[W][Y] = M[W][Z] = M[X][W] = M[Y][W] = M[Z][W] = 0.0; 
    M[W][W] = 1.0;
    return (Eul_FromHMatrix(M, order));
  }
};


#endif
