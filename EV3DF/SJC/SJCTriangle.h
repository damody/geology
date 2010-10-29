/************************************************************************
     Main File:
 
     File:        SJCTriangle.h
 
     Author:
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
   
     Comment:     This class implements a plane in 3 space and provides 
                  relevant methods. This class also provides overloaded 
                  stream input/output operators.

     Contructor:
                 0 paras: default to v1(0), v2(0), v3(0))
                 vector, vector, vector: contruct triangle from 3 vertices

     Function: 
                V1, V2, V3() return the vertices coordinate
                V1(vector), V2(vector), V3(vector) set the vertices value
                Set(vector, vector, vector0: set the vector value
                GetBarycentricPt(Real, Real): from u, v get the point
                SamplePoint(): return a sample point on the triangle
                GetMinDistance(): return a minimum distance
                 
                 >>: input operator
                 <<: output operator
     Compiler:    g++
 
     Platform:    Linux
*************************************************************************/

#ifndef _SJCTRIANGLE_H
#define _SJCTRIANGLE_H

#include <SJC/SJC.h>

#include <SJC/SJCVector3.h>
#include <SJC/SJCRandom.h>

class SJCTriangle {
 private:
  SJCVector3f    v1;  // vertex 1
  SJCVector3f    v2;  // vertex 2
  SJCVector3f    v3;  // vertex 3

  SJCVector3f    e12; // Edge vector from V1 to V2
  SJCVector3f    e31; // Edge vector from V3 to V1

  float          area; // Area of the triangle

  SJCVector3f    sampleV;  // Which vertex is the base sampling vertices
  SJCVector3f    sampleD1; // which two edge is the sampling direction
  SJCVector3f    sampleD2;
  float          sampleW;  // D1's length
  float          sampleH;  // D2's length
  float          sampleWp;

  void	         SetDerivedValues(void);
 public:
  // Constructor and destructor
  SJCTriangle(void) {}
  SJCTriangle(SJCVector3f p1, SJCVector3f p2, SJCVector3f p3);
  ~SJCTriangle(void) {}
  
  // Set and get method
  const SJCVector3f&   V1(void) const { return v1; }
  const SJCVector3f&   V2(void) const { return v2; }
  const SJCVector3f&   V3(void) const { return v3; }
  SJCVector3f&   V1(void) { return v1; }
  SJCVector3f&   V2(void) { return v2; }
  SJCVector3f&   V3(void) { return v3; }
  void  V1(const SJCVector3f& v) { v1 = v; SetDerivedValues(); }
  void  V2(const SJCVector3f& v) { v2 = v; SetDerivedValues(); }
  void  V3(const SJCVector3f& v) { v3 = v; SetDerivedValues(); }
  void	Set(const SJCVector3f &p1, const SJCVector3f &p2, 
	    const SJCVector3f &p3);
  
  const float Area(void) const { return area; }

  SJCVector3f GetBarycentricPt(const float u, const float v) const;

  SJCVector3f SamplePoint(SJCRandomf &random) const;

  float GetMinDistance(const SJCVector3f& point, float &u, float &v) const;
   
  friend std::istream& operator >>(std::istream& inStream, SJCTriangle& tri);
  friend std::ostream& operator <<(std::ostream& outStream, SJCTriangle& tri);
};// CTraingleSet

#endif // _CTRIANGLE
