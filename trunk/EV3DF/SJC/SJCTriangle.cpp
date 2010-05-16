/************************************************************************
     Main File:
 
     File:        SJCTriangle.cpp
 
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
                GetBarycentricPt(float, float): from u, v get the point
                SamplePoint(): return a sample point on the triangle
                GetMinDistance(): return a minimum distance
                 
                 >>: input operator
                 <<: output operator
     Compiler:    g++
 
     Platform:    Linux
*************************************************************************/

#include "SJCTriangle.h"
 
//**********************************************************************
//
// * Default constructor for the SimPlane class.
//======================================================================
SJCTriangle::
SJCTriangle(SJCVector3f p1, SJCVector3f p2, SJCVector3f p3)
  : v1(p1), v2(p2), v3(p3)
//======================================================================
{
  SetDerivedValues();
}
 

//**********************************************************************
//
// * Set the derivative data from vertices
//======================================================================
void SJCTriangle::
SetDerivedValues(void)
//======================================================================
{
  e12 = v2 - v1;
  e31 = v1 - v3;

  area = 0.5 * (e12 % -e31).length();

  // Find the longest edge.
  SJCVector3f e23 = v3 - v2;
  
  // Compute the length of each esge
  float l12 = e12.length();
  float l23 = e23.length();
  float l31 = e31.length();
  
  // Uset
  if ( l12 > l23 )  {
    if ( l12 > l31 )     {
      sampleV  = v1;
      sampleW  = l12;
      sampleD1 = e12.normal();
     
      sampleWp = (-e31).ComponentWiseMultiply(sampleD1);
      sampleD2 = -e31 - ( sampleD1.ComponentWiseMultiply(sampleWp));
      sampleH  = sampleD2.length();
      sampleD2.normalize();
    }
    else   {
      sampleV  = v3;
      sampleW  = l31;
      sampleD1 = e31.normal();
      sampleWp = (-e23).ComponentWiseMultiply(sampleD1);
      sampleD2 = -e23 - ( sampleD1.ComponentWiseMultiply(sampleWp) );
      sampleH  = sampleD2.length();
      sampleD2.normalize();
    } // end of else
  } // end of if
  else  {
    if ( l23 > l31 )    {
      sampleV  = v2;
      sampleW  = l23;
      sampleD1 = e23.normal();
      sampleWp = (-e12).ComponentWiseMultiply(sampleD1);
      sampleD2 = -e12 - ( sampleD1.ComponentWiseMultiply(sampleWp) );
      sampleH = sampleD2.length();
      sampleD2.normalize();
    }
    else    {
      sampleV = v3;
      sampleW = l31;
      sampleD1 = e31.normal();
      sampleWp = (-e23). ComponentWiseMultiply(sampleD1);
      sampleD2 = -e23 - (sampleD1.ComponentWiseMultiply(sampleWp) );
      sampleH = sampleD2.length();
      sampleD2.normalize();
    } // end of else 
  } // end of else
}

//**********************************************************************
//
// * Reset all the vertex
//======================================================================
void SJCTriangle::
Set(const SJCVector3f &p1, const SJCVector3f &p2,  const SJCVector3f &p3)
//======================================================================
{
  v1 = p1;
  v2 = p2;
  v3 = p3;
  SetDerivedValues();
}

//**********************************************************************
//
// * Get the barycentric points
//======================================================================
SJCVector3f SJCTriangle::
GetBarycentricPt(const float u, const float v) const
//======================================================================
{
  return v1 + e12 * u - e31 * v;
}


//**********************************************************************
//
// * Get the minimum distance to triangle
// ? only works for 2D ??
//======================================================================
float SJCTriangle::
GetMinDistance(const SJCVector3f& point, float &u, float &v) const
//======================================================================
{
  //cerr << "from the book\n";
  // coefficients of F(t0,t1), calculation of c is deferred until needed
  SJCVector3f D0 = v2 - v1;
  SJCVector3f D1 = v3 - v1;
  SJCVector3f Delta = point - v1;
  
  float a00 = D0 * D0; 
  float a01 = D0 * D1; 
  float a11 = D1 * D1; 
   
  float b0 = D0 * Delta; 
  float b1 = D1 * Delta; 
  
  // Grad F(t0, t1) = (0,0) at (t0, t1) = (n0 / d, n1 / d)
  float n0 = a11 * b0 - a01 * b1;
  float n1 = a00 * b1 - a01 * b0;
  float d  = a00 * a11 - a01 * a01;
  
  if (n0 + n1 <= d) {
    if (n0 >= 0) {
      if (n1 >= 0) {
	// REGION 0.  Point is inside the triangle, suqared distance is zero
	u = n0 / d;
	v = n1 / d;
	return 0;
      } 
      else {
	// REGION 5.  Minimize G(t0) = F(t0,0) for t0 in [0,1].  G'(t0) = 0
	// at t0 = b0 / a00.
	float c = Delta * Delta;//Dot(Delta,Delta);
	if (b0 > 0) {
	  if (b0 < a00) {
	    // closest point is interior to the edge
	    u = b0 / a00;
	    v = 0;
	    return c - b0 * b0 / a00; // F(b0 / a00, 0)
	  } 
	  else {
	    // closest point is end point (t0,t1) = (1,0)
	    u = 1.0;
	    v = 0.0;
	    return a00 - 2 * b0 + c;  // F(1,0)
	  }
	}
	else {
	  // closest point is end point (t0,t1) = (0,0)
	  u = 0.0;
	  v = 0.0;
	  return c; // F(0,0)
	}
      }
    }
    else if (n1 >= 0) {
      // REGION 3.  Minimize G(t1) = F(0,t1) for 51 in [0,1].  G'(t1) = 0
      // at t1 = b1 / a11.
      float c = Delta * Delta;//Dot(Delta,Delta);
      if (b1 > 0) {
	if (b1 < a11) {
	  u = 0.0;
	  v = b1 / a11;
	  return c - b1 * b1 / a11;  // F(0, b1 / a11)
	} 
	else {
	  // closest point is end point (t0,t1) = (0,1)
	  u = 0.0;
	  v = 1.0;
	  return a11 - 2 * b1 + c;  // F(0, 1)
	}
      } 
      else {
	// closest point is end point (t0, t1) = (0, 0)
	u = 0.0;
	v = 0.0;
	return c; // F(0,0)
      }
    } 
    else {
      // REGION 4.  Minimize G(t0) = F(t0, 0) for t0 in [0,1].  If t0 >
      // 1, the parameter pair (min{1, t0}, 0) produces the closest
      // point.  If t0 = 0, then minimize H(t1) = F(0, t1) for t1 in [0,
      // 1].  G'(t0) = 0 at t0 = b0 / a00.  H'(t1) = 0 at t1 = b1 / a11.
      
      float c = Delta * Delta;//Dot(Delta, Delta);
      
      // minimize on edge t1 = 0
      if (b0 < a00) {
	if (b0 > 0) {
	  // closest point is interior to edge
	  u = b0 / a00;
	  v = 0.0;
	  return c - b0 * b0 / a00;  // F(b0 / a00, 0)
	}
	else {
	  // minimize edge t0 = 0
	  if (b1 < a11) {
	    if (b1 > 0) {
	      // closest point is interionr to edge 
	      u = 0.0;
	      v = b1 / a11;
	      return c - b1 * b1 / a11; // F(0, b1/a11)
	    } else {
	      // closest point is end point (t0, t1) = (0,0)
	      u = 0.0;
	      v = 0.0;
	      return c; // F(0,0)
	    }
	  } 
	  else {
	    // closest point is end point (t0, t1) = (0,1)
	    u = 0.0;
	    v = 1.0;
	    return a11 - 2 * b1 + c; // F(0,1)
	  }
	}
      } 
      else {
	// closest point is end point (50, t1) = (1,0)
	u = 1.0;
	v = 0.0;
	return a00 - 2 * b0 + c;  // F(1,0)
      }
    }
  }
  else if (n0 < 0) {
    // REGION 2.  Minimize G(t1) = f(0,t1) for t1 in [0,1].  If t1 < 1,
    // the parameter pair (0, max{0,t1}) produces the closest point.
    // If t1 = 1, then minimize H(t0) = F(t0, 1-t0) for t0 in [0,1].
    // G'(t1) = 0 at t1 = b1 / a11.  H'(t0) = 0 at t0 = (a11 - a01 + b0
    // - b1) / (a00 - 2 * a01 + a11).
    float c = Delta * Delta; //Dot(Delta, Delta);
    
    // minimze on edge t0 = 0
    if (b1 > 0) {
      if (b1 < a11) {
	// closest point is interior to edge
	u = 0.0;
	v = b1 / a11;
	return c - b1 * b1 / a11; // F(0, b1 / a11)
      } 
      else {
	// minimze on the edge t0 + t1 = 1
	float n = a11 - a01 + b0 - b1;
	float d = a00 - 2 * a01 + a11;
	if (n > 0) {
	  if (n < d) {
	    // closest point is interior to the edge
	    u = n / d;
	    v = 1.0 - n / d;
	    return (a11 - 2 * b1 + c) - n * n / d; // F(n/d, 1-n/d)
	  } 
	  else {
	    // closest point is end point (t0,t1) = (1,0)
	    u = 1.0;
	    v = 0.0;
	    return a00 - 2 * b0 + c; // F(1,0)
	  }
	}
	else {
	  // closest point is end point (t0, t1) = (1, 0)
	  u = 1.0;
	  v = 0.0;
	  return a11 - 2 * b1 + c;  // F(0,1)
	}
      }
    }
    else {
      // the closest point is end point (t0, t1) = (0,0)
      u = 0.0;
      v = 0.0;
      return c;
    }
  }
  else if (n1 < 0) {
    // REGION 6.  Mimize G(t0) = F(t0, 0) for t0 in [0,1].  If t0 < 1,
    // the parameter pair (max{0,t0},0) produces the closest point.  If
    // t0 = 1, then minimze H(t1) = F(t1, 1-t1) for t1 in [0,1].
    // G'(t0) = 0 at t0 = b0 / a00.  H'(t1) = 0 at t1 = (a11 - a01 + b0
    // - b1) / (a00 - 2 * a01 + a11).
    
    float c = Delta * Delta;//Dot(Delta, Delta);
    
    // minimize on edge t1 = 0
    if (b0 > 0) {
      if (b0 < a00) {
	// closest point is interior to the edge
	u = b0 / a00;
	v = 0.0;
	return c - b0 * b0 / a00;  // F(b0 / a00, 0)
      } 
      else {
	float n = a11 - a01 + b0 - b1;
	float d = a00 - 2 * a01 + a11;
	if (n > 0) {
	  if (n < d) {
	    // closest point is interior to the edge
	    u = n / d;
	    v = 1.0 - n / d;
	    return (a11 - 2 * b1 + c) - n * n / d;  // F(n/d, 1-n/d)
	  } else {
	    // closest point is end point (t0, t1) = (1,0)
	    u = 1.0;
	    v = 0.0;
	    return a00 - 2 * b0 + c;  // F(1,0)
	  }
	} 
	else {
	  // closest point is end point (t0, t1) = (0,1)
	  u = 0.0;
	  v = 1.0;
	  return a11 - 2 * b1 + c; // F(0, 1)
	}
      }
    } 
    else {
      // closest point is end point (t0, t1) = (0, 0)
      u = 0.0;
      v = 0.0;
      return c; // F(0,0)
    }
  } 
  else {
    // REGION 1.  Minimize G(t0) = F(t0, 1 - t0) for t0 in [0,1].
    // G'(t0) = 0 at t0 = (a11 - a01 + b0 - b1) / (a00 - 2 * a01 +
    // a11).
   
    float c = Delta * Delta;//Dot(Delta, Delta);
    float n = a11 - a01 + b0 - b1;
    float d = a00 - 2 * a01 + a11;
    
    if (n > 0) {
      if (n < d) {
	// closest point is interior to edge 
	u = n / d;
	v = 1.0 - n / d;
	return (a11 - 2 * b1 + c) - n * n / d; // F(n / d, 1 - n / d)
      } 
      else {
	// closest point is end point (t0, t1) = (1, 0)
	u = 1.0;
	v = 0.0;
	return a00 - 2 * b0 + c; // F(1, 0)
      }
    } 
    else {
      // closest point is end point (t0, t1) = (0, 1)
      u = 0.0;
      v = 1.0;
      return a11 - 2 * b1 + c; // F(0, 1)
    }
  }
  
  SJCError("Shouldn't have ended up here");
  
  return 0.f;
}// GetMinDistance

//**********************************************************************
//
// * Default constructor for the SimPlane class.
//======================================================================
SJCVector3f SJCTriangle::
SamplePoint(SJCRandomf &random) const
//======================================================================
{
  float u = random.Uniform(0.f, 1.f) * sampleW;
  float v = random.Uniform(0.f, 1.f) * sampleH;
  
  if ( u <= sampleWp ) {
    if ( u == 0.0 )   {
      if ( v != 0.0 ) {
	u = sampleWp;
	v = sampleH - v;
      }
    }
    else if ( v / u > sampleH / sampleWp )    {
      u = sampleWp - u;
      v = sampleH - v;
    }
  }
  else  {
    float wp = sampleW - sampleWp;
    float du = sampleW - u;
    if ( du == 0.0 )     {
      if ( v != 0.0 ){
	u = sampleWp;
	v = sampleH - v;
      }
    }
    else if ( v / du > sampleH / wp )  {
      u = sampleWp + du;
      v = sampleH - v;
    }
  }

  return sampleV + sampleD1 * u + sampleD2 * v;
}

//**********************************************************************
//
// * output operator
//======================================================================
std::ostream& operator <<(std::ostream& outStream, SJCTriangle& tri)
//======================================================================
{
  outStream << "[ " << tri.v1 << " : " << tri.v2 << " : " << tri.v3 << " ]";
  return outStream;
}// operator >>

//**********************************************************************
//
// * input operator
//======================================================================
std::istream& operator >>(std::istream& inStream, SJCTriangle& tri)
//======================================================================
{
  inStream >> tri.v1 >> tri.v2 >> tri.v3;
  tri.setDerivedValues();
  return inStream;
}// operator >>

