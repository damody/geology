/************************************************************************
     Main File:
 
     File:        SJCPlane.h
 
     Author:
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
   
     Comment:     This class implements a plane in 3 space and provides 
                  relevant methods. This class also provides overloaded 
                  stream input/output operators.

     Contructor:
                 1. 0 paras: default to d(0), n(0, 0, 1)
                 2. vector, vector: contruct plane from normal and a point
                 3. vector, vector, vector: contruct plane from 3 points

     Function:
                 1. IsAbove(vector): is point above plane
                 2. IsParallel(vector): is vector parallel plane normal
                 3. IsParallel(plane&): is two plane parallel
                 4. Intersect(vector, vector, vector): intersect plane with ray
                 5. Distance(point): distance to a point from this plane
                 
                 6. >>: input operator
                 7. <<: output operator
     Compiler:    g++
 
     Platform:    Linux
*************************************************************************/

#ifndef _SJCPLANE_H
#define _SJCPLANE_H

// Use definition
#include <SJC/SJC.h>

// C++ library
#include <iostream>

#include <SJCVector3.h>

class SJCPlane  
{
  // members
 public:
   float       m_distance;     // distance from plane to origin
   SJCVector3f    m_normal;       // normal of the plane

   // Constructor
   SJCPlane(void);
   SJCPlane(const SJCVector3f& normal, const SJCVector3f& point);

   // * Construct the plane that contains the three points specified.  The 
   //   points must be in counter clockwise order or the m_normal of the plane 
   SJCPlane(const SJCVector3f& pointA, const SJCVector3f& pointB, 
	    const SJCVector3f& pointC);
        
   // Destructor for the SJCPlane class.
   ~SJCPlane();
   // * Evaluate if point lies above the plane.  The m_normal of the plane is 
   //   assumed to point in the upward direction.
   bool IsAbove(const SJCVector3f& point) const;
        
   // * Return true if the given vector is parallel to this plane, false 
   //   otherwise.      
   bool IsParallel(const SJCVector3f&) const;
        
   // * Return true if the given plane is parallel to this one, false 
   //   otherwise.
   bool IsParallel(const SJCPlane&) const;
        
   // * Find the intersection of this plane and a line defined by a direction
   //   and a point on the line.  Return true if the line intersects the
   //   plane and return the point of intersection through the reference 
   //   parameter, otherwise return false.
   bool Intersect(const SJCVector3f& point, const SJCVector3f& direction, 
		  SJCVector3f& intersection) const;

   // * Find the distance the given point lies from the plane.
   float Distance(const SJCVector3f& point) const;

   // * Overloaded stream input operator.
   friend inline std::istream& operator >>(std::istream &in, SJCPlane& plane);

   // * Overloaded stream output operator.
   friend inline std::ostream& operator <<(std::ostream &out, 
					   const SJCPlane& plane);
};
#endif  // _CPLANE








