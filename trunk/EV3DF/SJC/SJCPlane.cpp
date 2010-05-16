/************************************************************************
     Main File:
 
     File:        SJCPlane.cpp
 
     Author:
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
   
     Comment:     This class implements a plane in 3 space and provides 
                  relevant methods. This class also provides overloaded 
                  stream input/output operators.

     Compiler:    g++
 
     Platform:    Linux
*************************************************************************/

#include <SJCPlane.h>

//**********************************************************************
//
// * Default constructor for the SJCPlane class.
//======================================================================
SJCPlane::
SJCPlane(void)
//======================================================================
{
  m_distance = 0.f;
  m_normal.set(0.f, 0.f, 1.f);
}// SJCPlane


//**********************************************************************//
//
// * Construct the plane which contains the given point and has the given
//   m_normal.
//======================================================================
SJCPlane::
SJCPlane(const SJCVector3f& normal, const SJCVector3f& point) 
         : m_distance(-point * normal), m_normal(normal.normal())
//======================================================================
{

}// SJCPlane


//**********************************************************************//
//
//  * Construct the plane that contains the three points specified.  The 
//    points must be in counter clockwise order or the m_normal of the plane
//    will be inverted.
//======================================================================
SJCPlane::
SJCPlane(const SJCVector3f& pointA, const SJCVector3f& pointB,	
	 const SJCVector3f& pointC)
//======================================================================
{
  m_normal.x( pointA.y() * (pointB.z() - pointC.z()) + 
	      pointB.y() * (pointC.z() - pointA.z()) + 
	      pointC.y() * (pointA.z() - pointB.z())
	      );
  m_normal.y( pointA.z() * (pointB.x() - pointC.x()) + 
	      pointB.z() * (pointC.x() - pointA.x()) + 
	      pointC.z() * (pointA.x() - pointB.x())
	      );
  m_normal.z( pointA.x() * (pointB.y() - pointC.y()) + 
	      pointB.x() * (pointC.y() - pointA.y()) + 
	      pointC.x() * (pointA.y() - pointB.y())
	      );
  m_distance = -m_normal * pointA;
}// ConstructPlane

//**********************************************************************//
//
// * Destructor for the SJCPlane class.
//======================================================================
SJCPlane::
~SJCPlane(void)
//======================================================================
{
}// ~SJCPlane


//**********************************************************************//
//
// * Evaluate if point lies above the plane.  The m_normal of the plane is 
//   assumed to point in the upward direction.
//======================================================================
bool SJCPlane::
IsAbove(const SJCVector3f& point) const
//======================================================================
{
   return (m_normal * point + m_distance > SJC_EPSILON);
}// IsAbove


//**********************************************************************
//
// * Return true if the given vector is parallel to this plane, false 
//   otherwise.      
//
//======================================================================
bool SJCPlane::
IsParallel(const SJCVector3f& v) const
//======================================================================
{
   return fabs(v.normal() * m_normal) < SJC_EPSILON;
}// IsParallel


//**********************************************************************
//
// * Return true if the given plane is parallel to this one, false 
//   otherwise.
//======================================================================i
bool SJCPlane::
IsParallel(const SJCPlane& plane) const
//======================================================================i
{
   return fabs(plane.m_normal * m_normal) > (1.f - SJC_EPSILON);
}// IsParallel


//**********************************************************************
//
// * Find the intersection of this plane and a line defined by a direction
//   and a point on the line.  Return true if the line intersects the plane and
//   return the point of intersection through the reference parameter, 
//   otherwise return false.
//
//======================================================================
bool SJCPlane::
Intersect(const SJCVector3f& point, const SJCVector3f& direction, 
	  SJCVector3f& intersection) const
//======================================================================i
{
  if (IsParallel(direction))
    return false;
  else  {
   
    intersection = point + direction * -( (m_normal * point) + 
                   m_distance) / (m_normal * direction);
    return true;
  }// else 
}// Intersect


//**********************************************************************
//
// * Find the distance the given point lies from the plane.
//======================================================================
float SJCPlane::
Distance(const SJCVector3f& point) const
//======================================================================
{
  SJCVector3f intersectionPoint;
  if (!Intersect(point, m_normal, intersectionPoint))
    return 0;
  else {   
    intersectionPoint -= point;
    return intersectionPoint.length();
  }
}// Distance

//**********************************************************************
//
// * Overloaded stream input operator.
//======================================================================
std::istream& operator>>(std::istream& in, SJCPlane& plane)
//======================================================================
{
  //  in >> plane.m_normal >> plane.m_distance;
  return in;
}// operator >>


//**********************************************************************
//
// * Overloaded stream output operator.
//======================================================================
inline std::ostream& operator <<(std::ostream& out, const SJCPlane& plane)
//======================================================================
{
  out << plane.m_normal << "  " << plane.m_distance;
  return out;
}// operator <<
