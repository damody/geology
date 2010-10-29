/*
** $Header: /p/graphics/CVS/yu-chi/SJCMatrix44.cpp,v 1.1.1.1 2006/04/25 20:21:44 yu-chi Exp $
**
** (c) 2003 - 2005 Stephen Chenney
**
** $Log: SJCMatrix44.cpp,v $
** Revision 1.1.1.1  2006/04/25 20:21:44  yu-chi
**
**
** Revision 1.2  2005/08/31 16:42:04  schenney
** Renamed the std::vector classes for greater consistency.
**
** Revision 1.1.1.1  2005/08/29 20:24:14  schenney
** New libSJC source repository.
**
*/

#include <math.h>
#include <float.h>
#include "SJCRotateMatrix.h"
#include "SJCMatrix44.h"

#ifndef M_PI
#define M_PI            3.14159265358979323846
#define M_PI_2          1.57079632679489661923
#define M_PI_4          0.78539816339744830962
#define M_1_PI          0.31830988618379067154
#define M_2_PI          0.63661977236758134308
#define M_SQRT2         1.41421356237309504880
#define M_SQRT1_2       0.70710678118654752440
#endif


SJCMatrix44
SJCMatrix44::Rotation(const double angle, const SJCVector3d& axis)
{
  SJCRotateMatrixd r(angle, axis);
  return SJCMatrix44(r[0][0], r[0][1], r[0][2], 0.0,
		     r[1][0], r[1][1], r[1][2], 0.0,
		     r[2][0], r[2][1], r[2][2], 0.0,
		     0.0, 0.0, 0.0, 1.0);
}


SJCMatrix44
SJCMatrix44::Rotation(const float angle, const SJCVector3f& axis)
{
  SJCRotateMatrixf r(angle, axis);
  return SJCMatrix44(r[0][0], r[0][1], r[0][2], 0.0,
		     r[1][0], r[1][1], r[1][2], 0.0,
		     r[2][0], r[2][1], r[2][2], 0.0,
		     0.0, 0.0, 0.0, 1.0);
}

SJCMatrix44
SJCMatrix44::Scale(const SJCVector3d& s)
{
    return SJCMatrix44(s.x(), 0.0, 0.0, 0.0,
    		       0.0, s.y(), 0.0, 0.0,
    		       0.0, 0.0, s.z(), 0.0,
    		       0.0, 0.0, 0.0, 1.0);
}

SJCMatrix44
SJCMatrix44::Scale(const SJCVector3f& s)
{
    return SJCMatrix44(s.x(), 0.0, 0.0, 0.0,
    		       0.0, s.y(), 0.0, 0.0,
    		       0.0, 0.0, s.z(), 0.0,
    		       0.0, 0.0, 0.0, 1.0);
}

SJCMatrix44
SJCMatrix44::Translation(const SJCVector3d& t)
{
    return SJCMatrix44(1.0, 0.0, 0.0, t.x(),
    		       0.0, 1.0, 0.0, t.y(),
    		       0.0, 0.0, 1.0, t.z(),
    		       0.0, 0.0, 0.0, 1.0);
}

SJCMatrix44
SJCMatrix44::Translation(const SJCVector3f& t)
{
    return SJCMatrix44(1.0, 0.0, 0.0, t.x(),
    		       0.0, 1.0, 0.0, t.y(),
    		       0.0, 0.0, 1.0, t.z(),
    		       0.0, 0.0, 0.0, 1.0);
}


std::ostream& operator<<(std::ostream& o, const SJCMatrix44& m) {
    o << "[";
    for ( int i = 0 ; i < 4 ; i++ ) {
    o << "[";
    for ( int j = 0 ; j < 4 ; j++ ) {
        o << m.data[i][j];
        if ( j < 3 )
        o << ",";
    }
    o << "]";
    if ( i < 3 )
        o << ",";
    }
    return o;
}



