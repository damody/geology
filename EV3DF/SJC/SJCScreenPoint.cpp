/************************************************************************
     Main File:

     File:         ScreenPoint.h

     Author:     
                   Eric McDaniel, chate@cs.wisc.edu
 
     Comment:      This class implements a structure to represent a screen 
                   position. Overloaded stream input/output operators are
                   provided.

     Constructors:
                   
     Functions: 
     
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include <SJCScreenPoint.h>

//***************************************************************************
//
// * Overloaded assignment operator.  
//===========================================================================
const SJCScreenPoint& SJCScreenPoint::
operator =(const SJCScreenPoint& point)
//===========================================================================
{
  x = point.x;
  y = point.y;
  return *this;
}// operarator =


//***************************************************************************
//
//  * Overloaded addition operator.  Add the individual components of the 
//    points.
//===========================================================================
SJCScreenPoint SJCScreenPoint::
operator +(const SJCScreenPoint& point) const
//===========================================================================
{
  return SJCScreenPoint(x + point.x, y + point.y);
}// operator +


//*****************************************************************************
//
//  * Overloaded addition assignment operator.  Add the individual components
//    of the points and store the result in this instance.
//===========================================================================
const SJCScreenPoint& SJCScreenPoint::
operator +=(const SJCScreenPoint& point)
//===========================================================================
{
  x += point.x;
  y += point.y;
  return *this;
}// operator +=


//*****************************************************************************
//
// * Overloaded subtraction operator.  Subtract the components of the given
//   point from this one and return the resulting point.
//===========================================================================
SJCScreenPoint SJCScreenPoint::
operator -(const SJCScreenPoint& point) const
//===========================================================================
{
  return SJCScreenPoint(x - point.x, y - point.y);
}// operator -


//*****************************************************************************
//
// * Overloaded subtraction assignment operator.  Subtract the components of
//   the given point from this one and store the result in this instance.
//===========================================================================
const SJCScreenPoint& SJCScreenPoint::
operator -=(const SJCScreenPoint& point)
//===========================================================================
{
  x -= point.x;
  y -= point.y;
  return *this;
}// operator -=


//*****************************************************************************
//
// * Overloaded multiplication operator.  Multiple screen point by scalar.
//===========================================================================
SJCScreenPoint SJCScreenPoint::
operator *(float scalar) const
//===========================================================================
{
  return SJCScreenPoint(static_cast<int>(x * scalar), 
		      static_cast<int>(y * scalar));
}// operator *


//***************************************************************************
//
// * Overloaded multiplication assignment operator.  Multiple this screen 
//   point by scalar.
//===========================================================================
const SJCScreenPoint& SJCScreenPoint::
operator *=(float scalar)
//===========================================================================
{
  x = static_cast<int>(static_cast<float>(x) * scalar);
  y = static_cast<int>(static_cast<float>(y) * scalar);
  return *this;
}// operator *=


// friends 

//*****************************************************************************
//
// * Overloaded stream input operator.
//===========================================================================
std::istream& operator >>(std::istream& in, SJCScreenPoint& point)
//===========================================================================
{
  in >> point.x >> point.y;
  return in;
}// operator >>


//****************************************************************************
//
// * Overloaded stream output operator.
//===========================================================================
std::ostream& operator <<(std::ostream& out, const SJCScreenPoint& point)
//===========================================================================
{
  out << point.x << " " << point.y;
  return out;
}// operator <<
