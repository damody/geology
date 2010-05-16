/************************************************************************
     Main File:

     File:         SJCScreenPoint.h

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

#ifndef _SJCSCREENPOINT
#define _SJCSCREENPOINT

#include <SJC/SJC.h>

#include <iostream>

class SJCScreenPoint  
{

 public:
  int x;      // horizontal screen pos
  int y;      // vertical screen pos
 
  // Default constructor.
  SJCScreenPoint(void) {};
  
  // Constructor to initialize the points componenets.  
  SJCScreenPoint(int initX, int initY) : x(initX), y(initY) {};
  
  // Overloaded assignment operator.  
  const SJCScreenPoint& operator =(const SJCScreenPoint& point);
  
  // Overloaded addition operator.  Add the individual components of the points
  SJCScreenPoint operator +(const SJCScreenPoint& point) const;

  // Overloaded addition assignment operator.  Add the individual components
  // of the points and store the result in this instance.
  const SJCScreenPoint& operator +=(const SJCScreenPoint& point);

  // Overloaded subtraction operator.  Subtract the components of the given
  // point from this one and return the resulting point.
  SJCScreenPoint operator -(const SJCScreenPoint& point) const;

  // Overloaded subtraction assignment operator.  Subtract the components of
  // the given point from this one and store the result in this instance.
  const SJCScreenPoint& operator -=(const SJCScreenPoint& point);

  // Overloaded multiplication operator.  Multiple screen point by scalar.
  SJCScreenPoint operator *(float scalar) const;
  
  // Overloaded multiplication assignment operator.  Multiple this screen 
  // point by scalar.
  const SJCScreenPoint& operator *=(float scalar);

  // friends
  // Overloaded stream input operator.
  friend std::istream& operator >>(std::istream& in, 
					  SJCScreenPoint& point);
  // Overloaded stream output operator.
  friend std::ostream& operator <<(std::ostream& out, 
					  const SJCScreenPoint& point);
};

#endif // _CSCREENPOINT
