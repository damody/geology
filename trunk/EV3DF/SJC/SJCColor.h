/************************************************************************
     Main File:

     File:        SJCColor.h

     Author:      
                  Eric McDaniel, chat@cs.wisc.edu

     Written:     2002 summer
 
     Comment:     Color operation

     Constructor
                  1. (): Default constructor
                  2. (common): Set up all value at given value
                  3. (color): Copy constructor.  
                  4. (r, g, b, a): Initialize channels to the given values.
                  5. (byte r, g, b, a): Initialize channels to the given
                      values.  Covert from  [0..255] to [0..1].
     Functions:
                  1. =, +, +=, -, -=, * scalar, *=, 
                     * color: element to element, *=, / scalar, /=,
                     / color, /=, ==, !=, 
                  2. AsArray: Access the vector components as an array rgba.
                  3. Clamp: Clamp all channels to lie in [0..1].
                  4. Normalize: Normalize so max channel is 1.
                  5. RedByte: Get red channel as byte. (Clamp if neccessary)
                  6. GreenByte: Get green channel as byte.(Clamp if neccessary)
                  7. BlueByte: Get blue channel as byte.(Clamp if neccessary)
                  8. AlphaByte: Get alpha channel as byte.(Clamp if neccessary)

	
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#ifndef _SJCCOLOR_H
#define _SJCCOLOR_H

#include <SJC/SJC.h>

#include <math.h>

#include <Fl/gl.h>
#include <GL/glu.h>
 
class SJCColor
{
  // interface

 public:
  // methods
  // Default constructor.
  SJCColor(void);
  
  // Constructor.  Initialize all channel to the given value.
  SJCColor(const float common);

  // Copy constructor.  
  SJCColor(const SJCColor& color);

  // Constructor.  Initialize channels to the given values.
  SJCColor(const float redIn, const float greenIn, const float blueIn, 
	   const float alphaIn = 1.f);

  // Constructor.  Initialize channels to the given values.  Covert from 
  // [0..255] to [0..1].
  SJCColor(const ubyte redIn, const ubyte greenIn, const ubyte blueIn, 
	   const ubyte alphaIn = 255);


  // Destructor.
  ~SJCColor(void);

  // Overloaded assignment operator.    SJCColor = SJCColor
  const SJCColor& operator =(const SJCColor& color);

  // Overloaded addition operator.    SJCColor + SJCColor
  SJCColor operator +(const SJCColor& color) const;

  // Overloaded addition and assignment operator.   SJCColor += SJCColor
  const SJCColor& operator +=(const SJCColor& color);

  // Overloaded subtraction operator.    SJCColor - SJCColor
  SJCColor operator -(const SJCColor& color) const;

  // Overloaded subtraction and assignment operator.   SJCColor -= SJCColor
  const SJCColor& operator -=(const SJCColor& color);

  // Overloaded multiplication operator.    SJCColor * float
  SJCColor operator *(float num) const;

  // Overloaded multiplication and assignment operator.   SJCColor *= float
  const SJCColor& operator *=(float num);

  // Overloaded multiplication operator.    Piecewise multiplication.
  // SJCColor * SJCColor
  SJCColor operator *(const SJCColor& color) const;

  // Overloaded multiplication and assignment operator.   Piecewise 
  // multiplication. SJCColor *= SJCColor
  const SJCColor& operator *=(const SJCColor& color);

  // Overloaded division operator.    SJCColor / float
  SJCColor operator /(float num) const;

  // Overloaded division and assignment operator.   SJCColor /= float
  const SJCColor& operator /=(float num);

  // Overloaded division operator.   Piecewise division.  SJCColor / SJCColor
  SJCColor operator /(const SJCColor& color) const;

  // Overloaded division and assignment operator.   Piecewise division.  
  // SJCColor /= SJCColor
  const SJCColor& operator /=(const SJCColor& color);

  // Overloaded equality operator.
  bool operator ==(const SJCColor& color) const;

  // Overloaded inequality operator.
  bool operator !=(const SJCColor& color) const;

  // Access the vector components as an array rgba.
  const float* AsArray(void) const;

  // Clamp all channels to lie in [0..1].
  const SJCColor& Clamp(void);

  // Normalize so max channel is 1.
  const SJCColor& Normalize(void);

  // Get red channel as byte.  (Clamp if neccessary)
  byte RedByte(void) const;

  // Get green channel as byte.  (Clamp if neccessary)
  byte GreenByte(void) const;

  // Get blue channel as byte.  (Clamp if neccessary)
  byte BlueByte(void) const;

  // Get alpha channel as byte.  (Clamp if neccessary)
  byte AlphaByte(void) const;

  // Set the GL color
  void SetGLColor(void) const { glColor4f((float)red, (float)green, 
					  (float)blue, (float)alpha); }

 public:
  //members
  float    red;          // Component of red
  float    green;        // Component of green
  float    blue;         // Component of blue
  float    alpha;        // Component of alpha
};

#endif 
