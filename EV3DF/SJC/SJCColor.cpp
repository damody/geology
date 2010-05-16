/************************************************************************
     Main File:

     File:        SJCColor.h

     Author:      
                  Eric McDaniel, chat@cs.wisc.edu
 
     Comment:     Definitions of methods of the CColor class.  See header for 
                  documentation.
	
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include <SJCColor.h>

//*****************************************************************************
//
// * Default constructor.
//============================================================================
SJCColor::
SJCColor()
//============================================================================
{}// SJCColor


//*****************************************************************************
//
// * Constructor.  Initialize all channel to the given value.
//============================================================================
SJCColor::
SJCColor(const float common) 
  : red(common), green(common), blue(common), alpha(common)
//============================================================================
{}// SJCColor


//*****************************************************************************
//
// * Copy constructor.  
//============================================================================
SJCColor::
SJCColor(const SJCColor& color) 
  : red(color.red), green(color.green), blue(color.blue), alpha(color.alpha)
//============================================================================
{}// SJCColor


//*****************************************************************************
//
// * Constructor.  Initialize channels to the given values.
//============================================================================
SJCColor::
SJCColor(const float redIn, const float greenIn, const float blueIn, 
       const float alphaIn /* = 1.f */) 
  : red(redIn), green(greenIn), blue(blueIn), alpha(alphaIn)
//============================================================================
{}// SJCColor


//*****************************************************************************
//
// * Constructor.  Initialize channels to the given values.  Covert from 
//  [0..255] to [0..1].
//============================================================================
SJCColor::SJCColor(const ubyte redIn, const ubyte greenIn, 
		   const ubyte blueIn, const ubyte alphaIn /* = 255 */) 
  : red(redIn * .003921568627f), green(greenIn * .003921568627f), 
    blue(blueIn * .003921568627f), alpha(alphaIn * .003921568627f)
//============================================================================
{}// SJCColor

//*****************************************************************************
//
// * Destructor.
//============================================================================
SJCColor::
~SJCColor()
//============================================================================
{}// ~SJCColor


//*****************************************************************************
//
// * Overloaded assignment operator.    SJCColor = SJCColor
//============================================================================
const SJCColor& SJCColor::
operator =(const SJCColor& color)
//============================================================================
{
  red = color.red;
  green = color.green;
  blue = color.blue;
  alpha = color.alpha;
  
  return *this;
}// operator

#pragma ToDo("SJCColor +- SJCColor is broken.  Doesn't handle alpha correctly.")

//*****************************************************************************
//
// * Overloaded addition operator.    SJCColor + SJCColor
//============================================================================
SJCColor SJCColor::
operator +(const SJCColor& color) const
//============================================================================
{
   return SJCColor(red + color.red, green + color.green, blue + color.blue, 
		 alpha + color.alpha);
}// operator


//*****************************************************************************
//
// * Overloaded addition and assignment operator.   SJCColor += SJCColor
//============================================================================
const SJCColor& SJCColor::
operator +=(const SJCColor& color)
//============================================================================
{
  red += color.red;
  green += color.green;
  blue += color.blue;
  alpha += color.alpha;
  
  return *this;
}// operator


//*****************************************************************************
//
// * Overloaded subtraction operator.    SJCColor - SJCColor
//============================================================================
SJCColor SJCColor::
operator -(const SJCColor& color) const
//============================================================================
{
  return SJCColor(red - color.red, green - color.green, 
		blue - color.blue, alpha - color.alpha);
}// operator


//*****************************************************************************
//
// * Overloaded subtraction and assignment operator.   SJCColor -= SJCColor
//============================================================================
const SJCColor& SJCColor::
operator -=(const SJCColor& color)
//============================================================================
{
  red -= color.red;
  green -= color.green;
  blue -= color.blue;
  alpha -= color.alpha;
  
  return *this;
}// operator

//*****************************************************************************
//
// * Overloaded multiplication operator.    SJCColor * float
//============================================================================
SJCColor SJCColor::
operator *(float num) const
//============================================================================
{
  return SJCColor(red * num, green * num, blue * num, alpha * num);
}// operator

//*****************************************************************************
//
// * Overloaded multiplication and assignment operator.   SJCColor *= float
//============================================================================
const SJCColor& SJCColor::
operator *=(float num)
//============================================================================
{
  red *= num;
  green *= num;
  blue *= num;
  alpha *= num;
  
  return *this;
}// operator

//*****************************************************************************
//
// * Overloaded multiplication operator.    Piecewise multiplication.
//   SJCColor * SJCColor
//============================================================================
SJCColor SJCColor::
operator *(const SJCColor& color) const
//============================================================================
{
  return SJCColor(red * color.red, green * color.green, 
		blue * color.blue, alpha * color.alpha);
}// operator

//*****************************************************************************
//
// * Overloaded multiplication and assignment operator.   Piecewise 
//   multiplication. SJCColor *= SJCColor
//============================================================================
const SJCColor& SJCColor::operator *=(const SJCColor& color)
//============================================================================
{
  red *= color.red;
  green *= color.green;
  blue *= color.blue;
  alpha *= color.alpha;
  return *this;
}// operator


//*****************************************************************************
//
// * Overloaded division operator.    SJCColor / float
//============================================================================
SJCColor SJCColor::
operator /(float num) const
//============================================================================
{
  SJCAssert(SJCAbs(num) > SJC_EPSILON, "Attempt to divide be zero.");
  float invNum = 1 / num; 
  return SJCColor(red * invNum, green * invNum, blue * invNum, alpha * invNum);
}// operator


//*****************************************************************************
//
// * Overloaded division and assignment operator.   SJCColor /= float
//============================================================================
const SJCColor& SJCColor::
operator /=(float num)
//============================================================================
{
  SJCAssert(SJCAbs(num) > SJC_EPSILON, "Attempt to divide be zero.");
  float invNum = 1 / num; 
  
  red *= invNum;
  green *= invNum;
  blue *= invNum;
  alpha *= invNum;
  
  return *this;
}// operator

//*****************************************************************************
//
// * Overloaded division operator.   Piecewise division.  SJCColor / SJCColor
//============================================================================
SJCColor SJCColor::
operator /(const SJCColor& color) const
//============================================================================
{
  return SJCColor(red / color.red, green / color.green, 
		blue / color.blue, alpha / color.alpha);
}// operator


//*****************************************************************************
//
// * Overloaded division and assignment operator.   Piecewise division.  
//   SJCColor /= SJCColor
//============================================================================
const SJCColor& SJCColor::
operator /=(const SJCColor& color)
//============================================================================
{
  red /= color.red;
  green /= color.green;
  blue /= color.blue;
  alpha /= color.alpha;
  
  return *this;
}// operator


//*****************************************************************************
//
// * Overloaded equality operator.
//============================================================================
bool SJCColor::
operator ==(const SJCColor& color) const
//============================================================================
{
  return (fabs(red - color.red) < SJC_EPSILON) && 
         (fabs(green - color.green) < SJC_EPSILON) && 
         (fabs(blue - color.blue) < SJC_EPSILON) && 
         (fabs(alpha - color.alpha) < SJC_EPSILON);
}// operator


//*****************************************************************************
//
// * Overloaded inequality operator.
//============================================================================
bool SJCColor::
operator !=(const SJCColor& color) const
//============================================================================
{
  return !(*this == color);
}// operator

//*****************************************************************************
//
// * Access the vector components as an array rgb.
//============================================================================
const float* SJCColor::
AsArray() const
//============================================================================
{
  return &red;
}// AsArray


//*****************************************************************************
//
//      Clamp all channels to lie in [0..1].
//============================================================================
const SJCColor& SJCColor::
Clamp()
//============================================================================
{
  red = SJCMin(red, 1.f);
  green = SJCMin(green, 1.f);
  blue = SJCMin(blue, 1.f);
  alpha = SJCMin(alpha, 1.f);
  
   return *this;
}// Clamp


//*****************************************************************************
//
// * Normalize so max channel is 1.
//============================================================================
const SJCColor& SJCColor::
Normalize()
//============================================================================
{
  float scale = 1.f / SJCMax(red, SJCMax(green, SJCMax(blue, alpha)));
  red *= scale;
  green *= scale;
  blue *= scale;
  alpha *= scale;
  
  return *this;
}// Normalize


//*****************************************************************************
//
// * Get red channel as byte.  (Clamp if neccessary)
//============================================================================
byte SJCColor::
RedByte() const
//============================================================================
{
  return static_cast<byte>(SJCMax(0.f, SJCMin(1.f, red)) * 255);
}// RedByte


//*****************************************************************************
//
// * Get green channel as byte.  (Clamp if neccessary)
//============================================================================
byte SJCColor::
GreenByte() const
//============================================================================
{
  return static_cast<byte>(SJCMax(0.f, SJCMin(1.f, green)) * 255);
}// GreenByte


//*****************************************************************************
//
// * Get blue channel as byte.  (Clamp if neccessary)
//============================================================================
byte SJCColor::
BlueByte() const
//============================================================================
{
  return static_cast<byte>(SJCMax(0.f, SJCMin(1.f, blue)) * 255);
}// BlueByte


//*****************************************************************************
//
// * Get alpha channel as byte.  (Clamp if neccessary)
//============================================================================
byte SJCColor::
AlphaByte() const
//============================================================================
{
  return static_cast<byte>(SJCMax(0.f, SJCMin(1.f, alpha)) * 255);
}// AlphaByte

