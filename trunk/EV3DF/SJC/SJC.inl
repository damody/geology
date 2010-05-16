/************************************************************************
     File:        SJC.inl


     Author:     
	          Yu-Chi Lai, yu-chi@cs.wisc.edu
       	          Stephen Chenney, schenney@cs.wisc.edu


     Comment:     The common definitions for entire project

     Functions:   
	          1. SJCMin(a, b): return the minimum between a, b
		  2. SJCMax(a, b): return the maximum
		  3. SJCRadianToDegrees(r): return degrees from radian
		  4. SJCDegreesToRadians(deg): return radian from deg
		  5. SJCAbs(a): return absoluate value of a
		  6. WithinEpsilon(a, b): whether a and b close enough
     Compiler:    g++


     Platform:    Linux
*************************************************************************/
 

//***************************************************************************
//
// Global functions
//
//***************************************************************************

//***************************************************************************
//
//  * Get the minimum of two values.  Note:  operator < must be defined for
//    the given type.
//=========================================================================
template<class TType> inline TType SJCMin(TType valueA, TType valueB)
//=========================================================================
{
  return ((valueA < valueB) ? valueA : valueB);
}// SJCMin


//**************************************************************************
//
//  * Get the maximum of two values.  Note:  operator < must be defined for
//    the given type.
//==========================================================================
template<class TType> inline TType SJCMax(TType valueA, TType valueB)
//===========================================================================
{
  return ((valueA < valueB) ? valueB : valueA);
}// SJCMax

//***************************************************************************
//
// * Convert radians to degrees.
//===========================================================================
template<class TType> inline TType SJCRadianToDegrees(TType angle)
//===========================================================================
{
  return angle * 180.f / M_PI;
}// SJCRadianToDegrees

//***************************************************************************
//
// * Convert degress to radians.
//===========================================================================
template<class TType> inline TType SJCDegreesToRadians(TType angle)
//============================================================================
{
  return angle * M_PI / 180;
}// SJCDegreesToRadians


//****************************************************************************
//
// * Calculate the absolute value of the given item.
//============================================================================
template<class TType> inline TType SJCAbs(TType value)
//============================================================================
{
  return (value >= 0) ? value : -value;
}// SJCAbs


//***************************************************************************
//
//      Do the two given values differ by less than epsilon.
//============================================================================
template<class TTypeA, class TTypeB> inline bool WithinEpsilon(TTypeA a,
                                                               TTypeB b)
//===========================================================================
{
  return SJCAbs(a - b) < SJC_EPSILON;
}// WithinEpsilon
