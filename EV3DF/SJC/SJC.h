/************************************************************************
     File:        SJC.h


     Author:     
                  Stephen Chenney, schenney@cs.wisc.edu
                  Yu-Chi Lai, yu-chi@cs.wisc.edu

  
     Comment:     The common definitions for entire project

	
     Compiler:    g++


     Platform:    Linux
*************************************************************************/
#ifndef _SJC_H
#define _SJC_H

#define SJC_LIB 1

#ifdef _WIN32
#	define MS_WINDOWS
#	define WIN32_LEAN_AND_MEAN
#	include <windows.h>
#pragma warning(disable:4127)
#endif

#if defined __GNUC__
  #define _GNU
#endif


#ifdef WIN32
// 4996 is warning for the  sprintf, 4244 is conversion from double to float
// 4305 is truncation from double to float
#	pragma warning (disable: 4267 4251 4065 4102 4996)
#	pragma warning( disable: 4190 4244 4305)

#	ifdef SJC_SOURCE
#		define SJCDLL __declspec(dllexport)
#   elif defined SJC_LIB
#		define SJCDLL
#	else
#		define SJCDLL __declspec(dllimport)
#	endif
#	define DLLEXPORT __declspec(dllexport)
#else
#	define SJCDLL
#	define DLLEXPORT
#endif



/*
** Things that might need to be changed on a platform independent basis.
*/

/* The size of various types. */
#if defined _VISUAL_STUDIO || defined _WIN32
    typedef     unsigned char			byte; // change to unsigned byte because of redefinition in rpcndr.h
//    typedef     char       	    byte;
    typedef     unsigned char       ubyte;
    typedef     unsigned int        uint;
    typedef     signed char         int8;
    typedef     unsigned char       uint8;
    typedef     short               int16;
    typedef     unsigned short      uint16;
    typedef     int                 int32;
    typedef     unsigned int        uint32;
    typedef     __int64             int64;
    typedef     unsigned __int64    uint64;
    typedef     float               float32;
    typedef     double              float64;
#elif defined(_GNU)
    typedef     char       	    byte;
    typedef     unsigned char       ubyte;
    typedef     unsigned int        uint;
    typedef     signed char         int8;
    typedef     unsigned char       uint8;
    typedef     short               int16;
    typedef     unsigned short      uint16;
    typedef     int                 int32;
    typedef     unsigned int        uint32;
    typedef     long long           int64;
    typedef     unsigned long long  uint64;
    typedef     float               float32;
    typedef     double              float64;
#else
  #error Type definitions are not defined for this platform.
#endif 
 


// Some platform specific pragmas etc.
#ifdef _WIN32
  #ifndef _DEBUG
    #define _RELEASE
  #endif

  #ifdef _VISUAL_STUDIO_7
    #define _VISUAL_STUDIO

    #pragma warning(disable : 4100) // disable unused parameter warning
    #pragma warning(disable : 4127) // disable conditional is constant warning

    // macros to disable/enable conversion warnings, must be placed 
    // outside methods/classes
    #define MPushWarningState warning(push)
    #define MPopWarningState warning(pop)
    #define MDisableConversionWarnings  warning(disable : 4267; disable : 4244; disable : 4311; disable : 4312; disable : 4511; disable : 4512)
    #define MEnableConversionWarnings  warning(default : 4267; default : 4244; default : 4311; default : 4312; default : 4511; default : 4512)

    // disable those conversion warnings
    #pragma MDisableConversionWarnings
  #else 
    #ifndef _VISUAL_STUDIO_6
      #define _VISUAL_STUDIO_6
    #endif
    #define _VISUAL_STUDIO
    #define for if (false) {} else for  // fix for loop scoping problem in vc++,
                                        // this has no performance cost 
                                        // for a release build
    #pragma warning(disable : 4786)     // fix the "mangled name exceeds 
                                        // 255 characters" bug in vc
  #endif
#elif defined __GNUC__
  #ifdef NDEBUG
    #define _RELEASE
  #else
    #define _DEBUG
  #endif
#else
  #error Unknown compiler and platform.
#endif

//****************************************************************************
//
// Global includes
//
//****************************************************************************

#include <float.h> // DBL_EPSILON
#include <assert.h>

#//****************************************************************************
//
// Global Definition
//
//****************************************************************************

  #define M_2PI           6.28318530717958
  #define M_2PI           6.28318530717958
#ifndef M_PI
# define M_PI 3.1415926535897932384626433832795
#endif
  #define M_PI_2          1.57079632679489661923
  #define M_PI_4          0.78539816339744830962
  #define M_1_PI          0.31830988618379067154
  #define M_2_PI          0.63661977236758134308
  #define M_SQRT2         1.41421356237309504880
  #define M_SQRT1_2       0.70710678118654752440
  #define M_SQRT3         1.732050807569


#ifndef SJC_EPSILON      
  #define SJC_EPSILON       0.00001
  #define SJC_EPSILON2      1e-20
  #define SJC_INFINITE      9999999999999.99
  #define SJC_DEG_TO_RAD    0.01745329252
  #define SJC_RAD_TO_DEG    57.2957795131
  #define SJC_INVALIDHANDLE 0
#endif

#include "SJCErrorHandling.h"   // error handling
#include "SJC.inl"          // Global functions and templates
#include "SJCSTLExtensions.h"

//****************************************************************************
//
// * Count the number this object has been pointed
//
//****************************************************************************
class TCountPointTo {
 public:
  // Constructor
  TCountPointTo(void) { m_uNumPointTo = 0; }
 public:
  uint m_uNumPointTo; // Total number of other index to me

 private:
  // Copy constructor 
  TCountPointTo(const TCountPointTo &a);

  // Assign operator
  TCountPointTo &operator=(const TCountPointTo &);
};


//****************************************************************************
//
// Assume that the pointer send in must be a TCountPointTo
//
//****************************************************************************
template <class T> class TPointer {
 public:
  // Constructor
  TPointer(T *p = NULL) {
    m_pPointer = p;
    if (m_pPointer) ++m_pPointer->m_uNumPointTo;
  }

  // Copy contructor
  TPointer(const TPointer<T> &r) {
    m_pPointer = r.m_pPointer;
    if (m_pPointer) 
      ++m_pPointer->m_uNumPointTo;;
  }
  // Destructor
  ~TPointer(void) {
    if (m_pPointer && --m_pPointer->m_uNumPointTo == 0)
      delete m_pPointer;
  }

  // Assign operator, weird
  TPointer &operator=(const TPointer<T> &r) {
    if (r.m_pPointer) 
      r.m_pPointer->m_uNumPointTo++;
    if (m_pPointer && --m_pPointer->m_uNumPointTo == 0) 
      delete m_pPointer;
    m_pPointer = r.m_pPointer;
    return *this;
  }

  // Assign to a new pointer
  TPointer &operator=(T *p) {
    if (p) 
      p->m_uNumPointTo++;

    if (m_pPointer && --m_pPointer->m_uNumPointTo == 0) 
      delete m_pPointer;
    m_pPointer = p;
    return *this;
  }

  // Point to operator
  T *operator->() { return m_pPointer; }

  // Point to opertor
  const T *operator->() const { return m_pPointer; }

  // Whether the pointer exists or not
  operator bool() const { return m_pPointer != NULL; }

  // Compare two pointer's address
  bool operator<(const TPointer<T> &t2) const {
    return m_pPointer < t2.m_pPointer;
  }

 private:
  T *m_pPointer;                    // Store the pointer
};

#endif // _GLOBALS

