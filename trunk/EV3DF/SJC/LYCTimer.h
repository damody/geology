/************************************************************************
     File:        api.h

     Author:
                  Matt Pharr and Greg Humphreys
     Modifier:
                  Shaohua Fan, shaohua@cs.wisc.edu
                  Yu-Chi Lai, yu-chi@cs.wisc.edu

     Author Comment:
                  pbrt source code Copyright(c) 1998-2005 Matt Pharr and
                  Greg Humphreys

                  All Rights Reserved.
                  For educational use only; commercial use expressly forbidden.
                  NO WARRANTY, express or implied, for this software.
                 (See file License.txt for complete license)


     Comment:
                 Common function used for parsing data into the scene

     Functions:
                  1. Li : to get the radiance for single ray
                  2. RequestSamples: currently only request samples for direct
                     lighting
                  3. Preprocess: the main operation for this integrator
                  4. estimateImageAbsIllum: Estimate the absolute solution

 Last update: 10/11/04

 ************************************************************************/


#ifndef LYC_TIMER_H
#define LYC_TIMER_H

#if defined ( WIN32 )
#  include <windows.h>
#elif defined (IRIX) || defined (IRIX64)
#  include <stddef.h>
#  include <fcntl.h>
#  include <sys/time.h>
#  include <sys/types.h>
#  include <sys/mman.h>
#  include <sys/syssgi.h>
#  include <sys/errno.h>
#  include <unistd.h>
#else
#  include <sys/time.h>
#endif

// Timer Declarations
class LYCTimer {
 public:
  // Public LYCTimer Methods
  LYCTimer();
  ~LYCTimer();
  
  // Start the timer
  void Start();

  // Stop the timer
  void Stop();

  // Reset the timer
  void Reset();
  

  // Get the current time
  double Time();
 private:
  // Private LYCTimer Data
  double time0;
  double elapsed;
  bool   running;

  double GetTime();
#if defined( IRIX ) || defined( IRIX64 )
  // Private IRIX LYCTimer Data
  int fd;
  unsigned long long counter64;
  unsigned int counter32;
  unsigned int cycleval;
  
  typedef unsigned long long iotimer64_t;
  typedef unsigned int iotimer32_t;
  volatile iotimer64_t *iotimer_addr64;
  volatile iotimer32_t *iotimer_addr32;
  
  void *unmapLocation;
  int unmapSize;
#elif defined( WIN32 )
  // Private Windows LYCTimer Data
  LARGE_INTEGER performance_counter, performance_frequency;
  double one_over_frequency;
#else
  // Private UNIX LYCTimer Data
  struct timeval timeofday;
#endif

};
#endif
