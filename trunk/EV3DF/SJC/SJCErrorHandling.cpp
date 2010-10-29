/************************************************************************
     Main File:


     File:        ErrorHandling.h


     Author:    
                  Steven Chenney, schenney@cs.wisc.edu
                  Yu-Chi Lai, yu-chi@cs.wisc.edu


     Comment:     ErrorHandling functions include warning, error, assert
	

     Compiler:    g++


     Platform:    Linux
*************************************************************************/

#include "SJC.h"

#include "SJCErrorHandling.h"

#include <string>
#include <iostream>

// include the signal system on linux
#ifdef _GNU
   #include <unistd.h>
   #include <signal.h>
#endif

// Using the name without name space specification
using std::cerr;
using std::endl;

namespace NErrorHandling
{
  //***********************************************************************
  //
  // * Causes error message to be displayed if conditional fails in debug 
  //   mode.  Program is then halted and user is given a chance to start the 
  //   debugger.
  //==========================================================================
  SJCDLL void AssertFunction(bool bConditional, const char* sMessage,
		      const char* sMethod, const char* sFile, int line)
  //=========================================================================
  {
    if (!bConditional) {
      cerr  << std::endl
            << "** Assertion failed *****************************************"
	    << std::endl
            << sMessage << std::endl 
            << "Method:  " << sMethod << std::endl
            << "File:  " << sFile << std::endl
            << "Line:  " << line << std::endl
            << "*************************************************************" 
	    << std::endl
            << std::endl;

      // give the user a chance to use the debugger
      #ifdef MS_WINDOWS
	assert(0);
      #else
        kill(getpid(), SIGABRT);
      #endif
    }// if
  }// AssertFunction

  //************************************************************************
  //
  // * Causes error message to be displayed if conditional fails.  Program is
  //   then halted and user is given a chance to start the debugger.
  //===========================================================================
  SJCDLL void VerifyFunction(bool bConditional, const char* sMessage, 
		      const char* sMethod, const char* sFile, int line)
  //==========================================================================
  {
    if (!bConditional) {
      cerr  << std::endl
            << "** Verification failed **************************************" 
	    << std::endl
            << sMessage << std::endl 
            << "Method:  " << sMethod << std::endl
            << "File:  " << sFile << std::endl
            << "Line:  " << line << std::endl
            << "*************************************************************"
	    << std::endl
            << std::endl;

      // give the user a chance to use the debugger
      #ifdef MS_WINDOWS
         assert(0);
      #else
         kill(getpid(), SIGABRT);
      #endif
    }// if
  }// VerifyFunction

  //***************************************************************
  //
  // * Causes error message to be displayed.  If _HALT_ON_WARNING is defined, 
  //   program is halted and user is given a chance to start the debugger.
  //==========================================================================
  SJCDLL void WarningFunction(const std::string& sMessage, const char* sMethod,
		       const char* sFile, int line)
  //=========================================================================
  {
    cerr   << std::endl
           << "** Warning ****************************************************"
	   << std::endl
           << sMessage << std::endl 
           << "Method:  " << sMethod << std::endl
           << "File:  " << sFile << std::endl
           << "Line:  " << line << std::endl
           << "**************************************************************"
	   << std::endl
           << std::endl;

    // if we're halting on warnings do so, otherwise just return
    #ifdef _HALT_ON_WARNING
      // give the user a chance to use the debugger
      #ifdef MS_WINDOWS
         assert(0);
      #else
         kill(getpid(), SIGABRT);
      #endif
    #endif
  }// WarningFunction


  //***************************************************************************
  //
  // * Causes error message to be displayed.  Program is then halted and user
  //   is given a chance to start the debugger.
  //
  //===========================================================================
  SJCDLL void ErrorFunction(const std::string& sMessage, const char* sMethod,
		     const char* sFile, int line)
  //===========================================================================
  {
    cerr  << std::endl
	  << "** Error ******************************************************"
	  << std::endl
	  << sMessage << std::endl 
	  << "Method:  " << sMethod << std::endl
	  << "File:  " << sFile << std::endl
	  << "Line:  " << line << std::endl 
	  << "***************************************************************"
	  << std::endl
          << std::endl;

    // give the user a chance to use the debugger
    #ifdef MS_WINDOWS
      assert(0);
    #else
      kill(getpid(), SIGABRT);
    #endif
  }// ErrorFunction
}// NErrorHandling

