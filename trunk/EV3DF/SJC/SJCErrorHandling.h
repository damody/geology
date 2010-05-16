/************************************************************************
     Main File:   


     File:        SJCErrorHandling.h

     Author:    
                  Steven Chenney, schenney@cs.wisc.edu
                  Yu-Chi Lai, yu-chi@cs.wisc.edu


     Comment:     ErrorHandling functions include warning, error, assert

	
     Compiler:    g++


     Platform:    Linux
*************************************************************************/

#ifndef _ERROR_HANDLING
#define _ERROR_HANDLING


#include <string>
#include <stdexcept>
#include <iostream>
#include <sstream>

//****************************************************************************
//
// Function declarations
//****************************************************************************
namespace NErrorHandling
{
  //*********************************************************************
  //
  // Causes error message to be displayed if conditional fails in debug 
  // mode.  Program is then halted and user is given a chance to start the 
  // debugger.
  //*********************************************************************
  SJCDLL void AssertFunction(bool bConditional, const char* sMessage, 
		      const char* sMethod, const char* sFile, int line);
  
  //**************************************************************************
  //
  // Causes error message to be displayed if conditional fails.  Program is
  // then halted and user is given a chance to start the debugger.
  //*************************************************************************
  SJCDLL void VerifyFunction(bool bConditional, const char* sMessage, 
		      const char* sMethod, const char* sFile, int line);

  //*************************************************************************
  //
  // Causes error message to be displayed.  If _HALT_ON_WARNING is defined, 
  // program is halted and user is given a chance to start the debugger.
  //**************************************************************************
  SJCDLL void WarningFunction(const std::string& sMessage, const char* sMethod,
		       const char* sFile, int line);

  //**************************************************************************
  //
  // Causes error message to be displayed.  Program is then halted and user
  // is given a chance to start the debugger.
  //***************************************************************************
  SJCDLL void ErrorFunction(const std::string& sMessage, const char* sMethod, 
		     const char* sFile, int line);
}// NErrorHandling


// assert macro definitions
//#ifdef _VISUAL_STUDIO_6
//  #define __FUNCTION__ "Not Available"
//#endif

#ifdef _DEBUG
   #define _USE_ASSERT
#endif

#ifdef _USE_ASSERT
  #define SJCAssert(bConditional, sMessage) NErrorHandling::AssertFunction(bConditional, sMessage, __FUNCTION__, __FILE__, __LINE__)
#else
  #define SJCAssert(bConditional, sMessage)
#endif

#define SJCVerify(bConditional, sMessage) NErrorHandling::VerifyFunction(bConditional, sMessage, __FUNCTION__, __FILE__, __LINE__)
#define SJCWarning(sMessage) NErrorHandling::WarningFunction(sMessage, __FUNCTION__, __FILE__, __LINE__)
#define SJCError(sMessage) NErrorHandling::ErrorFunction(sMessage, __FUNCTION__, __FILE__, __LINE__)


// ToDo macro definitions
#ifdef _VISUAL_STUDIO
  // display todos in vc++ build window, gcc will just ignore them, 
  // unless you do a post build process to find them that is
  #define _TODO_QUOTE_QUOTE(x) # x
  #define _TODO_QUOTE(x) _TODO_QUOTE_QUOTE(x)
  #define ToDo(x) message(__FILE__"(" _TODO_QUOTE(__LINE__) "): ToDo: " #x"\n") 
#endif



//*************************************************************************
//
// * For debug exception information and it is from Tom bronet's code
//
//*************************************************************************
class LYCDebugException : public std::runtime_error {
 public:
  LYCDebugException(const std::string& msg) : std::runtime_error(msg) {
    // Put a breakpoint here to see where the exception originated from
  }
};

namespace LYCDebug {
  static std::ostringstream streamToStringConverter;
};

//*************************************************************************
//
// The STREAM_TO_STRING macro converts a set of stream operations 
// into a string, e.g STREAM_TO_STRING("Hello" << ", " << str
//*************************************************************************
#define STREAM_TO_STRING(arg) static_cast<std::ostringstream&>(	LYCDebug::streamToStringConverter.str(""), LYCDebug::streamToStringConverter << arg).str()

#define STREAM_TO_CHSTR(arg) STREAM_TO_STRING(arg).c_str()

#define LYCThrow(arg) throw LYCDebugException(STREAM_TO_STRING(arg))



#endif // _ERROR_HANDLING
