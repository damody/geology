
/*
 * $Header: /p/graphics/CVS/yu-chi/SJCException.h,v 1.1.1.1 2006/04/25 20:21:39 yu-chi Exp $
 *
 * $Log: SJCException.h,v $
 * Revision 1.1.1.1  2006/04/25 20:21:39  yu-chi
 *
 *
 * Revision 1.1.1.1  2005/08/29 20:24:14  schenney
 * New libSJC source repository.
 *
 */

#ifndef _SJCEXCEPTION_H_
#define _SJCEXCEPTION_H_

#include "SJC.h"
#include <iostream>

class SJCDLL SJCException {
  private:
    char    *message;	// Just contains a message

    static char*    DEFAULT_MESSAGE;

  public:
    SJCException(const char *m = 0);
    ~SJCException();

    // Copy operator
    SJCException&	operator=(const SJCException&);

    // Return the error message string associated with the exception.
    const char* Message(void) { return message; };

    friend std::ostream& operator<<(std::ostream&o, const SJCException &e);
};


#endif

