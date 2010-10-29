/*
** $Header: /p/graphics/CVS/yu-chi/SJCException.cpp,v 1.1.1.1 2006/04/25 20:21:47 yu-chi Exp $
**
** (c) 2001-2005 Stephen Chenney
**
** $Log: SJCException.cpp,v $
** Revision 1.1.1.1  2006/04/25 20:21:47  yu-chi
**
**
** Revision 1.1.1.1  2005/08/29 20:24:14  schenney
** New libSJC source repository.
**
**/

#include "SJCException.h"
#include <string.h>

char*	SJCException::DEFAULT_MESSAGE = "SJCException";


// Constructor stores the message.
SJCException::SJCException(const char *m)
{
    if ( m )
    {
	message = new char[strlen(m) + 4];
	strcpy(message, m);
    }
    else
	message = DEFAULT_MESSAGE;
}


SJCException::~SJCException()
{
    if ( message != DEFAULT_MESSAGE )
	delete message;
};


// Copy operator. Allocate new space and copy the message.
SJCException&
SJCException::operator=(const SJCException& e)
{
    if ( this != &e )
    {
	if ( message != DEFAULT_MESSAGE )
	    delete message;

	if ( e.message == DEFAULT_MESSAGE )
	    message = DEFAULT_MESSAGE;
	else
	{
	    message = new char[strlen(e.message)+4];
	    strcpy(message, e.message);
	}
    }

    return *this;
}



std::ostream&
operator<<(std::ostream&o, const SJCException &e)
{
    o << e.message;
    return o;
}


