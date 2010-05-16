/************************************************************************
     Main File:

     File:        STLExtensions.h

     Author:      
                  Eric McDaniel, chat@cs.wisc.edu
 
     Comment:     The extension of standard container
	
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include <string>

// extenstions to the STL	
namespace std
{

  //**************************************************************************
  //
  // * Functor to delete an object.
  //==========================================================================
  template<class TType> struct FDelete:public std::unary_function<TType, void>
  //==========================================================================
  {
    inline void operator ()(TType object) {
      delete object;
    }// operator ()
  };// FDelete
  
  //**************************************************************************
  //
  // * Templated find method that takes a comparison functor.  STL does not 
  //   supply this.
  //==========================================================================
  template<class TType, class TIteratorType, class FComparator>
  inline TIteratorType find(TIteratorType iBegin, TIteratorType iEnd, 
                            TType value, FComparator fCompare)
  //==========================================================================
  {
    for (;iBegin != iEnd; ++iBegin) {
      if (fCompare(value, *iBegin))
	break;
    }// for

    return iBegin;
  }// find

}// std
