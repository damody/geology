/************************************************************************
     Main File:

     File:        LYCProgressReporter.h

     Author:
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
 
     Comment:    
     Constructors:
                  1. 0 : the default contructor
                  3. 1 : copy contructor
                  4. 1 : Set up from the scalar field

     Functions:
                 1. = : Assign operator which copy the parameter of random
                 2. (): Get the value of the scalar field
                 3. VX: get the reference of X component of i, j
                 4. VY: get the reference of Y component of i, j
                 5. MakeDivergenceFree: make the velocity field divergence free
                 6. MinX, MinY, MaxX, MaxY: get the maximum and minimum value
                    of X, y
                 7. NumX, NumY: get the number of sample points in X, Y
                 8. >>: output in the ascii form
 ************************************************************************/

#ifndef _LYC_PROGRESS_REPORTER_H_
#define _LYC_PROGRESS_REPORTER_H_

#include <LYCTimer.h>

//*****************************************************************************
//
// * Progress reporter
//
//*****************************************************************************
class LYCProgressReporter 
{
 public:
  // Constructor
  LYCProgressReporter(int totalWork, const string &title,  int barLength=58);

  // Destructor
  ~LYCProgressReporter(void);

  // Update the count
  void Update(int num = 1) const;

  // Finish the process
  void Done(void) const;

 private:
  const int         m_iTotalPlusses;    // Total number of pluses print out
  float             m_dFrequency;       // Update frequency, to add one plus
  mutable float     m_dCount;
  mutable int       m_iPlussesPrinted;  // How many pluses for print out

  mutable LYCTimer *m_pTimer;           // Measure the time
  FILE             *m_pOutFile;         // File handler to print out the 
                                        // progress
  char             *m_pBuf;             // Progress space
  mutable char     *m_pCurSpace;        // Space for the current progress
};


#endif
