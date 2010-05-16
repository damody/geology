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

#include <LYCProgressReporter.h>

//****************************************************************************
//
// * Constructor
//============================================================================
LYCProgressReporter::
LYCProgressReporter(int totalWork, const string &title, int bar_length)
  : m_iTotalPlusses(bar_length - title.size()) 
//============================================================================
{

  m_iPlussesPrinted = 0;
  m_dFrequency      = (float)totalWork / (float)m_iTotalPlusses;
  m_dCOunt          = frequency;
  m_pTimer          = new LYCTimer();
  m_pTimer->Start();

  m_pOutFile        = stdout;

  // Initialize progress string
  m_pBuf = new char[title.size() + m_iTotalPlusses + 64];
  sprintf(m_pBuf, "\r%s: [", title.c_str());

  m_pCurSpace = m_pBuf + strlen(m_pBuf);
  char *s = m_pCurSpace;

  for (int i = 0; i < m_iTotalPlusses; ++i)
    *s++ = ' ';
  *s++ = ']';
  *s++ = ' ';
  *s++ = '\0';

  // Print out the basic information
  fprintf(m_pOutFile, m_pBuf);
  fflush(m_pOutFile);
}

//****************************************************************************
//
// * Destructor
//============================================================================
LYCProgressReporter::
~LYCProgressReporter(void) 
//============================================================================
{
  delete[] m_pBuf; 
  delete m_pTimer; 
}


//****************************************************************************
//
// * Update the progress
//============================================================================
void LYCProgressReporter::
Update(int num) const 
//============================================================================
{
  m_dCOunt         -= num;
  bool updatedAny   = false;

  while (m_dCOunt <= 0) {
    m_dCOunt += m_dFrequency;
    if (m_iPlussesPrinted++ < m_iTotalPlusses)
      *m_pCurSpace++ = '+';
    updatedAny = true;
  }

  if (updatedAny) {
    fputs(m_pBuf, m_pOutFile);

    // Update elapsed time and estimated time to completion
    float percentDone  = (float)m_iPlussesPrinted / (float)m_iTotalPlusses;
    float seconds      = (float) m_pTimer->Time();
    float estRemaining = seconds / percentDone - seconds;

    if (percentDone == 1.f)
      fprintf(m_pOutFile, " (%.1fs)       ", seconds);
    else
      fprintf(m_pOutFile, " (%.1fs|%.1fs)  ", seconds, max(0.f, estRemaining));

    fflush(m_pOutFile);
  }

}

//****************************************************************************
//
// * Finish the progress
//============================================================================
void LYCProgressReporter::
Done(void) const 
//============================================================================
{
  while (m_iPlussesPrinted++ < m_iTotalPlusses)
    *m_pCurSpace++ = '+';

  fputs(m_pBuf, m_pOutFile);
  float seconds = (float) m_pTimer->Time();

  fprintf(m_pOutFile, " (%.1fs)       \n", seconds);
  fflush(m_pOutFile);
}
