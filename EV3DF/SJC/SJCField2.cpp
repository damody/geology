/************************************************************************
     Main File:

     File:        SJCField2.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
  
     Comment:     Base class for the field to handle the dimensional 
                  information
 
     Constructors:
                  1. 0 : default
                  2. 6 : constructor to set up all value by input parameters
                  3. 1 : Set up the data from input stream
                  4. 1 : copy contructor
                   
     Functions:  
                 1. = : Assign operator which copy the parameter of random
                 2. MinX, MinY, MaxX, MaxY: get the maximum and minimum value 
                    of X, y
                 3. DiffMaxX, DiffMaxY, DiffMinX, DiffMinY: get the max and
                    min value of X, Y
                 8. NumX, NumY: get the number of sample points in X, Y
                 9. Write: write out the data into stream in binary form
                10. >>: output in the ascii form

************************************************************************/

#include "SJCField2.h"


//****************************************************************************
//
// * Contructor to set up all value
//============================================================================
SJCField2d::
SJCField2d(const uint nx,        const uint ny,
	   const double dx,      const double dy,
	   const SJCBoundary bx, const SJCBoundary by)
//============================================================================
{
  m_uNX          = nx;
  m_uNY          = ny;

  m_dDX          = dx;
  m_dDY          = dy;

  m_dHalfDX      = dx * 0.5;
  m_dHalfDY      = dy * 0.5;

  m_dLengthX     = m_dDX * (double)m_uNX;
  m_dLengthY     = m_dDY * (double)m_uNY;
  
  m_VBoundary[0] = bx;
  m_VBoundary[1] = by;

}


//****************************************************************************
//
// * Get the scalar field from input
//============================================================================
SJCField2d::
SJCField2d(std::istream &f)
//============================================================================
{
  Read(f);
}



//****************************************************************************
//
// * Copy contructor
//============================================================================
SJCField2d::
SJCField2d(const SJCField2d &vf)
//============================================================================
{
  Assign(vf);
}

//****************************************************************************
//
// * Copy contructor
//============================================================================
void SJCField2d::
Assign(const SJCField2d &vf)
//============================================================================
{
  m_uNX          = vf.m_uNX;
  m_uNY          = vf.m_uNY;

  m_dDX          = vf.m_dDX;
  m_dDY          = vf.m_dDY;
  
  m_dHalfDX      = m_dDX * 0.5;
  m_dHalfDY      = m_dDY * 0.5;
  
  m_dLengthX     = m_dDX * (double)m_uNX;
  m_dLengthY     = m_dDY * (double)m_uNY;

  m_VBoundary[0] = vf.m_VBoundary[0];
  m_VBoundary[1] = vf.m_VBoundary[1];

}



//****************************************************************************
//
// * Destructor
//============================================================================
SJCField2d::
~SJCField2d(void)
//============================================================================
{
  Destroy();
}


//****************************************************************************
//
// * Assign operator
//============================================================================
SJCField2d& SJCField2d::
operator=(const SJCField2d &vf)
//============================================================================
{
  // Clear data
  Destroy();

  // Do the copy
  Assign(vf);

  return *this;
}


//****************************************************************************
//
// * Contructor to set up all value
//============================================================================
void SJCField2d::
Set(const uint nx,        const uint ny,
    const double dx,      const double dy,
    const SJCBoundary bx, const SJCBoundary by)
//============================================================================
{
  Destroy();
  m_uNX          = nx;
  m_uNY          = ny;

  m_dDX          = dx;
  m_dDY          = dy;

  m_dHalfDX      = dx * 0.5;
  m_dHalfDY      = dy * 0.5;

  m_dLengthX     = m_dDX * (double)m_uNX;
  m_dLengthY     = m_dDY * (double)m_uNY;
  
  m_VBoundary[0] = bx;
  m_VBoundary[1] = by;

}


//****************************************************************************
//
// * Get the minimum differential value in x direction
//============================================================================
double SJCField2d::
BoundMinX(void) const
//============================================================================
{
  switch ( m_VBoundary[0] )   {
    case BOUNDARY_NOWRAP:      // The outest layer is a wall
      return m_dDX;
    case BOUNDARY_NOWRAP_FREE: // Not a wall but a constant boundary
      return m_dDX;
    case BOUNDARY_WRAP: 
      return 0.0f;
  }// end of switch

  return m_dDX;
}

//****************************************************************************
//
// * Return the minimum differential value in y direction
//============================================================================
double SJCField2d::
BoundMinY(void) const
//============================================================================
{
  switch ( m_VBoundary[1] ) {
    case BOUNDARY_NOWRAP: // The outest layer is a wall
      return m_dDY;
    case BOUNDARY_NOWRAP_FREE: // Not a wall but a constant boundary
      return m_dDY;
    case BOUNDARY_WRAP: 
      return 0.0f;
  }

  return m_dDY;
}


//****************************************************************************
//
// * Return the differential maximum value in x direction
//============================================================================
double SJCField2d::
BoundMaxX(void) const
//============================================================================
{
  switch ( m_VBoundary[0] )    {
    case BOUNDARY_NOWRAP:  // The outest layer is a wall
      return m_dLengthX - m_dDX;
    case BOUNDARY_NOWRAP_FREE: // Not a wall but a constant boundary
      return m_dLengthX - m_dDX;
    case BOUNDARY_WRAP: 
      return m_dLengthX;
  }

  return m_dLengthX - m_dDX;

}

//****************************************************************************
//
// * Return the differential maximum value in y direction
//============================================================================
double SJCField2d::
BoundMaxY(void) const
//============================================================================
{
  switch ( m_VBoundary[1] )    {
    case BOUNDARY_NOWRAP: 
      return m_dLengthY - m_dDY;
    case BOUNDARY_NOWRAP_FREE: // Not a wall but a constant boundary
      return m_dLengthY - m_dDY;
    case BOUNDARY_WRAP: 
      return m_dLengthY;
  }
  return m_dLengthY - m_dDY;
}

//****************************************************************************
//
// * Clear the m_VData
//============================================================================
void SJCField2d::
Destroy(void)
//============================================================================
{
  // Release the data
  m_uNX      = m_uNY      = 0;
  m_dDX      = m_dDY      = 0.f;
  m_dHalfDX  = m_dHalfDY  = 0.f;
  m_dLengthX = m_dLengthY = 0.f;
  
}


//****************************************************************************
//
// * Output operator
//============================================================================
std::ostream&
operator<<(std::ostream &o, const SJCField2d &vf)
//============================================================================
{
  // Output NX, NY
  o << std::setw(4) << vf.m_uNX << " " << std::setw(4) << vf.m_uNY << " ";
  // Output DX, DY
  o << std::setw(8) << vf.m_dDX << " " << std::setw(8) << vf.m_dDY << " ";
  // Output Boundary
  o << vf.m_VBoundary[0] << " "<< vf.m_VBoundary[1] << " " << std::endl;

  return o;
}

//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
void
SJCField2d::Write(std::ostream &out)
//============================================================================
{
  // Output NX, NY
  out.write((char*)&m_uNX, sizeof(uint));
  out.write((char*)&m_uNY, sizeof(uint));
  // Output DX, DY
  out.write((char*)&m_dDX, sizeof(double));
  out.write((char*)&m_dDY, sizeof(double));
  // Output Boundary
  out.write((char*)&(m_VBoundary[0]), sizeof(SJCBoundary));
  out.write((char*)&(m_VBoundary[1]), sizeof(SJCBoundary));
}

//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
bool SJCField2d::
Read(std::istream &in)
//============================================================================
{
  // Output NX, NY
  in.read((char*)&m_uNX, sizeof(uint));
  in.read((char*)&m_uNY, sizeof(uint));

  // Output DX, DY
  in.read((char*)&m_dDX, sizeof(double));
  in.read((char*)&m_dDY, sizeof(double));

  m_dHalfDX = 0.5 * m_dDX;
  m_dHalfDY = 0.5 * m_dDY;

  m_dLengthX      = m_dDX * (double)m_uNX;
  m_dLengthY      = m_dDY * (double)m_uNY;

  // Output Boundary
  in.read((char*)&(m_VBoundary[0]), sizeof(SJCBoundary));
  in.read((char*)&(m_VBoundary[1]), sizeof(SJCBoundary));

  return true;
}
