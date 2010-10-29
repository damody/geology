/************************************************************************
     Main File:

     File:        SJCField3.cpp

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     Basic class for 3D field

     Constructors:
                  1. 0 : the default contructor
                  2. 2 :
                  3. 3 : 
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2.
************************************************************************/

#include "SJCField3.h"

//****************************************************************************
//
// * Contructor to set up all value
//============================================================================
SJCField3d::
SJCField3d(const uint nx,        const uint ny,        const uint nz,
	   const double dx,      const double dy,      const double dz,
	   const SJCBoundary bx, const SJCBoundary by, const SJCBoundary bz)
//============================================================================
{
  m_uNX          = nx;
  m_uNY          = ny;
  m_uNZ          = nz;

  m_dDX          = dx;
  m_dDY          = dy;
  m_dDZ          = dz;

  m_dHalfDX      = dx * 0.5;
  m_dHalfDY      = dy * 0.5;
  m_dHalfDZ      = dz * 0.5;

  m_dLengthX     = m_dDX * (double)m_uNX;
  m_dLengthY     = m_dDY * (double)m_uNY;
  m_dLengthZ     = m_dDZ * (double)m_uNZ;
  
  m_VBoundary[0] = bx;
  m_VBoundary[1] = by;
  m_VBoundary[2] = by;

}


//****************************************************************************
//
// * Get the scalar field from input
//============================================================================
SJCField3d::
SJCField3d(std::istream &f)
//============================================================================
{
  f.read((char*)&m_uNX, sizeof(uint));
  f.read((char*)&m_uNY, sizeof(uint));
  f.read((char*)&m_uNZ, sizeof(uint));

  f.read((char*)&m_dDX, sizeof(double));
  f.read((char*)&m_dDY, sizeof(double));
  f.read((char*)&m_dDZ, sizeof(double));

  m_dHalfDX = 0.5 * m_dDX;
  m_dHalfDY = 0.5 * m_dDY;
  m_dHalfDZ = 0.5 * m_dDZ;

  m_dLengthX      = m_dDX * (double)m_uNX;
  m_dLengthY      = m_dDY * (double)m_uNY;
  m_dLengthZ      = m_dDZ * (double)m_uNZ;
  
  f.read((char*)&(m_VBoundary[0]), sizeof(SJCBoundary));
  f.read((char*)&(m_VBoundary[1]), sizeof(SJCBoundary));
  f.read((char*)&(m_VBoundary[2]), sizeof(SJCBoundary));

}



//****************************************************************************
//
// * Copy contructor
//============================================================================
SJCField3d::
SJCField3d(const SJCField3d &vf)
//============================================================================
{
  Assign(vf);
}

//****************************************************************************
//
// * Copy contructor
//============================================================================
void SJCField3d::
Assign(const SJCField3d &vf)
//============================================================================
{
  m_uNX          = vf.m_uNX;
  m_uNY          = vf.m_uNY;
  m_uNZ          = vf.m_uNZ;
 
  m_dDX          = vf.m_dDX;
  m_dDY          = vf.m_dDY;
  m_dDZ          = vf.m_dDZ;
  
  m_dHalfDX      = m_dDX * 0.5;
  m_dHalfDY      = m_dDY * 0.5;
  m_dHalfDZ      = m_dDZ * 0.5;
  
  m_dLengthX     = m_dDX * (double)m_uNX;
  m_dLengthY     = m_dDY * (double)m_uNY;
  m_dLengthZ     = m_dDZ * (double)m_uNZ;

  m_VBoundary[0] = vf.m_VBoundary[0];
  m_VBoundary[1] = vf.m_VBoundary[1];
  m_VBoundary[2] = vf.m_VBoundary[2];

}



//****************************************************************************
//
// * Destructor
//============================================================================
SJCField3d::
~SJCField3d(void)
//============================================================================
{
  Destroy();
}


//****************************************************************************
//
// * Assign operator
//============================================================================
SJCField3d& SJCField3d::
operator=(const SJCField3d &vf)
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
void SJCField3d::
Set(const uint nx,        const uint ny,        const uint nz,
    const double dx,      const double dy,      const double dz,
    const SJCBoundary bx, const SJCBoundary by, const SJCBoundary bz)
//============================================================================
{
  Destroy();
  m_uNX          = nx;
  m_uNY          = ny;
  m_uNZ          = nz;

  m_dDX          = dx;
  m_dDY          = dy;
  m_dDZ          = dz;

  m_dHalfDX      = dx * 0.5;
  m_dHalfDY      = dy * 0.5;
  m_dHalfDZ      = dz * 0.5;

  m_dLengthX     = m_dDX * (double)m_uNX;
  m_dLengthY     = m_dDY * (double)m_uNY;
  m_dLengthZ     = m_dDZ * (double)m_uNZ;
  
  m_VBoundary[0] = bx;
  m_VBoundary[1] = by;
  m_VBoundary[2] = bz;

}


//****************************************************************************
//
// * Get the minimum differential value in x direction
//============================================================================
double SJCField3d::
BoundMinX(void) const
//============================================================================
{
  switch ( m_VBoundary[0] )   {
    case BOUNDARY_NOWRAP:  // The outest layer is a wall
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
double SJCField3d::
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
// * Return the minimum differential value in y direction
//============================================================================
double SJCField3d::
BoundMinZ(void) const
//============================================================================
{
 switch ( m_VBoundary[2] ) {
    case BOUNDARY_NOWRAP: // The outest layer is a wall
      return m_dDZ;
    case BOUNDARY_NOWRAP_FREE: // Not a wall but a constant boundary
      return m_dDZ;
    case BOUNDARY_WRAP: 
      return 0.0f;
  }

  return m_dDZ;

 
}


//****************************************************************************
//
// * Return the differential maximum value in x direction
//============================================================================
double SJCField3d::
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
double SJCField3d::
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
// * Return the differential maximum value in y direction
//============================================================================
double SJCField3d::
BoundMaxZ(void) const
//============================================================================
{
  switch ( m_VBoundary[2] )    {
    case BOUNDARY_NOWRAP: 
     return m_dLengthZ - m_dDZ;
    case BOUNDARY_NOWRAP_FREE: // Not a wall but a constant boundary
      return m_dLengthZ - m_dDZ;
    case BOUNDARY_WRAP: 
      return m_dLengthZ;
  }
  return m_dLengthY - m_dDZ;

}

//****************************************************************************
//
// * Clear the m_VData
//============================================================================
void SJCField3d::
Destroy(void)
//============================================================================
{
  // Release the data
  m_uNX      = m_uNY      = m_uNZ      = 0;
  m_dDX      = m_dDY      = m_dDZ      = 0.f;
  m_dHalfDX  = m_dHalfDY  = m_dHalfDZ  = 0.f;
  m_dLengthX = m_dLengthY = m_dLengthX = 0.f;
  
}


//****************************************************************************
//
// * Output operator
//============================================================================
std::ostream&
operator<<(std::ostream &o, const SJCField3d &vf)
//============================================================================
{
  // Output NX, NY, NZ
  o << std::setw(4) << vf.m_uNX << " " 
    << std::setw(4) << vf.m_uNY << " "
    << std::setw(4) << vf.m_uNZ << " ";

  // Output DX, DY, DZ
  o << std::setw(8) << vf.m_dDX << " " 
    << std::setw(8) << vf.m_dDY << " " 
    << std::setw(8) << vf.m_dDZ << " " ;
 
  // Output Boundary
  o << vf.m_VBoundary[0] << " "
    << vf.m_VBoundary[1] << " " 
    << vf.m_VBoundary[1] << " \n";

  return o;
}

//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
void
SJCField3d::Write(std::ostream &o)
//============================================================================
{
  // Output NX, NY, NZ
  o.write((char*)&m_uNX, sizeof(uint));
  o.write((char*)&m_uNY, sizeof(uint));
  o.write((char*)&m_uNZ, sizeof(uint));
  // Output DX, DY, DZ
  o.write((char*)&m_dDX, sizeof(double));
  o.write((char*)&m_dDY, sizeof(double));
  o.write((char*)&m_dDZ, sizeof(double));
  // Output Boundary
  o.write((char*)&(m_VBoundary[0]), sizeof(SJCBoundary));
  o.write((char*)&(m_VBoundary[1]), sizeof(SJCBoundary));
  o.write((char*)&(m_VBoundary[2]), sizeof(SJCBoundary));
}


