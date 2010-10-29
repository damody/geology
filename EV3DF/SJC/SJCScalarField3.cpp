/************************************************************************
     Main File:

     File:        SJCScalarField3.cpp

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     Class to handle the scalar field

     Constructors:
                  1. 0 : the default contructor
                  2. 2 :
                  3. 3 : 
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2.
************************************************************************/

#include "SJCScalarField3.h"

#include <iomanip>
#include <assert.h>

//****************************************************************************
//
// * Contructor to set up all value
// * step on use for increasing not 1
// * step 用在非1遞增的位移中，一次跳一個double的大小
//============================================================================
SJCScalarField3d::
SJCScalarField3d(const uint nx,        const uint ny,       const uint nz,
		 const double dx,      const double dy,     const double dz,
		 const SJCBoundary bx, const SJCBoundary by, 
		 const SJCBoundary bz, const double *d, const uint step)
		 : SJCField3d(nx, ny, nz, dx, dy, dz, bx, by, bz)
		 //============================================================================
{
	m_VData = new double[m_uNX * m_uNY * m_uNZ];
	if (d) {
		for ( uint i = 0 ; i < m_uNX * m_uNY * m_uNZ ; i++ )
			m_VData[i] = d[i*step];
	}
	else {
		for( uint i = 0 ; i < m_uNX * m_uNY * m_uNZ; i++)
			m_VData[i] = 0;
	}
}

//****************************************************************************
//
// * Get the scalar field from input
//============================================================================
SJCScalarField3d::
SJCScalarField3d(std::istream &f)
  : SJCField3d(f)
//============================================================================
{
  m_VData = new double[m_uNX * m_uNY * m_uNZ];
  f.read((char*)m_VData, m_uNX * m_uNY * m_uNZ * sizeof(double));
}



//****************************************************************************
//
// * Copy contructor
//============================================================================
SJCScalarField3d::
SJCScalarField3d(const SJCScalarField3d &vf)
  : SJCField3d(vf)
//============================================================================
{
  Assign(vf);
}

//****************************************************************************
//
// * Copy contructor
//============================================================================
void SJCScalarField3d::
Assign(const SJCScalarField3d &vf)
//============================================================================
{
  SJCField3d::Assign(vf);

  m_VData = new double[m_uNX * m_uNY * m_uNZ];
  for ( uint i = 0 ; i < m_uNX * m_uNY * m_uNZ ; i++ )
    m_VData[i] = vf.m_VData[i];
}



//****************************************************************************
//
// * Destructor
//============================================================================
SJCScalarField3d::
~SJCScalarField3d(void)
//============================================================================
{
  Destroy();
}


//****************************************************************************
//
// * Clear the m_VData
//============================================================================
void SJCScalarField3d::
Destroy(void)
//============================================================================
{
  SJCField3d::Destroy();

  // Release the data
  delete [] m_VData;
  m_VData   = 0;
  
}

//****************************************************************************
//
// * Assign operator
//============================================================================
SJCScalarField3d& SJCScalarField3d::
operator=(const SJCScalarField3d &vf)
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
// * According to the boundary get the value at x, y, z
//============================================================================
double SJCScalarField3d::
Value(const double x_pos, const double y_pos, const double z_pos)
//============================================================================
{
 
  // Check whether the position is within valid simulation area
  if(x_pos < 0 || x_pos > m_dLengthX || 
     y_pos < 0 || y_pos > m_dLengthX ||
     z_pos < 0 || z_pos > m_dLengthX)  {
    
    SJCWarning("X or Y out of bound in getting value");
    return 0;
  }
  
  // Compute the index position
  double x = (x_pos - m_dHalfDX) / m_dDX;
  double y = (y_pos - m_dHalfDY) / m_dDY;
  double z = (z_pos - m_dHalfDZ) / m_dDZ;

  uint	   index_xl, index_xr;  // the left and right index in x
  uint	   index_yb, index_yt;  // The top and bottom index in y
  uint	   index_zb, index_zf;  // The front and back index in z
  double   partial_x, partial_y, partial_z; // The partial in x, y, z

  // Here should have judgement when x_pos and y_pos are close to index 
  // position
  switch ( m_VBoundary[0] )   { // Check and set up the correct value for x,y 
                                // according the boundary conditions
    case BOUNDARY_NOWRAP: 
    case BOUNDARY_NOWRAP_FREE: 
      if ( x == m_uNX - 1.f) {
	index_xr  = (uint)m_uNX - 1;
	index_xl  = index_xr - 1;
	partial_x = 1.0f;
      }
      else {
	index_xl  = (uint)floor(x);
	index_xr  = index_xl + 1;
	partial_x = x - (double)index_xl;
      }
      break;
      
    case BOUNDARY_WRAP: 
      double	xp = fmod(x, (double)m_uNX);
      if ( xp < 0.0 )
	xp = xp + (int)m_uNX;
      
      index_xl  = (uint)floor(xp);
      index_xr  = (index_xl + 1) % m_uNX;
      partial_x = xp - (double)index_xl;
      break;
  } // end of switch

  switch ( m_VBoundary[1] )   {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE: 
      if ( y == m_uNY - 1.f ) {
	index_yt  = (uint)m_uNY - 1;
	index_yb  = index_yt - 1;
	partial_y = 1.0f;
      }
      else {
	index_yb  = (uint)floor(y);
	index_yt  = index_yb + 1;
 	partial_y = y - (double)index_yb;
      }
      break;

    case BOUNDARY_WRAP: 
      double 	yp = fmod(y, (double)m_uNY);
      if ( yp < 0.0 )
	yp = yp + (int)m_uNY;
      
      index_yb  = (uint)floor(yp);
      index_yt  = (index_yb + 1) % m_uNY;
      partial_y = yp - (double)index_yb;
      break;
  }
  switch ( m_VBoundary[2] )   {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE: 
      if ( z == m_uNZ - 1.f ) {
	index_zf  = (uint)m_uNZ - 1;
	index_zb  = index_zf - 1;
	partial_z = 1.0f;
      }
      else {
	index_zb  = (uint)floor(z);
	index_zf  = index_zb + 1;
 	partial_z = z - (double)index_zb;
      }
      break;

    case BOUNDARY_WRAP: 
      double 	zp = fmod(z, (double)m_uNZ);
      if ( zp < 0.0 )
	zp = zp + (int)m_uNZ;
      
      index_zb  = (uint)floor(zp);
      index_zf  = (index_zb + 1) % m_uNZ;
      partial_z = zp - (double)index_zb;
      break;
  }

  double vLeftBottomBack   = m_VData[Index(index_xl, index_yb, index_zb)];
  double vLeftBottomFront  = m_VData[Index(index_xl, index_yb, index_zf)];
  double vLeftTopBack      = m_VData[Index(index_xl, index_yt, index_zb)];
  double vLeftTopFront     = m_VData[Index(index_xl, index_yt, index_zf)];
  double vRightBottomBack  = m_VData[Index(index_xr, index_yb, index_zb)];
  double vRightBottomFront = m_VData[Index(index_xr, index_yb, index_zf)];
  double vRightTopBack     = m_VData[Index(index_xr, index_yt, index_zb)];
  double vRightTopFront    = m_VData[Index(index_xr, index_yt, index_zf)];

  return
    (1.f-partial_x)*(1.f-partial_y)*(1.f-partial_z) * vLeftBottomBack +
    (1.f-partial_x)*(1.f-partial_y)*partial_z       * vLeftBottomFront +
    (1.f-partial_x)*partial_y      *(1.f-partial_z) * vLeftTopBack +
    (1.f-partial_x)*partial_y      *partial_z       * vLeftTopFront +
    partial_x      *(1.f-partial_y)*(1.f-partial_z) * vRightBottomBack+
    partial_x      *(1.f-partial_y)* partial_z      * vRightBottomFront+
    partial_x      * partial_y     * (1.f-partial_z)* vRightTopBack +
    partial_x      *partial_y      * partial_z      * vRightTopFront ;
}

//****************************************************************************
//
// * Access the value at (x, y)
//============================================================================
double& SJCScalarField3d::
operator()(const uint index_x, const uint index_y, const uint index_z)
//============================================================================
{
	int i = Index(index_x, index_y, index_z);
  return m_VData[i];
}


//****************************************************************************
//
// * Compute the gradient of the scalar field
//   partial p / partial x = (p(x + .5d, y, z) - p(x - .5d, y, z)) / d
//   partial p / partial x = (p(x, y + .5d, z) - p(x, y - .5d, z)) / d
//   partial p / partial z = (p(x, y, z + .5d) - p(x, y, z - .5d)) / d
//============================================================================
SJCVector3d
SJCScalarField3d::Grad(const double x, const double y, const double z)
//============================================================================
{
  double   pleft    = Value(x - m_dHalfDX, y, z);
  double   pright   = Value(x + m_dHalfDX, y, z);

  double   ptop     = Value(x, y + m_dHalfDY, z);
  double   pbottom  = Value(x, y - m_dHalfDY, z);

  double   pfront   = Value(x, y, z + m_dHalfDZ);
  double   pback    = Value(x, y, z - m_dHalfDZ);

  return SJCVector3d( (pright - pleft) / m_dDX , 
		      (ptop - pbottom) / m_dDY, 
		      (pfront - pback) / m_dDZ);
}


//****************************************************************************
//
// * Output operator
//============================================================================
std::ostream&
operator<<(std::ostream &o, const SJCScalarField3d &vf)
//============================================================================
{
  o << (SJCField3d&)vf;

  for(uint k = 0; k < vf.m_uNZ - 1; k++) {
    for ( uint j = 0; j < vf.m_uNY - 1 ; j++ )   {
      for ( uint i = 0 ; i < vf.m_uNX ; i++ ) {
	o << std::setw(8) 
	  << vf.m_VData[i * vf.m_uNY * vf.m_uNZ + j * vf.m_uNZ + k] << " ";
      } // end of for i
    } // end of for k
  } // end of for j
  o << std::endl;

  return o;
}


//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
void
SJCScalarField3d::Write(std::ostream &o)
//============================================================================
{
  SJCField3d::Write(o);
  o.write((char*)m_VData, m_uNX * m_uNY * m_uNZ * sizeof(double));
}


