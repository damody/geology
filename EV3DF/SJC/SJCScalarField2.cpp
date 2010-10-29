/************************************************************************
     Main File:

     File:        SJCScalarField2.cpp

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

#include "SJCScalarField2.h"



//****************************************************************************
//
// * Contructor to set up all value
//============================================================================
SJCScalarField2d::
SJCScalarField2d(const uint nx,        const uint ny,
		 const double dx,      const double dy,
		 const SJCBoundary bx, const SJCBoundary by,
		 const double *d)
  : SJCField2d(nx, ny, dx, dy, bx, by)
//============================================================================
{
  m_VData = new double[m_uNX * m_uNY];
  if (d) { // If we have data, copy it in
    for ( uint i = 0 ; i < m_uNX * m_uNY ; i++ )
      m_VData[i] = d[i];
  }
  else { // no data, reset to zero
    for( uint i = 0 ; i < m_uNX * m_uNY; i++)
      m_VData[i] = 0;
  }
}


//****************************************************************************
//
// * Get the scalar field from input
//============================================================================
SJCScalarField2d::
SJCScalarField2d(std::istream &f)
  : SJCField2d(f)
//============================================================================
{
  m_VData = new double[m_uNX * m_uNY];
  f.read((char*)m_VData, m_uNX * m_uNY * sizeof(double));
}



//****************************************************************************
//
// * Copy contructor
//============================================================================
SJCScalarField2d::
SJCScalarField2d(const SJCScalarField2d &vf)
  : SJCField2d(vf)
//============================================================================
{
  Assign(vf);
}

//****************************************************************************
//
// * Copy contructor
//============================================================================
void SJCScalarField2d::
Assign(const SJCScalarField2d &vf)
//============================================================================
{
  SJCField2d::Assign(vf);

  m_VData = new double[m_uNX * m_uNY];
  for ( uint i = 0 ; i < m_uNX * m_uNY ; i++ )
    m_VData[i] = vf.m_VData[i];
}



//****************************************************************************
//
// * Destructor
//============================================================================
SJCScalarField2d::
~SJCScalarField2d(void)
//============================================================================
{
  Destroy();
}

//****************************************************************************
//
// * Clear the m_VData
//============================================================================
void SJCScalarField2d::
Destroy(void)
//============================================================================
{

  SJCField2d::Destroy();

  // Release the data
  delete [] m_VData;
  m_VData   = 0;
  
}

//****************************************************************************
//
// * Assign operator
//============================================================================
SJCScalarField2d& SJCScalarField2d::
operator=(const SJCScalarField2d &vf)
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
// * According to the boundary get the value at x, y
//============================================================================
double SJCScalarField2d::
Value(const double x_pos, const double y_pos)
//============================================================================
{
  // Check whether the position is within valid simulation area
  if(x_pos < MinX() || x_pos > MaxX() || 
     y_pos < MinY() || y_pos > MaxY()){
    SJCWarning("X or Y out of bound in getting value");
    return 0;
  }
  

  // Compute the index position
  double x = (x_pos - m_dHalfDX) / m_dDX;
  double y = (y_pos - m_dHalfDY) / m_dDY;
 
  uint	   index_xl, index_xr;    // the left and right index in x
  uint	   index_yb, index_yt;    // The top and bottom index in y
  double   partial_x,  partial_y; // The partial in x and y

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

  double vBottomLeft  = m_VData[Index(index_xl, index_yb)];
  double vTopLeft     = m_VData[Index(index_xl, index_yt)];
  double vBottomRight = m_VData[Index(index_xr, index_yb)];
  double vTopRight    = m_VData[Index(index_xr, index_yt)];

  return (1.f-partial_x)*(1.f-partial_y) * vBottomLeft + 
         partial_y *     (1.f-partial_x) * vTopLeft + 
         partial_x *     (1.f-partial_y) * vBottomRight + 
         partial_x *      partial_y      * vTopRight;
}



//****************************************************************************
//
// * Access the value at (x, y)
//============================================================================
double& SJCScalarField2d::
operator()(const uint index_x, const uint index_y)
//============================================================================
{
  return m_VData[Index(index_x, index_y)];
}


//****************************************************************************
//
// * Compute the gradient of the scalar field
//   partial p / partial x = (p(x + .5d, y) - p(x - .5d, y)) / d
//   partial p / partial x = (p(x, y + .5d) - p(x, y - .5d)) / d
//============================================================================
SJCVector2d SJCScalarField2d::
Grad(const double x, const double y)
//============================================================================
{
  double pleft   = Value(x - m_dHalfDX, y);
  double pright  = Value(x + m_dHalfDX, y);
  double ptop    = Value(x, y + m_dHalfDY);
  double pbottom = Value(x, y - m_dHalfDY);

  return SJCVector2d( (pright - pleft) / m_dDX , (ptop - pbottom) / m_dDY);
}

//****************************************************************************
//
// * Compute the curl of the scalar field
//   dx = p(x, y + 0.5) - p(x, y - 0.5)
//   dy = p(x + 0.5, y) - p(x - 0.5, y) 
//   weird
//============================================================================
SJCVector2d
SJCScalarField2d::Curl(const double x, const double y)
//============================================================================
{
  double pleft   = Value(x - m_dHalfDX, y);
  double pright  = Value(x + m_dHalfDX, y);
  double ptop    = Value(x, y + m_dHalfDY);
  double pbottom = Value(x, y - m_dHalfDY);
  
  return SJCVector2d((ptop - pbottom) / m_dDX, (pleft - pright) / m_dDY);
}

//****************************************************************************
//
// * Output operator
//============================================================================
std::ostream&
operator<<(std::ostream &o, const SJCScalarField2d &vf)
//============================================================================
{
  o << (SJCField2d&)vf;

  for ( uint j = 0; j < vf.m_uNY ; j++ )   {
    for ( uint i = 0 ; i < vf.m_uNX ; i++ ){
      o << std::setw(3) << vf.m_VData[i * vf.m_uNY + j] << " ";
    }
    o << std::endl;
  }
  return o;
}

//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
void
SJCScalarField2d::Write(std::ostream &o)
//============================================================================
{
  SJCField2d::Write(o);
  o.write((char*)m_VData, m_uNX * m_uNY * sizeof(double));
}


