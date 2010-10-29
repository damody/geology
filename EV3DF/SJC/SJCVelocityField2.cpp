/************************************************************************
     Main File:

     File:        SJCVelocityField2.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     To handle the 2D std::vector field

     Constructors:
                  1. 0 : the default contructor
                  2. 4 : constructor to set up all value by input parameters
                  3. 1 : copy contructor
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2. (): Get the value of the scalar field
                 3. >>: output in the ascii form
 ************************************************************************/

#include "SJCVelocityField2.h"

const uint  SJCVelocityField2d::m_cuIntegralSteps = 20;

//*****************************************************************************
//
// * Contructor to set up everything
//============================================================================
SJCVelocityField2d::
SJCVelocityField2d(const uint nx, const uint ny,
		   const double dx, const double dy,
		   const SJCBoundary bx, const SJCBoundary by,
		   const SJCVector2d *d)
  : SJCField2d(nx, ny, dx, dy, bx, by)
//============================================================================
{
  m_uNVX         = nx + 1;
  m_uNVY         = ny + 1;
 
  m_VData = new SJCVector2d[m_uNVX * m_uNVY];
  if (d) {
    for ( uint i = 0 ; i < m_uNVX * m_uNVY ; i++ )
      m_VData[i] = d[i];
  }
}

//*****************************************************************************
//
// * Contructor from scalar field
//============================================================================
SJCVelocityField2d::
SJCVelocityField2d(const SJCScalarField2d &pf)
  : SJCField2d(pf)
//============================================================================
{
  m_uNVX         = pf.NumX() + 1;
  m_uNVY         = pf.NumY() + 1;
 
  m_VData = new SJCVector2d[m_uNVX * m_uNVY];

  SJCWarning("Not implement here");
}



//*****************************************************************************
//
// * Copy contructor
//============================================================================
SJCVelocityField2d::
SJCVelocityField2d(const SJCVelocityField2d &vf)
  : SJCField2d(vf)
//============================================================================
{
  m_VData = 0;
  Assign(vf);
}

//****************************************************************************
//
// * Assign 
//============================================================================
void SJCVelocityField2d::
Assign(const SJCVelocityField2d &vf)
//============================================================================
{
  SJCField2d::Assign(vf);
  m_uNVX = m_uNX + 1;
  m_uNVY = m_uNY + 1;

  m_VData = new SJCVector2d[m_uNVX * m_uNVY];
 
  for ( uint i = 0 ; i < m_uNVX * m_uNVY ; i++ ){
    m_VData[i] = vf.m_VData[i];
  }
}

//*****************************************************************************
//
// * Copy contructor
//============================================================================
SJCVelocityField2d::
SJCVelocityField2d(std::istream& f)
  : SJCField2d(f)
//============================================================================
{
  ReadCommon(f);
}

//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
void SJCVelocityField2d::
ReadCommon(std::istream &i)
//============================================================================
{
  if(m_VData)
    delete [] m_VData;

  m_uNVX = m_uNX + 1;
  m_uNVY = m_uNY + 1;
  
  
  m_VData = new SJCVector2d[m_uNVX * m_uNVY];
  i.read((char*)m_VData, m_uNVX * m_uNVY * sizeof(SJCVector2d));
}

//*****************************************************************************
//
// * Destructor
//============================================================================
SJCVelocityField2d::
~SJCVelocityField2d(void)
//============================================================================
{
  Destroy();
}

//*****************************************************************************
//
// * Clear the data
//============================================================================
void SJCVelocityField2d::
Destroy(void)
//============================================================================
{
  SJCField2d::Destroy();

  delete[] m_VData;
  m_VData   = 0;

}

//*****************************************************************************
//
// * Assign operator
//============================================================================
SJCVelocityField2d& SJCVelocityField2d::
operator=(const SJCVelocityField2d &vf)
//============================================================================
{
  // Clear data
  Destroy();

  // Do the copy
  Assign(vf);
   
  return *this;
}

//*****************************************************************************
//
// * Get the std::vector field at x, y
//============================================================================
SJCVector2d SJCVelocityField2d::
Value(const double x_pos, const double y_pos)
//============================================================================
{ 
  if(x_pos < MinX() || x_pos > MaxX() ||
     y_pos < MinY() || y_pos > MaxY()){
    SJCWarning("The position is out of bound in velocity field");
  }
  
  //**************************************************************************
  // For VX
  //**************************************************************************
  double  x = x_pos / m_dDX;
  double  y = (y_pos - m_dHalfDY) / m_dDY;

  uint	   index_xl, index_xr;    // the left and right index in x
  uint	   index_yb, index_yt;    // The top and bottom index in y
  double   partial_x,  partial_y; // The partial in x and y

  //*************************************************************************
  //
  // We can do this because x from 0 to NX
  // Need to consider about what's the periodic behavior in VX
  //
  //************************************************************************
  switch ( m_VBoundary[0] )   {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      if ( x >= m_uNVX - 1.f ) { 
	index_xr  = (uint)m_uNVX - 1;
	index_xl  = index_xr - 1;
	partial_x = 1.0f;
      }
      else if( x <= 0.f){
	index_xr  = 1;
	index_xl  = 0;
	partial_x = 0.f;    
      }
      else { 
	index_xl  = (uint)floor(x);
	index_xr  = index_xl + 1;
	partial_x = x - (double)index_xl;
      }
      break;
    case BOUNDARY_WRAP: 
      double 	xp = fmod(x, (double)m_uNX);
      if ( xp < 0.0 )
	xp = xp + (int)m_uNX;
      
      index_xl  = (uint)floor(xp); 
      index_xr  = index_xl + 1; // Do not need mode because xl is from 0
                                // NX - 1
      partial_x = xp - (double)index_xl;
      break;
  } // end of switch
  
  switch ( m_VBoundary[1] )   {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      if ( y >= m_uNY - 1.f ) {
	index_yt  = (uint)m_uNY - 1;
	index_yb  = index_yt - 1;
	partial_y = 1.0f;
      }
      else if( y <= 0.f) {
	index_yt  = 1;
	index_yb  = 0;
	partial_y = 0.f;
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
  } // end of switch

  double vBottomLeft  = m_VData[Index(index_xl, index_yb)].x();
  double vTopLeft     = m_VData[Index(index_xl, index_yt)].x();
  double vBottomRight = m_VData[Index(index_xr, index_yb)].x();
  double vTopRight    = m_VData[Index(index_xr, index_yt)].x();
  double vx = (1.f-partial_x)*(1.f-partial_y) * vBottomLeft + 
              (1.f-partial_x)*partial_y       * vTopLeft + 
               partial_x     *(1.f-partial_y) * vBottomRight + 
               partial_x     * partial_y      * vTopRight;

  // Here should have judgement when x_pos and y_pos are close to index 
  // position
  x = (x_pos - m_dHalfDX) / m_dDX;
  y = y_pos / m_dDY;
  

  switch ( m_VBoundary[0] )   { // Check and set up the correct value for x,y 
                                // according the boundary conditions
     case BOUNDARY_NOWRAP_FREE:
     case BOUNDARY_NOWRAP: 
      if ( x >= m_uNX - 1.f) {
	index_xr  = (uint)m_uNX - 1;
	index_xl  = index_xr - 1;
	partial_x = 1.0f;
      }
      else if( x <= 0){
	index_xr  = 1;
	index_xl  = 0;
	partial_x = 0.f;
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
      partial_x = xp - index_xl;
      break;
  } // end of switch

  switch ( m_VBoundary[1] )   {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      if ( y >= m_uNVY - 1.f ) { // should never happen
	index_yt  = (uint)m_uNY - 1;
	index_yb  = index_yt - 1;
	partial_y = 1.0f;
      }
      else  if(y <= 0.f){
	index_yt  = 1;
	index_yb  = 0;
	partial_y = 0.f;
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
      index_yt  = index_yb + 1;          // Because we guaratee 1 to nx -1
      partial_y = yp - (double)index_yb;
      break;
  } // end of switch

 
  vBottomLeft  = m_VData[Index(index_xl, index_yb)].y();
  vTopLeft     = m_VData[Index(index_xl, index_yt)].y();
  vBottomRight = m_VData[Index(index_xr, index_yb)].y();
  vTopRight    = m_VData[Index(index_xr, index_yt)].y();

  double vy = (1.f-partial_x)*(1.f-partial_y) * vBottomLeft + 
              (1.f-partial_x)*partial_y       *  vTopLeft + 
               partial_x     *(1.f-partial_y) * vBottomRight + 
               partial_x     *partial_y       * vTopRight;

  // Out of tiling bound.
  if(x_pos < 0 && x_pos >= m_dLengthX)
    vx = 0;
  if(y_pos < 0 && y_pos >= m_dLengthY)
    vy = 0;

  return SJCVector2d(vx, vy);
}

//*****************************************************************************
//
// * Get the std::vector field at x, y
//============================================================================
SJCVector2d SJCVelocityField2d::
Value(const double x_pos, const double y_pos,  const unsigned char r)
//============================================================================
{ 
  double   xr;
  double   yr;

  if ( r <= 1 )   {
    if ( r < 1 ) {
      xr = x_pos;
      yr = y_pos;
    }
    else {
      xr = y_pos;
      yr = (double)m_uNX - 1.0f - x_pos;
    }
  }
  else  {
    if ( r < 3 ) {
      xr = m_uNX - 1.0f - x_pos;
      yr = m_uNY - 1.0f - y_pos;
    }
    else {
      xr = m_uNX - 1.0f - y_pos;
      yr = x_pos;
    }
  }

  return Value(xr, yr);
}

//*****************************************************************************
//
// * Put the velocity at the center of the voxel
//============================================================================
SJCVector2d  SJCVelocityField2d:: 
VoxelCenterVelocity(const uint index_x, const uint index_y)
//============================================================================
{
  SJCVector2d center = m_VData[Index(index_x, index_y)];
  SJCVector2d right  = m_VData[Index(index_x + 1, index_y)];
  SJCVector2d top    = m_VData[Index(index_x, index_y + 1)];

  return SJCVector2d(.5f * (center.x() + right.x()), 
		     .5f * (center.y() + top.y()));
}

//*****************************************************************************
//
// * Put the velocity at the center of the voxel
//============================================================================
void  SJCVelocityField2d:: 
AddForce(SJCScalarField2d* bound, SJCVectorField2d* force, 
	 const double dt)
//============================================================================
{
  for(uint j = 1; j < m_uNY - 1; j++){
    for(uint i = 1; i < m_uNX - 1; i++) {

      // No problem about this because we can access (nx -1, ny-1)
      SJCVector2d center = (*force)(i, j);
      SJCVector2d left   = (*force)(i - 1, j);
      SJCVector2d bottom = (*force)(i, j - 1);
      SJCVector2d delta; 

      // Left bottom corner
      if( i == 1 && j == 1) 
	continue;

      // Left edge and it include the left top corner
      else if( i == 1 ){
	delta.set(0.f, 0.5 * dt * (center.y() + bottom.y()));
      }

      // Bottom edge include the right bottom corner
      else if( j == 1){
	delta.set(0.5 * dt * (center.x() + left.x()), 0.f);
      }
      else    // Other include the top and right edge
	delta.set(0.5 * dt * (center.x() + left.x()), 
		  0.5 * dt * (center.y() + bottom.y()));
      m_VData[Index(i, j)] += delta;
    } // end of i
  } // end of j
}


//*****************************************************************************
//
// * Return the X component of i, j component
//============================================================================
double& SJCVelocityField2d::
VX(const uint index_x, const uint index_y)
//============================================================================
{
  int ip, jp;
  
  switch ( m_VBoundary[0] ) {
    case BOUNDARY_NOWRAP_FREE:
    case BOUNDARY_NOWRAP:
      ip = index_x;
    case BOUNDARY_WRAP:
      ip = index_x % m_uNVX;
      if ( ip < 0 ) 
	ip += m_uNVX;
  }
  
  switch (m_VBoundary[1]){
    case BOUNDARY_NOWRAP_FREE:
    case BOUNDARY_NOWRAP:
      jp = index_y;
  
    case BOUNDARY_WRAP:
      jp = index_y % m_uNY;
      if ( jp < 0 ) 
	jp += m_uNY;
  }
  
  return m_VData[Index(index_x, index_y)][0];
}


//*****************************************************************************
//
// * Return the Y component of i, j element
//============================================================================
double& SJCVelocityField2d::VY(const uint index_x, const uint index_y)
//============================================================================
{
 int ip, jp;
  
  switch ( m_VBoundary[0] ) {
    case BOUNDARY_NOWRAP_FREE:
    case BOUNDARY_NOWRAP:
      ip = index_x;
    case BOUNDARY_WRAP:
      ip = index_x % m_uNX;
      if ( ip < 0 ) 
	ip += m_uNX;
  }
  
  switch (m_VBoundary[1]){
    case BOUNDARY_NOWRAP_FREE:
    case BOUNDARY_NOWRAP:
      jp = index_y;
  
    case BOUNDARY_WRAP:
      jp = index_y % m_uNVY;
      if ( jp < 0 ) 
	jp += m_uNVY;
  }
  
  return m_VData[Index(index_x, index_y)][1];
}

//*****************************************************************************
//
// * Trace the particle back, I should send in a boundary
//============================================================================ 
void SJCVelocityField2d::
TraceParticle(const double dt, const SJCVector2d curr, SJCVector2d& prev)
//============================================================================ 
{
  // Cacluate the steps
  double step_dt = dt / (double)m_cuIntegralSteps;

  prev = curr;
  
  // Backward integrate to find the start position
  for (double time = dt; time <= 0; time -= step_dt) {
    SJCVector2d velocity = (*this)(prev);
    
    prev += step_dt * velocity;

    // Check whether hit the x boundary
    if ( prev[0] < BoundMinX() )
      prev[0]= BoundMinX();
    if ( prev[0] >= BoundMaxX() )
      prev[0] = BoundMaxX() - 1.0e-6;

    // Check whether hit the y boundary
    if ( prev[1] < BoundMinY() )
      prev[1]= BoundMinY();
    if ( prev[1] >= BoundMaxY() )
      prev[1] = BoundMaxY() - 1.0e-6;
  }
}

//*****************************************************************************
//
// * Get the divergence at voxel index (x,y) 
//============================================================================ 
double SJCVelocityField2d::
Divergence(const uint x, const uint y)
//============================================================================ 
{
  double vx_left  = m_VData[Index(x, y)].x();
  double vx_right = m_VData[Index(x + 1, y)].x();
  
  double vy_bottom = m_VData[Index(x, y)].y();
  double vy_top    = m_VData[Index(x, y + 1)].y();
 
  return ((vx_right - vx_left) / m_dDX + 
	  (vy_top - vy_bottom) / m_dDY);
}



//*****************************************************************************
//
// * The output operator
//============================================================================
std::ostream& operator<<(std::ostream &o, const SJCVelocityField2d &vf)
//============================================================================
{
  o << (SJCField2d&)vf;

  for ( uint j = 0; j < vf.m_uNVY ; j++ )   {
    o << "================================================================\n";
    for ( uint i = 0 ; i < vf.m_uNVX ; i++ ){
      o << std::setw(4) << vf.m_VData[i + j * vf.m_uNVX] << " ";
    }
    o << std::endl;
  }

  return o;
}


//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
void SJCVelocityField2d::
Write(std::ostream &o)
//============================================================================
{
  SJCField2d::Write(o);

  o.write((char*)m_VData, m_uNVX * m_uNVY * sizeof(SJCVector2d));
}

//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
bool SJCVelocityField2d::
Read(std::istream &i)
//============================================================================
{
  SJCField2d::Read(i);

  ReadCommon(i);
  return true;
}

//****************************************************************************
//
// * Set up the initial velocity condition
//============================================================================
void SJCVelocityField2d::
Set(SJCScalarField2D &pf)
//============================================================================
{
  for(uint  index_x = 1; index_x < m_uNX; index_x++) {
    for(uint index_y = 1; index_y < m_uNY; index_y ++) {
      SJCVector2d& data = m_VData[Index(index_x, index_y)];
      
      // Set up the x velocity
      if(index_y != m_uNY - 1) {
	float x = (float)index_x - 0.5;
	float y = (float)index_y ;
	SJCVector2f velocity = pf.curl(x, y);
	data.x((double)velocity.x());
      }
      else 
	data.x(0.f);

      // Set up the y velocity
      if(index_x != m_uNX - 1) {
	float x = (float)index_x;
	float y = (float)index_y - 0.5;
	SJCVector2f velocity = pf.curl(x, y);
	data.y((double)velocity.y());
      }
      else
	data.y(0.f);
    } // end of for index y
  } // end of index x

  SetBoundaryByCopy();
  
}


//****************************************************************************
//
// * Set up the boundary velocity by copy the nearest cell
//============================================================================
void SJCVelocityField2d::
SetBoundaryByCopy(void)
//============================================================================
{
  // Set up the left and right boundary by copy the next one
  for(uint index_y = 1; index_y < m_uNY - 1; index_y++) {

    // Left boundary
    m_VData[Index(0, index_y)] = m_VData[Index(1, index_y)];

    // Right boundary
    m_VData[Index(m_uNX, index_y)][0] = 
      m_VData[Index(m_uNX - 1, index_y)][0];

    m_VData[Index(m_uNX - 1, index_y)][1] =
      m_VData[Index(m_uNX - 2, index_y)][1];
  } // end of for for index_y

  // Set up the y boundary
  for(uint index_x = 1; index_x < m_uNX - 1; index_x++) {

    // bottom boundary
    m_VData[Index(index_x, 0)] = m_VData[Index(index_x, 1)];

    // Top boundary
    m_VData[Index(index_x, m_uNY - 1)][0] = 
      m_VData[Index(index_x, m_uNY - 2)][0];
    m_VData[Index(index_x, m_uNY)][1] = 
      m_VData[Index(index_x, m_uNY - 1)][1];
  } // end of for for index_y

  // Four corner
  // Left bottom
  m_VData[Index(0, 0)][0] = m_VData[Index(1, 1)][0];
  m_VData[Index(0, 0)][1] = m_VData[Index(1, 1)][1];

  // Left Top
  m_VData[Index(0, m_uNY - 1)][1] = m_VData[Index(0, m_uNY - 2)][1];
  m_VData[Index(0, m_uNY-1)][0]   = m_VData[Index(1, m_uNY - 2)][0];
  m_VData[Index(0, m_uNY)][1]     = m_VData[Index(1, m_uNY - 1)][1];

  // Right bottom
  m_VData[Index(m_uNX - 1, 0)][0] = m_VData[Index(m_uNX - 2, 1)][0];
  m_VData[Index(m_uNX, 0)][0]     = m_VData[Index(m_uNX - 1, 1)][0];
  m_VData[Index(m_uNX - 1, 0)][1] = m_VData[Index(m_uNX - 2, 1)][1];

  // Right Top
  m_VData[Index(m_uNX-1, m_uNY-1)][0] = m_VData[Index(m_uNX-2, m_uNY-1)][0];
  m_VData[Index(m_uNX-1, m_uNY-1)][1] = m_VData[Index(m_uNX-1, m_uNY-2)][1];
  m_VData[Index(m_uNX, m_uNY - 1)][0] = m_VData[Index(m_uNX - 1, m_uNY-2)][0];
  m_VData[Index(m_uNX - 1, m_uNY)][1] = m_VData[Index(m_uNX - 2, m_uNY-1)][1];
	
}


