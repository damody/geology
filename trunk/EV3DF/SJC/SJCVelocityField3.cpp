/************************************************************************
     Main File:

     File:        SJCVelocityField3.h

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

#include "SJCVelocityField3.h"

const uint  SJCVelocityField3d::m_cuIntegralSteps = 20;

//*****************************************************************************
//
// * Contructor to set up everything
//============================================================================
SJCVelocityField3d::
SJCVelocityField3d(const uint nx,        const uint ny,       const uint nz,
		   const double dx,      const double dy,     const double dz,
		   const SJCBoundary bx, const SJCBoundary by, 
		   const SJCBoundary bz, const SJCVector3d *d)
  : SJCField3d(nx, ny, nx, dx, dy, dz, bx, by, bz)
//============================================================================
{
  m_uNVX         = nx + 1;
  m_uNVY         = ny + 1;
  m_uNVZ         = nz + 1;
  
  m_VData = new SJCVector3d[m_uNVX * m_uNVY * m_uNVZ];
  if (d) {
    for ( uint i = 0 ; i < m_uNVX * m_uNVY * m_uNVZ ; i++ )
      m_VData[i] = d[i];
  }
}

//*****************************************************************************
//
// * Contructor from scalar field
//============================================================================
SJCVelocityField3d::
SJCVelocityField3d(const SJCScalarField3d &pf)
  : SJCField3d(pf)
//============================================================================
{
  m_uNVX         = pf.NumX() + 1;
  m_uNVY         = pf.NumY() + 1;
  m_uNVZ         = pf.NumZ() + 1;

  m_VData = new SJCVector3d[m_uNX * m_uNY * m_uNZ];
  SJCWarning("No Data extraction implemnt here");
  
}

//*****************************************************************************
//
// * Copy contructor
//============================================================================
SJCVelocityField3d::
SJCVelocityField3d(const SJCVelocityField3d &vf)
  : SJCField3d(vf)
//============================================================================
{
  Assign(vf);
}

//****************************************************************************
//
// * Assign 
//============================================================================
void SJCVelocityField3d::
Assign(const SJCVelocityField3d &vf)
//============================================================================
{
  SJCField3d::Assign(vf);

  m_VData = new SJCVector3d[m_uNVX * m_uNVY * m_uNVZ];
  for ( uint i = 0 ; i < m_uNVX * m_uNVY * m_uNVZ ; i++ )
    m_VData[i] = vf.m_VData[i];
}

//*****************************************************************************
//
// * Destructor
//============================================================================
SJCVelocityField3d::
~SJCVelocityField3d(void)
//============================================================================
{
  Destroy();
}

//*****************************************************************************
//
// * Clear the data
//============================================================================
void SJCVelocityField3d::Destroy(void)
//============================================================================
{
  SJCField3d::Destroy();

  delete[] m_VData;
  m_VData   = 0;

}

//*****************************************************************************
//
// * Assign operator
//============================================================================
SJCVelocityField3d& SJCVelocityField3d::
operator=(const SJCVelocityField3d &vf)
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
SJCVector3d SJCVelocityField3d::
Value(const double x_pos, const double y_pos, const double z_pos)
//============================================================================
{  
  if(x_pos < MinX() || x_pos > MaxX() ||
     y_pos < MinY() || y_pos > MaxY() ||
     z_pos < MinZ() || z_pos > MaxZ()){
    SJCWarning("The position is out of bound in velocity field");
  }
  //***********************************************************************
  // Work on VX
  //***********************************************************************
  // Compute the index position
  double x = x_pos / m_dDX;
  double y = (y_pos - m_dHalfDY) / m_dDY;
  double z = (z_pos - m_dHalfDZ) / m_dDZ;

  uint	   index_xl, index_xr;  // the left and right index in x
  uint	   index_yb, index_yt;  // The top and bottom index in y
  uint	   index_zb, index_zf;  // The front and back index in z
  double   partial_x, partial_y, partial_z; // The partial in x, y, z

  // Here should have judgement when x_pos and y_pos are close to index 
  // position
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
  switch ( m_VBoundary[2] )   {
    case BOUNDARY_NOWRAP_FREE:
    case BOUNDARY_NOWRAP:
      if ( z >= m_uNZ - 1.f ) {
	index_zf  = (uint)m_uNZ - 1;
	index_zb  = index_zf - 1;
	partial_z = 1.0f;
      }
      else if ( z <= 0.f ) {
	index_zf  = 1;
	index_zb  = 0;
	partial_z = 0.f;
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

  double vLeftBottomBack   = m_VData[Index(index_xl, index_yb, index_zb)].x();
  double vLeftBottomFront  = m_VData[Index(index_xl, index_yb, index_zf)].x();
  double vLeftTopBack      = m_VData[Index(index_xl, index_yt, index_zb)].x();
  double vLeftTopFront     = m_VData[Index(index_xl, index_yt, index_zf)].x();
  double vRightBottomBack  = m_VData[Index(index_xr, index_yb, index_zb)].x();
  double vRightBottomFront = m_VData[Index(index_xr, index_yb, index_zf)].x();
  double vRightTopBack     = m_VData[Index(index_xr, index_yt, index_zb)].x();
  double vRightTopFront    = m_VData[Index(index_xr, index_yt, index_zf)].x();

  double vx =
         (1.f-partial_x)*(1.f-partial_y)*(1.f-partial_z) * vLeftBottomBack+
         (1.f-partial_x)*(1.f-partial_y)*partial_z       * vLeftBottomFront +
         (1.f-partial_x)*partial_y      *(1.f-partial_z) * vLeftTopBack +
         (1.f-partial_x)*partial_y      *partial_z       * vLeftTopFront +
         partial_x      *(1.f-partial_y)*(1.f-partial_z) * vRightBottomBack+
         partial_x      *(1.f-partial_y)*partial_z       * vRightBottomFront+
         partial_x      *partial_y      *(1.f-partial_z) * vRightTopBack +
         partial_x      *partial_y      * partial_z      * vRightTopFront ;

  //***********************************************************************
  // Work on VY
  //***********************************************************************
  // Here should have judgement when x_pos and y_pos are close to index 
  // position
  x = (x_pos - m_dHalfDX) / m_dDX;
  y = y_pos / m_dDY;
  z = (z_pos - m_dHalfDZ) / m_dDZ;
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

  switch ( m_VBoundary[2] )   {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      if ( z >= m_uNZ - 1.f ) {
	index_zf  = (uint)m_uNZ - 1;
	index_zb  = index_zf - 1;
	partial_z = 1.0f;
      }
      else if( z <= 0.f){
	index_zf  = 1;
	index_zb  = 0;
	partial_z = 0.f;

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

  vLeftBottomBack   = m_VData[Index(index_xl, index_yb, index_zb)].y();
  vLeftBottomFront  = m_VData[Index(index_xl, index_yb, index_zf)].y();
  vLeftTopBack      = m_VData[Index(index_xl, index_yt, index_zb)].y();
  vLeftTopFront     = m_VData[Index(index_xl, index_yt, index_zf)].y();
  vRightBottomBack  = m_VData[Index(index_xr, index_yb, index_zb)].y();
  vRightBottomFront = m_VData[Index(index_xr, index_yb, index_zf)].y();
  vRightTopBack     = m_VData[Index(index_xr, index_yt, index_zb)].y();
  vRightTopFront    = m_VData[Index(index_xr, index_yt, index_zf)].y();

  double vy =
         (1.f-partial_x)*(1.f-partial_y)*(1.f-partial_z) * vLeftBottomBack+
         (1.f-partial_x)*(1.f-partial_y)*partial_z       * vLeftBottomFront +
         (1.f-partial_x)*partial_y      *(1.f-partial_z) * vLeftTopBack +
         (1.f-partial_x)*partial_y      *partial_z       * vLeftTopFront +
         partial_x      *(1.f-partial_y)*(1.f-partial_z) * vRightBottomBack+
         partial_x      *(1.f-partial_y)*partial_z       * vRightBottomFront+
         partial_x      *partial_y      *(1.f-partial_z) * vRightTopBack +
         partial_x      *partial_y      * partial_z      * vRightTopFront ;
 
  //***********************************************************************
  // Work on VZ
  //***********************************************************************
  x = (x_pos - m_dHalfDX) / m_dDX;
  y = (y_pos - m_dHalfDY) / m_dDY;
  z = z_pos / m_dDZ;

  // Here should have judgement when x_pos and y_pos are close to index 
  // position
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
 
  switch ( m_VBoundary[2] )   {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      if ( z >= m_uNVZ - 1.f ) { // should never happen
	index_zf  = (uint)m_uNZ - 1;
	index_zb  = index_zf - 1;
	partial_z = 1.0f;
      }
      else if( z <= 0){
	index_zf  = 1;
	index_zb  = 0;
	partial_z = 0.f;
      }
      else {
	index_zb  = (uint)floor(z);
	index_zf  = index_zb + 1;
	partial_z = z - index_zb;
      }
      break;

    case BOUNDARY_WRAP: 
      double 	zp = fmod(z, (double)m_uNZ);
      if ( zp < 0.0 )
	zp = zp + (int)m_uNZ;
      
      index_zb  = (uint)floor(zp);
      index_zf  = index_zb + 1;
      partial_z = zp - (double)index_zb;
      break;
  }

  vLeftBottomBack   = m_VData[Index(index_xl, index_yb, index_zb)].z();
  vLeftBottomFront  = m_VData[Index(index_xl, index_yb, index_zf)].z();
  vLeftTopBack      = m_VData[Index(index_xl, index_yt, index_zb)].z();
  vLeftTopFront     = m_VData[Index(index_xl, index_yt, index_zf)].z();
  vRightBottomBack  = m_VData[Index(index_xr, index_yb, index_zb)].z();
  vRightBottomFront = m_VData[Index(index_xr, index_yb, index_zf)].z();
  vRightTopBack     = m_VData[Index(index_xr, index_yt, index_zb)].z();
  vRightTopFront    = m_VData[Index(index_xr, index_yt, index_zf)].z();

  double vz =
         (1.f-partial_x)*(1.f-partial_y)*(1.f-partial_z) * vLeftBottomBack+
         (1.f-partial_x)*(1.f-partial_y)*partial_z       * vLeftBottomFront +
         (1.f-partial_x)*partial_y      *(1.f-partial_z) * vLeftTopBack +
         (1.f-partial_x)*partial_y      *partial_z       * vLeftTopFront +
         partial_x      *(1.f-partial_y)*(1.f-partial_z) * vRightBottomBack+
         partial_x      *(1.f-partial_y)*partial_z       * vRightBottomFront+
         partial_x      *partial_y      *(1.f-partial_z) * vRightTopBack +
         partial_x      *partial_y      * partial_z      * vRightTopFront ;


  // Out of tiling bound.
  if(x_pos < 0 && x_pos >= m_dLengthX)
    vx = 0;
  if(y_pos < 0 && y_pos >= m_dLengthY)
    vy = 0;
  if(z_pos < 0 && z_pos >= m_dLengthZ)
    vz = 0;


  return SJCVector3d(vx, vy, vz);
}



//*****************************************************************************
//
// * Return the X component of i, j component
//============================================================================
SJCVector3d SJCVelocityField3d::
VoxelCenterVelocity(const uint index_x, const uint index_y, const uint index_z)
//============================================================================
{
   SJCVector3d center = m_VData[Index(index_x, index_y, index_z)];
   SJCVector3d right  = m_VData[Index(index_x + 1, index_y, index_z)];
   SJCVector3d top    = m_VData[Index(index_x, index_y + 1, index_z)];
   SJCVector3d front  = m_VData[Index(index_x, index_y, index_z + 1)];
   
   double vx = 0.5 * (center.x() + right.x());
   double vy = 0.5 * (center.y() + top.y());
   double vz = 0.5 * (center.z() + front.z());
   return SJCVector3d(vx, vy, vz);
}

//*****************************************************************************
//
// * Put the velocity at the center of the voxel
//============================================================================
void  SJCVelocityField3d:: 
AddForce(SJCScalarField3d* bound,
	 SJCVectorField3d* omega, const double dt)
//============================================================================
{
  for(uint k = 1; k < m_uNZ - 1; k++){
    for(uint j = 1; j < m_uNY - 1; j++){
      for(uint i = 1; i < m_uNX - 1; i++){

	SJCVector3d center = (*omega)(i, j, k);
	SJCVector3d left   = (*omega)(i-1, j, k);
	SJCVector3d bottom = (*omega)(i, j-1, k);
	SJCVector3d back   = (*omega)(i, j, k-1);
	SJCVector3d delta;

	// Left, bottom, back corner
	if(i == 1 && j == 1 && k == 1)
	  continue;
 
	// Left, bottom edge and it include the left top corner
	else if( i == 1 && j == 1){
	  delta.set( 0.f, 0.f, 0.5 * dt * (center.z() + back.z()));
	}

	// Left, back edge and it include the left top corner
	else if( i == 1 && k == 1){
	  delta.set( 0.f, 0.5 * dt * (center.y() + bottom.y()), 0.f);
	}

	// bottom, back edge and it include the left top corner
	else if( i == 1 && j == 1){
	  delta.set( 0.5 * dt * (center.x() + left.x()), 0.f, 0.f);
	}

	// Left face
	else if(i == 1){
	  delta.set( 0.f, 
		     0.5 * dt * (center.y() + bottom.y()), 
		     0.5 * dt * (center.z() + back.z()));
	}

	// Bottom face
	else if(j == 1){
	  delta.set( 0.5 * dt * (center.x() + left.x()), 
		     0.f, 
		     0.5 * dt * (center.z() + back.z()));
	}

	// Back face
	else if(k == 1){
	  delta.set( 0.5 * dt * (center.x() + left.x()), 
		     0.5 * dt * (center.y() + bottom.y()), 
		     0.f);
	}
	else {
	  delta.set(0.5 * dt * (center.x() + left.x()), 
		    0.5 * dt * (center.y() + bottom.y()),
		    0.5 * dt * (center.z() + back.z()) );
	}

	m_VData[Index(i, j, k)] += delta;
      } // end of i
    } // end of j
  }// end of k
}

//*****************************************************************************
//
// * Return the X component of i, j component
//============================================================================
double& SJCVelocityField3d::
VX(const uint index_x, const uint index_y, const uint index_z)
//============================================================================
{
  int ip, jp, kp;
 switch ( m_VBoundary[0] ) {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      ip = index_x;
    case BOUNDARY_WRAP:
      ip = index_x % m_uNVX;
      if ( ip < 0 ) 
	ip += m_uNVX;
  }
  
  switch (m_VBoundary[1]){
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      jp = index_y;
  
    case BOUNDARY_WRAP:
      jp = index_y % m_uNY;
      if ( jp < 0 ) 
	jp += m_uNY;
  }
 switch (m_VBoundary[2]){
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      kp = index_z;
  
    case BOUNDARY_WRAP:
      kp = index_z % m_uNZ;
      if ( kp < 0 ) 
	kp += m_uNZ;
  }
  return m_VData[Index(ip, jp, kp)][0];
 
}


//*****************************************************************************
//
// * Return the Y component of i, j element
//============================================================================
double& SJCVelocityField3d::
VY(const uint index_x, const uint index_y, const uint index_z)
//============================================================================
{
  int ip, jp, kp;
  switch ( m_VBoundary[0] ) {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      ip = index_x;
    case BOUNDARY_WRAP:
      ip = index_x % m_uNX;
      if ( ip < 0 ) 
	ip += m_uNX;
  }
  
  switch (m_VBoundary[1]){
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      jp = index_y;
  
    case BOUNDARY_WRAP:
      jp = index_y % m_uNVY;
      if ( jp < 0 ) 
	jp += m_uNVY;
  }

 switch (m_VBoundary[2]){
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      kp = index_z;
  
    case BOUNDARY_WRAP:
      kp = index_z % m_uNZ;
      if ( kp < 0 ) 
	kp += m_uNZ;
  }
  return m_VData[Index(ip, jp, kp)][1];
}

//*****************************************************************************
//
// * Return the Y component of i, j element
//============================================================================
double& SJCVelocityField3d::
VZ(const uint index_x, const uint index_y, const uint index_z)
//============================================================================
{
  int ip, jp, kp;
  switch ( m_VBoundary[0] ) {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      ip = index_x;
    case BOUNDARY_WRAP:
      ip = index_x % m_uNX;
      if ( ip < 0 ) 
	ip += m_uNX;
  }
  switch (m_VBoundary[1]){
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      jp = index_y;
  
    case BOUNDARY_WRAP:
      jp = index_y % m_uNY;
      if ( jp < 0 ) 
	jp += m_uNY;
  }

  switch (m_VBoundary[2]){
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
      kp = index_z;
  
    case BOUNDARY_WRAP:
      kp = index_z % m_uNVZ;
      if ( kp < 0 ) 
	kp += m_uNVZ;
  }
  return m_VData[Index(ip, jp, kp)][2];

}


//*****************************************************************************
//
// * Trace the particle back 
//============================================================================ 
void SJCVelocityField3d::
TraceParticle(const double dt, const SJCVector3d curr, SJCVector3d& prev)
//============================================================================ 
{
  // Cacluate the steps
  double step_dt = dt / (double)m_cuIntegralSteps;

  prev = curr;
  
  // Backward integrate to find the start position
  for (double time = dt; time <= 0; time -= step_dt) {
    SJCVector3d velocity = (*this)(prev);

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
    
    // Check whether hit the y boundary
    if ( prev[2] < BoundMinZ() )
      prev[2]= BoundMinZ();
    if ( prev[2] >= BoundMaxZ() )
      prev[2] = BoundMaxZ() - 1.0e-6;
  }
}

//*****************************************************************************
//
// * Get the divergence at voxel index (x,y) 
//============================================================================ 
double SJCVelocityField3d::
Divergence(const uint x, const uint y, const uint z)
//============================================================================ 
{
  double vx_left  = m_VData[Index(x, y, z)].x();
  double vx_right = m_VData[Index(x + 1, y, z)].x();
  
  double vy_bottom  = m_VData[Index(x, y, z)].y();
  double vy_top     = m_VData[Index(x, y + 1, z)].y();

  double vz_back    = m_VData[Index(x, y, z)].z();
  double vz_front   = m_VData[Index(x, y, z + 1)].z();
 
  return (vx_right - vx_left)   / m_dDX + 
         (vy_top   - vy_bottom) / m_dDY + 
	 (vz_front - vz_back)   / m_dDZ;
}



//*****************************************************************************
//
// * The output operator
//============================================================================
std::ostream& operator<<(std::ostream &o, const SJCVelocityField3d &vf)
//============================================================================
{
  o << (SJCField3d&)vf;

  for ( uint k = 0; k < vf.m_uNVZ ; k++ )   {
    for ( uint j = 0; j < vf.m_uNVY ; j++ )   {
      for ( uint i = 0 ; i < vf.m_uNVX ; i++ ){
	o << std::setw(8)
	  << vf.m_VData[i * vf.m_uNVY * vf.m_uNVZ + j * vf.m_uNVZ + k] << " ";
      }
    }
  }
  
  o << std::endl;

  return o;
}


//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
void SJCVelocityField3d::
Write(std::ostream &o)
//============================================================================
{
  SJCField3d::Write(o);

  o.write((char*)m_VData, m_uNVX * m_uNVY * m_uNVZ* sizeof(SJCVector3d));
}

