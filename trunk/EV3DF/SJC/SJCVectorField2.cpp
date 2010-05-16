/************************************************************************
     Main File:

     File:        SJCVectorField2.h

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
  
#include "SJCVectorField2.h"
#include "SJCVelocityField2.h"

//****************************************************************************
//
// * The constructor
//============================================================================
SJCVectorField2d::
SJCVectorField2d(const uint        nx, 
		 const uint        ny, 
		 const double      dx,
		 const double      dy,
		 const SJCBoundary bx,
		 const SJCBoundary by,
		 const SJCVector2d *d)
  : SJCField2d(nx, ny, dx, dy, bx, by)
//============================================================================
{
  m_VMagnitute = 0;

  m_VData = new SJCVector2d[m_uNX * m_uNY];
  if (d) {
    for ( uint i = 0 ; i < m_uNX * m_uNY ; i++ )
      m_VData[i] = d[i];
  }
 
}

//****************************************************************************
//
// * Get the scalar field from input
//============================================================================
SJCVectorField2d::
SJCVectorField2d(std::istream &f)
  : SJCField2d(f)
//============================================================================
{
  m_VMagnitute = 0;
  m_VData = new SJCVector2d[m_uNX * m_uNY];
  f.read((char*)m_VData, m_uNX * m_uNY * sizeof(SJCVector2d));
}



//****************************************************************************
//
// * Copy operator
//============================================================================
SJCVectorField2d::SJCVectorField2d(const SJCVectorField2d &vf)
  : SJCField2d(vf)
//============================================================================
{
  Assign(vf);
}

//****************************************************************************
//
// * Copy contructor
//============================================================================
void SJCVectorField2d::
Assign(const SJCVectorField2d &vf)
//============================================================================
{
  SJCField2d::Assign(vf);
 
  m_VData = new SJCVector2d[m_uNX * m_uNY];
  for ( uint i = 0 ; i < m_uNX * m_uNY ; i++ )
    m_VData[i] = vf.m_VData[i];

  if(vf.m_VMagnitute){
    m_VMagnitute = new double[m_uNX * m_uNY];
    for ( uint i = 0 ; i < m_uNX * m_uNY ; i++ )
      m_VMagnitute[i] = vf.m_VMagnitute[i];
  }
}

//****************************************************************************
//
// * Destructor
//============================================================================
SJCVectorField2d::~SJCVectorField2d(void)
//============================================================================
{
  Destroy();
}


//****************************************************************************
//
// * Clear the data
//============================================================================
void SJCVectorField2d::
Destroy(void)
//============================================================================
{
  SJCField2d::Destroy();

  // Release the data
  delete [] m_VData;
  m_VData   = 0;

  if(m_VMagnitute)
    delete [] m_VMagnitute;
  m_VMagnitute = 0;
 
}

//****************************************************************************
//
// * Assign operator
//============================================================================
SJCVectorField2d& SJCVectorField2d::
operator=(const SJCVectorField2d &vf)
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
// * Set up the value
//============================================================================
void SJCVectorField2d::
Set(const uint nx, const uint ny, 
    const double dx, const double dy,
    const SJCBoundary bx, const SJCBoundary by,
    const SJCVector2d *d)  
//============================================================================
{
  Destroy();

  SJCField2d::Set(nx, ny, dx, dy, bx, by);
  m_VData = new SJCVector2d[m_uNX * m_uNY];
  if (d) {
    for ( uint i = 0 ; i < m_uNX * m_uNY ; i++ )
      m_VData[i] = d[i];
  }
  m_VMagnitute = 0;
}

//****************************************************************************
//
// * Set up the value from the velocity field
//============================================================================
void SJCVectorField2d::
Vorticity(SJCVelocityField2d* velocity)
//============================================================================
{
  // Check whether the magniture has been generated
  if(!m_VMagnitute){
    m_VMagnitute = new double [m_uNX * m_uNY];
    for( uint i = 0 ; i < m_uNX * m_uNY; i++){
      m_VMagnitute[i] = 0.f;
    }
  }

  // Go through all valid element 
  for(uint j = 1; j < m_uNY - 1; j++){
    for(uint i = 1; i < m_uNX - 1; i++){
      //      double vx = 0;
      //      double vy = 0;
      double vz = 
	((velocity->VoxelCenterVelocity(i + 1, j).y() - 
	  velocity->VoxelCenterVelocity(i - 1, j).y()) * .5 / m_dDX) -
	((velocity->VoxelCenterVelocity(i, j + 1).x() - 
	  velocity->VoxelCenterVelocity(i, j - 1).x()) * .5 / m_dDY);

      //      printf("%d %d %f %f %f %f %f\n", i, j, 
      //	     velocity->VoxelCenterVelocity(i + 1, j).y(), 
      //	     velocity->VoxelCenterVelocity(i - 1, j).y(),
      //	     velocity->VoxelCenterVelocity(i, j + 1).x(), 
      //	     velocity->VoxelCenterVelocity(i, j - 1).x(), vz);
      //      getchar();
      

      // Store in z
      m_VData[Index(i, j)].set(vz, 0.f);
      m_VMagnitute[Index(i, j)] = fabs(vz);
    } // end of for i
  } // end of for j
}

//****************************************************************************
//
// * Set up the force value for vorticity
//============================================================================
void SJCVectorField2d::
VorticityForce(SJCScalarField2d* bound, 
	       SJCVectorField2d* vorticity, const double vortCons)
//============================================================================
{
  // Go through all valid element
  for(uint j = 1; j < m_uNY - 1; j++){
    for(uint i = 1; i < m_uNX - 1; i++){

      if ((*bound)(i, j) == 0.f) {
	
	// Compute the Normal component which gradient of the 
	double x = (vorticity->Magnitute(i+1, j) - 
		    vorticity->Magnitute(i-1, j)) * 0.5 / m_dDX;
	double y = (vorticity->Magnitute(i ,j+1) - 
		    vorticity->Magnitute(i, j-1)) * 0.5 / m_dDY;
	
	// Compute the length
	double norm     = sqrt(x * x + y * y);
	if (norm > SJC_EPSILON) {
	  x /= norm;
	  y /= norm;
	  // Get the z component which is stored in x
	  double vort_mag = (*vorticity)(i, j).x();
	  m_VData[Index(i, j)].set( vortCons * y * vort_mag, 
				    -vortCons * x * vort_mag);
	}
	else {
	  m_VData[Index(i, j)].set( 0.f, 0.f);
	}
      } // end of if boudn
      else {
      	m_VData[Index(i, j)].set( 0.f, 0.f);
      }
    } // end of for i
  } // end of for j
}

//****************************************************************************
//
// * Get the value at x, y
//============================================================================
SJCVector2d SJCVectorField2d::
Value(const double x_pos, const double y_pos)
//============================================================================
{
  // Check whether the position is out of bound
  if(x_pos < MinX() || x_pos > MaxX() || 
     y_pos < MinY() || y_pos > MaxY()){
    SJCWarning("X or Y out of bound in getting value");
    return SJCVector2d(0.f, 0.f);
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

  SJCVector2d vBottomLeft  = m_VData[Index(index_xl, index_yb)];
  SJCVector2d vTopLeft     = m_VData[Index(index_xl, index_yt)];
  SJCVector2d vBottomRight = m_VData[Index(index_xr, index_yb)];
  SJCVector2d vTopRight    = m_VData[Index(index_xr, index_yt)];

  return (1.f-partial_x)*(1.f-partial_y)   * vBottomLeft + 
         partial_y      *(1.f - partial_x) * vTopLeft + 
         partial_x      *(1.f - partial_y) * vBottomRight + 
         partial_x      *partial_y         * vTopRight;
}



//****************************************************************************
//
// * Output operator
//============================================================================
std::ostream& operator <<(std::ostream &o, const SJCVectorField2d &vf)
//============================================================================
{
  o << (SJCField2d&)vf;

  for ( uint j = 0; j < vf.m_uNY ; j++ )   {
    for ( uint i = 0 ; i < vf.m_uNX ; i++ ){
      o << std::setw(8) << vf.m_VData[j * vf.m_uNX + i] << " ";
    }
    o << std::endl;
  }
  return o;

}


//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
void SJCVectorField2d::
Write(std::ostream &o)
//============================================================================
{
  SJCField2d::Write(o);

  o.write((char*)m_VData, m_uNX * m_uNY * sizeof(SJCVector2d));
}
