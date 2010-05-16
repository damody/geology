/************************************************************************
     Main File:

     File:        ScalarField2D.cpp

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     Class to handle the scalar field

     Constructors:
                  1. 0 : the default contructor
                  2. 4 : constructor to set up all value by input parameters
                  3. 1 : set up the class by using the scalar field
                  4. 1 : copy contructor
                   
     Functions:  what r for?
                 1. = : Assign operator which copy the parameter of random
                 2. (): Get the value of the scalar field
                 3. value: get the value of the scalar field
                 4. grad: get the gradient of the scalar field
                 5. curl: get the curl of the scalar field
                 6. MinX, MinY, MaxX, MaxY: get the maximum and minimum value 
                    of X, y
                 7. DiffMaxX, DiffMaxY, DiffMinX, DiffMinY: get the max and
                    min value of X, Y
                 8. NumX, NumY: get the number of sample points in X, Y
                 9. Write: write out the data into stream in binary form
                10. >>: output in the ascii form

************************************************************************/

#include <SJCScalarField2D.h>
#include <iomanip>
#include <assert.h>


//****************************************************************************
//
// * Contructor to set up all value
//============================================================================
SJCScalarField2D::SJCScalarField2D(const uint x, const uint y,
			     	   const Boundary bx, const Boundary by,
			     	   const float *d)
//============================================================================
{
  bound_cond[0] = bx;
  bound_cond[1] = by;
  nx = x;
  ny = y;
  data = new float[nx * ny];

  if ( d )   {
    for ( uint i = 0 ; i < nx * ny ; i++ )
      data[i] = d[i];
  }
}


//****************************************************************************
//
// * Get the scalar field from input
//============================================================================
SJCScalarField2D::SJCScalarField2D(std::istream &f)
//============================================================================
{
  data = 0;
  Read(f);
}



//****************************************************************************
//
// * Copy contructor
//============================================================================
SJCScalarField2D::SJCScalarField2D(const SJCScalarField2D &vf)
//============================================================================
{
  bound_cond[0] = vf.bound_cond[0];
  bound_cond[1] = vf.bound_cond[1];
  nx = vf.nx;
  ny = vf.ny;
  
  data = new float[nx * ny];
  
  for ( uint i = 0 ; i < nx * ny ; i++ )
    data[i] = vf.data[i];
}



//****************************************************************************
//
// * Destructor
//============================================================================
SJCScalarField2D::~SJCScalarField2D(void)
//============================================================================
{
  Destroy();
}


//****************************************************************************
//
// * Assign operator
//============================================================================
SJCScalarField2D&
SJCScalarField2D::operator=(const SJCScalarField2D &vf)
//============================================================================
{
  Destroy();
  
  bound_cond[0] = vf.bound_cond[0];
  bound_cond[1] = vf.bound_cond[1];
  nx = vf.nx;
  ny = vf.ny;
  data = new float[nx * ny];
  
  for ( uint i = 0 ; i < nx * ny ; i++ )
    data[i] = vf.data[i];
  
  return *this;
}


//****************************************************************************
//
// * Get the minimum differential value in x direction
//============================================================================
float   
SJCScalarField2D::DiffMinX(void) const
//============================================================================
{
  switch ( bound_cond[0] )   {
    case BOUNDARY_NOWRAP: 
      return 0.5f;
    case BOUNDARY_WRAP: 
      return 0.0f;
  }

  return 0.0f;
}

//****************************************************************************
//
// * Return the minimum differential value in y direction
//============================================================================
float
SJCScalarField2D::DiffMinY(void) const
//============================================================================
{
  switch ( bound_cond[1] ) {
    case BOUNDARY_NOWRAP: 
      return 0.5f;
    case BOUNDARY_WRAP: return 0.0f;
  }

  return 0.0f;
}


//****************************************************************************
//
// * Return the differential maximum value in x direction
//============================================================================
float   
SJCScalarField2D::DiffMaxX(void) const
//============================================================================
{
  switch ( bound_cond[0] )    {
    case BOUNDARY_NOWRAP: 
      return nx - 1.5f;
    case BOUNDARY_WRAP: 
      return (float)nx;
  }

  return 0.0f;
}

//****************************************************************************
//
// * Return the differential maximum value in y direction
//============================================================================
float
SJCScalarField2D::DiffMaxY(void) const
//============================================================================
{
  switch ( bound_cond[1] )    {
    case BOUNDARY_NOWRAP: 
      return ny - 1.5f;
    case BOUNDARY_WRAP: 
      return (float)ny;
  }

  return 0.0f;
}

//****************************************************************************
//
// * According to the boundary get the value at x, y
//   Bilinear interpolation
//============================================================================
float
SJCScalarField2D::value(const float x, const float y)
//============================================================================
{
  int	  il, ir;
  int	  jb, jt;
  float   a, b;
  
  switch ( bound_cond[0] )   { // Check and set up the correct value for x,y 
                          // according the boundary conditions
    case BOUNDARY_NOWRAP: {
      assert( x >= 0.0f && x <= nx - 1.0f );
      
      if ( x == nx - 1.0f ) {
	ir = (int)nx - 1;
	il = ir - 1;
	a = 1.0f;
      }
      else	{
	il = (int)floor(x);
	a = x - il;
	ir = (int)floor(x + 1.0);
      }
    } break;

    case BOUNDARY_WRAP: {
      float	xp = fmod(x, (float)nx);
      if ( xp < 0.0 )
	xp = xp + (int)nx;
      
      il = (int)floor(xp);
      a = xp - il;
      ir = (int)floor(xp + 1.0);
    } break;
  }

  switch ( bound_cond[1] )   {
    case BOUNDARY_NOWRAP: {
      assert( y >= 0.0f && y <= ny - 1.0f );
      
      if ( y == ny - 1.0f ) {
	jt = (int)ny - 1;
	jb = jt - 1;
	b = 1.0f;
      }
      else	{
	jb = (int)floor(y);
	b = y - jb;
	jt = (int)floor(y + 1.0);
      }
    } break;

    case BOUNDARY_WRAP: {
      float 	yp = fmod(y, (float)ny);
      if ( yp < 0.0 )
	yp = yp + (int)ny;
      
      jb = (int)floor(yp);
      b = yp - jb;
      jt = (int)floor(yp + 1.0);
    }
  }

  float   dtl, dtr, dbr;
				      
  if ( ir < (int)nx && jt < (int)ny )   {
    dtl = data[il * ny + jt];
    dtr = data[ir * ny + jt];
    dbr = data[ir * ny + jb];
  }
  else if ( ir == (int)nx && jt == (int)ny )   {
    dtl = data[il*ny+ny-1] + ( data[il*ny+1] - data[il*ny] );
    dbr = data[(nx-1)*ny+jb] + ( data[ny+jb] - data[jb] );
    dtr = data[(nx-1)*ny+ny-1] + ( data[ny+1] - data[0] );
  }
  else if ( ir == (int)nx )    {
    dtl = data[il * ny + jt];
    dbr = data[(nx-1)*ny+jb] + ( data[ny+jb] - data[jb] );
    dtr = data[(nx-1)*ny+jt] + ( data[ny+jt] - data[jt] );
  }
  else { // jt = ny    
    dtl = data[il*ny+ny-1] + ( data[il*ny+1] - data[il*ny] );
    dbr = data[ir * ny + jb];
    dtr = data[ir*ny+ny-1] + ( data[ir*ny+1] - data[ir*ny] );
  }
  
  return ( 1.0f - a - b + a * b ) * data[il * ny + jb] + 
         ( b - a * b ) * dtl + 
         ( a - a * b ) * dbr + 
         a * b * dtr;
}


//****************************************************************************
//
// * Get the value of the x, y
//   r is the rotation, r = 0 => rotate 0 deg, r = 1 => rotate 90 deg about
//   the left bottom corner.
//============================================================================
float
SJCScalarField2D::value(const float x, const float y, const unsigned char r)
//============================================================================
{
  float   xr;
  float   yr;

  if ( r <= 1 )   {
    if ( r < 1 ) {
      xr = x;
      yr = y;
    }
    else {
      xr = y;
      yr = ny - 1.0f - x;
    }
  }
  else    {
    if ( r < 3 ) {
      xr = nx - 1.0f - x;
      yr = ny - 1.0f - y;
    }
    else {
      xr = nx - 1.0f - y;
      yr = x;
    }
  }

  return value(xr, yr);
}


//****************************************************************************
//
// * Access the value at (x, y)
//============================================================================
float&
SJCScalarField2D::operator()(const uint x, const uint y)
//============================================================================
{
  assert( x >= 0 && x < nx && y >= 0 && y < ny );

  return data[x * ny + y];
}


//****************************************************************************
//
// * Compute the gradient of the scalar field
//   dx = p(x + 0.5, y) - p(x - 0.5, y) 
//   dy = p(x, y + 0.5) - p(x, y - 0.5)
//============================================================================
SJCVector2f
SJCScalarField2D::grad(const float x, const float y)
//============================================================================
{
  float   pl = value(x - 0.5, y);
  float   pr = value(x + 0.5, y);
  float   pt = value(x, y + 0.5);
  float   pb = value(x, y - 0.5);

  return SJCVector2f(pr - pl, pt - pb);
}


//****************************************************************************
//
// * Compute the gradient
//   r is use to set up the boundary but how?
//============================================================================
SJCVector2f
SJCScalarField2D::grad(const float x, const float y, const unsigned char r)
//============================================================================
{
  float   xr;
  float   yr;

  if ( r <= 1 )   {
    if ( r < 1 ){
      xr = x;
      yr = y;
    }
    else {
      xr = y;
      yr = ny - 1.0f - x;
    }
  }
  else   {
    if ( r < 3 ) {
      xr = nx - 1.0f - x;
      yr = ny - 1.0f - y;
    }
    else {
      xr = nx - 1.0f - y;
      yr = x;
    }
  }

  float   pl = value(xr - 0.5, yr);
  float   pr = value(xr + 0.5, yr);
  float   pt = value(xr, yr + 0.5);
  float   pb = value(xr, yr - 0.5);
  
  return SJCVector2f(pr - pl, pt - pb);
}


//****************************************************************************
//
// * Compute the curl of the scalar field
//   dx = p(x, y + 0.5) - p(x, y - 0.5)
//   dy = p(x + 0.5, y) - p(x - 0.5, y) 
//============================================================================
SJCVector2f
SJCScalarField2D::curl(const float x, const float y)
//============================================================================
{
  float   pl = value(x - 0.5, y);
  float   pr = value(x + 0.5, y);
  float   pt = value(x, y + 0.5);
  float   pb = value(x, y - 0.5);
  
  return SJCVector2f(pt - pb, pl - pr);
}


//****************************************************************************
//
// * Compute the curl of the system
//   r for boundary check but how?
//============================================================================
SJCVector2f
SJCScalarField2D::curl(const float x, const float y, const unsigned char r)
//============================================================================
{
  float   xr;
  float   yr;

  if ( r <= 1 )   {
    if ( r < 1 ){
      xr = x;
      yr = y;
    }
    else     {
      xr = y;
      yr = ny - 1.0f - x;
    }
  }
  else   {
    if ( r < 3 )	{
      xr = nx - 1.0f - x;
      yr = ny - 1.0f - y;
    }
    else {
      xr = nx - 1.0f - y;
      yr = x;
    }
  }

  float   pl = value(xr - 0.5, yr);
  float   pr = value(xr + 0.5, yr);
  float   pt = value(xr, yr + 0.5);
  float   pb = value(xr, yr - 0.5);
  
  return SJCVector2f(pt - pb, pl - pr);
}

//****************************************************************************
//
// * Clear the data
//============================================================================
void
SJCScalarField2D::Destroy(void)
//============================================================================
{
  delete[] data;
  nx = ny = 0;
  data = 0;
}


//****************************************************************************
//
// * Output operator
//============================================================================
std::ostream&
operator<<(std::ostream &o, const SJCScalarField2D &vf)
//============================================================================
{
  o << std::setw(4) << vf.nx << " " << std::setw(4) << vf.ny << " ";
  o << vf.bound_cond << std::endl;
  for ( int j = vf.ny - 1 ; j >= 0 ; j-- )   {
    for ( uint i = 0 ; i < vf.nx ; i++ ){
      o << std::setw(8) << vf.data[i * vf.ny + j] << " ";
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
SJCScalarField2D::Write(std::ostream &o)
//============================================================================
{
  o.write((char*)&nx, sizeof(uint));
  o.write((char*)&ny, sizeof(uint));
  o.write((char*)&(bound_cond[0]), sizeof(Boundary));
  o.write((char*)&(bound_cond[1]), sizeof(Boundary));
  o.write((char*)data, nx * ny * sizeof(float));
}

//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
bool SJCScalarField2D::
Read(std::istream &in)
//============================================================================
{
  in.read((char*)&nx, sizeof(uint));
  in.read((char*)&ny, sizeof(uint));

  in.read((char*)&(bound_cond[0]), sizeof(Boundary));
  in.read((char*)&(bound_cond[1]), sizeof(Boundary));
 
  if(data)
    delete [] data;

  data = new float[ nx * ny];
  
  in.read((char*)data, nx * ny * sizeof(float));
  return true;
  
}


