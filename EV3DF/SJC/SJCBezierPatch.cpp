/************************************************************************
     Main File:

     File:        BezierPatch.cpp

     Author:     
                  Steven Chenney, schenney@cs.wisc.edu
     Modifier:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
  
     Comment:     The random variable operation

     Constructors:
                  1. 1 : Set up the idum and calculate the following
                  2. 5 : Set up all preset variable
                  3. 1 : Set up assignment
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2. <<: output operator
************************************************************************/

#include <SJCBezierPatch.h>
#include <SJCException.h>
#include <math.h>

static int  LU_Decomp(double **a, int n, int *indx, int *d, double *vv);
static void LU_Back_Subst(double **a, int n, int *indx, double *b);

//***************************************************************************
//
// * Default constructor
//===========================================================================
SJCBezierPatch::
SJCBezierPatch(void)
//===========================================================================
{
  // Reset everything to zero
  int	i, j;
  
  d = 0;
  
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )
      c_pts[i][j] = NULL;
}


//***************************************************************************
//
// * Initializes with the given dimension and control points.
//===========================================================================
SJCBezierPatch::
SJCBezierPatch(const unsigned short dim, float * const c_in[4][4])
//===========================================================================
{
  int	i, j;

  // Set up the dimension
  d = dim;
  // Set the cotnrol point to null
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )
      c_pts[i][j] = NULL;
  
  Copy_Controls(c_in);
}

//***************************************************************************
//
// * Constructor that takes 16 points on the patch and their coresponding
//   parametric values and then solve the least square fit problem
//===========================================================================
SJCBezierPatch::
SJCBezierPatch(const unsigned short dim,
	       float u[16], float v[16], float *p[16])
//===========================================================================
{

  double  *m[16];
  for ( uint i = 0 ; i < 16 ; i++ )
    m[i] = new double[16];

  int	  index[16];
  double  vv[16];
  double  b[16];
  float   ub[4];
  float   vb[4];
  
  d = dim;
  
  // Allocate data for the control points
  for ( uint i = 0 ; i < 4 ; i++ )
    for ( uint j = 0 ; j < 4 ; j++ )
      c_pts[i][j] = new float[d];
  
  // Evaluate the basis value of the ub, vb for the coefficients and then
  // Compute the coefficient for that points
  for ( uint l = 0 ; l < d ; l++ )   {
    for ( uint k = 0 ; k < 16 ; k++ ){
      Evaluate_Basis(u[k], ub);
      Evaluate_Basis(v[k], vb);

      for ( uint i = 0 ; i < 4 ; i++ )
	for ( uint j = 0 ; j < 4 ; j++ )	{
	  m[k][i * 4 + j] = ub[i] * vb[j];
	}
      b[k] = p[k][l];
    }// end of k

    int	flip;
    LU_Decomp(m, 16, index, &flip, vv);
    LU_Back_Subst(m, 16, index, b);
    
    for ( uint i = 0 ; i < 4 ; i++ )
      for ( uint j = 0 ; j < 4 ; j++ )
	c_pts[i][j][l] = b[i * 4 + j];
  }

  for ( uint i = 0 ; i < 16 ; i++ )
    delete m[i];
}


//***************************************************************************
//
// * Destructor
//===========================================================================
SJCBezierPatch::
~SJCBezierPatch(void)
//===========================================================================
{
  Delete_Controls();
}


//***************************************************************************
//
// * Copy operator.
//===========================================================================
SJCBezierPatch&
SJCBezierPatch::operator=(const SJCBezierPatch &src)
//===========================================================================
{
  if ( this != &src )   {
    d = src.d;
    Copy_Controls(src.c_pts);
  }
  
  return *this;
}


//***************************************************************************
//
// * Query a control point, putting the value into the given array, pt.
//   Throws an exception if the index is out of range.
//===========================================================================
void SJCBezierPatch::
C(const unsigned short s, const unsigned short t, float *pt)
//===========================================================================
{
  int i;

  if ( s >= 4 && t >= 4 )
    throw new SJCException("SJCBezierPatch::C - Index out of range");

  for ( i = 0 ; i < d ; i++ )
    pt[i] = c_pts[s][t][i];
}


//***************************************************************************
//
// * Change a control point at the given position.
//   Will throw an exception if the position is out of range
//===========================================================================
void SJCBezierPatch:: 
Set_Control(const float *pt, const unsigned short s,
	    const unsigned short t)
//===========================================================================
{
  int i;
  
  if ( s >= 4 || t >= 4 )
    throw new SJCException("SJCBezierPatch::Set_Control - Posn out of range");
  
  for ( i = 0 ; i < d ; i++ )
    c_pts[s][t][i] = pt[i];
}

 
//***************************************************************************
//
// * According to the u to evaluate the 4 basis contribution
//===========================================================================
void SJCBezierPatch::
Evaluate_Basis(const float u, float vals[4])
//===========================================================================
{
  float u_sq = u * u;
  float u_cube = u * u_sq;
  
  vals[0] =        -u_cube + 3.0f * u_sq - 3.0f * u + 1.0f;
  vals[1] =  3.0f * u_cube - 6.0f * u_sq + 3.0f * u;
  vals[2] = -3.0f * u_cube + 3.0f * u_sq;
  vals[3] =         u_cube;
}


//***************************************************************************
//
// * Evaluate the curve at a parameter value and copy the result into
//   the given array. Throws an exception if the parameter is out of
//   range, unless the curve is a loop
//===========================================================================
void SJCBezierPatch::
Evaluate_Point(const float s, const float t, float *pt)
//===========================================================================
{
  float   s_basis[4];
  float   t_basis[4];
  int     i, j, k;
  
  if ( s < 0 || s > 1 || t < 0 || t > 1 )   {
    throw new SJCException(
	    "SJCBezierPatch::EvaluatePoint - Parameter value out of range");
    }

  Evaluate_Basis(s, s_basis);
  Evaluate_Basis(t, t_basis);
  
  for ( k = 0 ; k < d ; k++ )
    pt[k] = 0.0f;
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )
      for ( k = 0 ; k < d ; k++ )
	pt[k] += c_pts[i][j][k] * s_basis[i] * t_basis[j];
}


//***************************************************************************
//
// * Evaluate the derivative at a parameter value and copy the result into
//   the given array. Throws an exception if the parameter is out of
//   range, unless the curve is a loop.
//===========================================================================
void SJCBezierPatch::
Evaluate_Dx_Ds(const float s, const float t, float *deriv)
//===========================================================================
{
  float   s_sq;
  float   t_sq;
  float   t_cube;
  float   s_basis[4];
  float   t_basis[4];
  int     i, j, k;
  
  if ( s < 0 || s > 1 || t < 0 || t > 1 )   {
    throw new SJCException(
	  "SJCBezierPatch::EvaluateDerivative - Parameter value out of range");
  }

  s_sq = s * s;
  t_sq = t * t;
  t_cube = t * t_sq;
  
  s_basis[0] = -3.0f * s_sq + 6.0f * s - 3.0f;
  s_basis[1] = 9.0f * s_sq - 12.0f * s + 3.0f;
  s_basis[2] = -9.0f * s_sq + 6.0f * s;
  s_basis[3] = 3.0f * s_sq;
  
  t_basis[0] = -t_cube + 3.0f * t_sq - 3.0f * t + 1.0f;
  t_basis[1] = 3.0f * t_cube - 6.0f * t_sq + 3.0f * t;
  t_basis[2] = -3.0f * t_cube + 3.0f * t_sq;
  t_basis[3] = t_cube;
  
  for ( k = 0 ; k < d ; k++ )
    deriv[k] = 0.0f;
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )
      for ( k = 0 ; k < d ; k++ )
	deriv[k] += c_pts[i][j][k] * s_basis[i] * t_basis[j];
}


//***************************************************************************
//
// * Evaluate the derivative at a parameter value and copy the result into
//   the given array. Throws an exception if the parameter is out of
//   range, unless the curve is a loop.
//===========================================================================
void SJCBezierPatch::
Evaluate_Dx_Dt(const float s, const float t, float *deriv)
//===========================================================================
{
  float   s_sq;
  float   s_cube;
  float   t_sq;
  float   s_basis[4];
  float   t_basis[4];
  int     i, j, k;
  
  if ( s < 0 || s > 1 || t < 0 || t > 1 )  {
    throw new SJCException(
	  "SJCBezierPatch::EvaluateDerivative - Parameter value out of range");
  }

  s_sq = s * s;
  s_cube = s * s_sq;
  t_sq = t * t;
  
  s_basis[0] = -s_cube + 3.0f * s_sq - 3.0f * s + 1.0f;
  s_basis[1] = 3.0f * s_cube - 6.0f * s_sq + 3.0f * s;
  s_basis[2] = -3.0f * s_cube + 3.0f * s_sq;
  s_basis[3] = s_cube;
  
  t_basis[0] = -3.0f * t_sq + 6.0f * t - 3.0f;
  t_basis[1] = 9.0f * t_sq - 12.0f * t + 3.0f;
  t_basis[2] = -9.0f * t_sq + 6.0f * t;
  t_basis[3] = 3.0f * t_sq;
  
  for ( k = 0 ; k < d ; k++ )
    deriv[k] = 0.0f;
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )
      for ( k = 0 ; k < d ; k++ )
	deriv[k] += c_pts[i][j][k] * s_basis[i] * t_basis[j];
}


//***************************************************************************
//
// * Refine the curve, putting the result into the given curve. This
//   will correctly account for looped curves.
//===========================================================================
void SJCBezierPatch::
Refine(SJCBezierPatch &r00, SJCBezierPatch &r01,
       SJCBezierPatch &r10, SJCBezierPatch &r11)
//===========================================================================
{

  float   *temp_c[4][7];
  float   *new_c[7][7];
  float   *temp;
  int	    i, j, k;
  
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 7 ; j++ )
      temp_c[i][j] = new float[d];
  for ( i = 0 ; i < 7 ; i++ )
    for ( j = 0 ; j < 7 ; j++ )
      new_c[i][j] = new float[d];
  temp = new float[d];
  
  for ( i = 0 ; i < 4 ; i++ )   {
    for ( k = 0 ; k < d ; k++ ) {
      temp_c[i][0][k] = c_pts[i][0][k];
      temp_c[i][6][k] = c_pts[i][3][k];
      temp_c[i][1][k] = ( c_pts[i][0][k] + c_pts[i][1][k] ) * 0.5f;
      temp_c[i][5][k] = ( c_pts[i][2][k] + c_pts[i][3][k] ) * 0.5f;
      temp[k] = ( c_pts[i][1][k] + c_pts[i][2][k] ) * 0.5f;
      temp_c[i][2][k] = ( temp_c[i][1][k] + temp[k] ) * 0.5f;
      temp_c[i][4][k] = ( temp_c[i][5][k] + temp[k] ) * 0.5f;
      temp_c[i][3][k] = ( temp_c[i][2][k] + temp_c[i][4][k] ) * 0.5f;
    }
  }
  
  for ( i = 0 ; i < 7 ; i++ )   {
    for ( k = 0 ; k < d ; k++ ) {
      new_c[0][i][k] = temp_c[0][i][k];
      new_c[6][i][k] = temp_c[3][i][k];
      new_c[1][i][k] = ( temp_c[0][i][k] + temp_c[1][i][k] ) * 0.5f;
      new_c[5][i][k] = ( temp_c[2][i][k] + temp_c[3][i][k] ) * 0.5f;
      temp[k] = ( temp_c[1][i][k] + temp_c[2][i][k] ) * 0.5f;
      new_c[2][i][k] = ( new_c[1][i][k] + temp[k] ) * 0.5f;
      new_c[4][i][k] = ( new_c[5][i][k] + temp[k] ) * 0.5f;
      new_c[3][i][k] = ( new_c[2][i][k] + new_c[4][i][k] ) * 0.5f;
    }
  }

  r00.Delete_Controls();
  r01.Delete_Controls();
  r10.Delete_Controls();
  r11.Delete_Controls();
  
  r00.d = d;
  r01.d = d;
  r10.d = d;
  r11.d = d;
  
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ ) {
      r00.c_pts[i][j] = new float[d];
      for ( k = 0 ; k < d ; k++ )
	r00.c_pts[i][j][k] = new_c[i][j][k];
      r01.c_pts[i][j] = new float[d];
      for ( k = 0 ; k < d ; k++ )
	r01.c_pts[i][j][k] = new_c[i][j+3][k];
      r10.c_pts[i][j] = new float[d];
      for ( k = 0 ; k < d ; k++ )
	r10.c_pts[i][j][k] = new_c[i+3][j][k];
      r11.c_pts[i][j] = new float[d];
      for ( k = 0 ; k < d ; k++ )
	r11.c_pts[i][j][k] = new_c[i+3][j+3][k];
    }

  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 7 ; j++ )
      delete[] temp_c[i][j];
  for ( i = 0 ; i < 7 ; i++ )
    for ( j = 0 ; j < 7 ; j++ )
      delete[] new_c[i][j];
  delete[] temp;
}


//***************************************************************************
//
// * Refine the curve, putting the result into the given curve. This
//   will correctly account for looped curves.
//===========================================================================
void SJCBezierPatch::
Refine_T(SJCBezierPatch &r0, SJCBezierPatch &r1)
//===========================================================================
{
  
  float   *new_c[4][7];
  float   *temp;
  int	    i, j, k;
  
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 7 ; j++ )
      new_c[i][j] = new float[d];
  temp = new float[d];

  for ( i = 0 ; i < 4 ; i++ )   {
    for ( k = 0 ; k < d ; k++ ) {
      new_c[i][0][k] = c_pts[i][0][k];
      new_c[i][6][k] = c_pts[i][3][k];
      new_c[i][1][k] = ( c_pts[i][0][k] + c_pts[i][1][k] ) * 0.5f;
      new_c[i][5][k] = ( c_pts[i][2][k] + c_pts[i][3][k] ) * 0.5f;
      temp[k] = ( c_pts[i][1][k] + c_pts[i][2][k] ) * 0.5f;
      new_c[i][2][k] = ( new_c[i][1][k] + temp[k] ) * 0.5f;
      new_c[i][4][k] = ( new_c[i][5][k] + temp[k] ) * 0.5f;
      new_c[i][3][k] = ( new_c[i][2][k] + new_c[i][4][k] ) * 0.5f;
    }
  }

  r0.Delete_Controls();
  r1.Delete_Controls();
  
  r0.d = d;
  r1.d = d;
  
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )	{
      r0.c_pts[i][j] = new float[d];
      for ( k = 0 ; k < d ; k++ )
	r0.c_pts[i][j][k] = new_c[i][j][k];
      r1.c_pts[i][j] = new float[d];
      for ( k = 0 ; k < d ; k++ )
	r1.c_pts[i][j][k] = new_c[i][j+3][k];
    }
  
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 7 ; j++ )
      delete[] new_c[i][j];
  delete[] temp;

}

//***************************************************************************
//
// * Refine the curve, putting the result into the given curve. This
//   will correctly account for looped curves.
//===========================================================================
void SJCBezierPatch::
Refine_S(SJCBezierPatch &r0, SJCBezierPatch &r1)
//===========================================================================
{

  float   *new_c[7][4];
  float   *temp;
  int	    i, j, k;
    
  for ( i = 0 ; i < 7 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )
      new_c[i][j] = new float[d];
  temp = new float[d];

  for ( i = 0 ; i < 4 ; i++ )    {
    for ( k = 0 ; k < d ; k++ )	{
      new_c[0][i][k] = c_pts[0][i][k];
      new_c[6][i][k] = c_pts[3][i][k];
      new_c[1][i][k] = ( c_pts[0][i][k] + c_pts[1][i][k] ) * 0.5f;
      new_c[5][i][k] = ( c_pts[2][i][k] + c_pts[3][i][k] ) * 0.5f;
      temp[k] = ( c_pts[1][i][k] + c_pts[2][i][k] ) * 0.5f;
      new_c[2][i][k] = ( new_c[1][i][k] + temp[k] ) * 0.5f;
      new_c[4][i][k] = ( new_c[5][i][k] + temp[k] ) * 0.5f;
      new_c[3][i][k] = ( new_c[2][i][k] + new_c[4][i][k] ) * 0.5f;
    }
  }
  
  r0.Delete_Controls();
  r1.Delete_Controls();
  
  r0.d = d;
  r1.d = d;

  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )     {
      r0.c_pts[i][j] = new float[d];
      for ( k = 0 ; k < d ; k++ )
	r0.c_pts[i][j][k] = new_c[i][j][k];
      r1.c_pts[i][j] = new float[d];
      for ( k = 0 ; k < d ; k++ )
	r1.c_pts[i][j][k] = new_c[i+3][j][k];
    }

  for ( i = 0 ; i < 7 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )
      delete[] new_c[i][j];
  delete[] temp;
}


//***************************************************************************
//
// * T direction's error must be within limit
//===========================================================================
bool SJCBezierPatch::
Within_Tol_T(const float tol)
//===========================================================================
{
  float   *p1;
  float   *p2;
  float   *x1_x0;
  float   *x2_x0;
  float   *x3_x0;
  float   l_03, l_1p, l_2p, dot1, dot2;
  int     i, j;
  
  p1 = new float[d];
  p2 = new float[d];
  x1_x0 = new float[d];
  x2_x0 = new float[d];
  x3_x0 = new float[d];
  
  for ( i = 0 ; i < 4 ; i++ )   {
    dot2 = 0.0f;
    dot1 = 0.0f;
    l_03 = 0.0f;
    for ( j = 0 ; j < d ; j++ ) {
      x1_x0[j] = c_pts[i][1][j] - c_pts[i][0][j];
      x2_x0[j] = c_pts[i][2][j] - c_pts[i][0][j];
      x3_x0[j] = c_pts[i][3][j] - c_pts[i][0][j];
      dot1 += ( x1_x0[j] * x3_x0[j] );
      dot2 += ( x2_x0[j] * x3_x0[j] );
      l_03 += ( x3_x0[j] * x3_x0[j] );
    }
    if ( l_03 == 0.0f )
      continue;
    l_1p = 0.0f;
    l_2p = 0.0f;
    for ( j = 0 ; j < d ; j++ )	{
      p1[j] = c_pts[i][0][j] + dot1 * x3_x0[j] / l_03;
      p2[j] = c_pts[i][0][j] + dot2 * x3_x0[j] / l_03;
      l_1p += ( c_pts[i][1][j] - p1[j] ) * ( c_pts[i][1][j] - p1[j] );
      l_2p += ( c_pts[i][2][j] - p2[j] ) * ( c_pts[i][2][j] - p2[j] );
    }
    if ( l_1p > tol * tol || l_2p > tol * tol )	{
      delete[] p1;
      delete[] p2;
      delete[] x1_x0;
      delete[] x2_x0;
      delete[] x3_x0;
      return false;
    }
  }

  delete[] p1;
  delete[] p2;
  delete[] x1_x0;
  delete[] x2_x0;
  delete[] x3_x0;

  return true;
}


//***************************************************************************
//
// * S direction's error must be within thereshold
//===========================================================================
bool SJCBezierPatch::
Within_Tol_S(const float tol)
//===========================================================================
{
  float   *p1;
  float   *p2;
  float   *x1_x0;
  float   *x2_x0;
  float   *x3_x0;
  float   l_03, l_1p, l_2p, dot1, dot2;
  int     i, j;
  
  p1 = new float[d];
  p2 = new float[d];
  x1_x0 = new float[d];
  x2_x0 = new float[d];
  x3_x0 = new float[d];
  
  for ( i = 0 ; i < 4 ; i++ )    {
    dot1 = 0.0f;
    dot2 = 0.0f;
    l_03 = 0.0f;
    for ( j = 0 ; j < d ; j++ )	{
      x1_x0[j] = c_pts[1][i][j] - c_pts[0][i][j];
      x2_x0[j] = c_pts[2][i][j] - c_pts[0][i][j];
      x3_x0[j] = c_pts[3][i][j] - c_pts[0][i][j];
      dot1 += ( x1_x0[j] * x3_x0[j] );
      dot2 += ( x2_x0[j] * x3_x0[j] );
      l_03 += ( x3_x0[j] * x3_x0[j] );
    }
    if ( l_03 == 0.0f )
      continue;
    l_1p = 0.0f;
    l_2p = 0.0f;
    for ( j = 0 ; j < d ; j++ )      {
      p1[j] = c_pts[0][i][j] + dot1 * x3_x0[j] / l_03;
      p2[j] = c_pts[0][i][j] + dot2 * x3_x0[j] / l_03;
      l_1p += ( c_pts[1][i][j] - p1[j] ) * ( c_pts[1][i][j] - p1[j] );
      l_2p += ( c_pts[2][i][j] - p2[j] ) * ( c_pts[2][i][j] - p2[j] );
    }
    if ( l_1p > tol * tol || l_2p > tol * tol )	{
      delete[] p1;
      delete[] p2;
      delete[] x1_x0;
      delete[] x2_x0;
      delete[] x3_x0;
      return false;
    }
  }

  delete[] p1;
  delete[] p2;
  delete[] x1_x0;
  delete[] x2_x0;
  delete[] x3_x0;

  return true;
}


//***************************************************************************
//
// * Copy a set of control points
//===========================================================================
void SJCBezierPatch::
Copy_Controls(float * const c_in[4][4])
//===========================================================================
{
  int i, j, k;
  
  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ )      {
      if ( c_pts[i][j] )
	delete[] c_pts[i][j];
      c_pts[i][j] = new float[d];
      for ( k = 0 ; k < d ; k++ )
	c_pts[i][j][k] = c_in[i][j][k];
    }
}


//***************************************************************************
//
// * Delete a set of control points 
//===========================================================================
void SJCBezierPatch::
Delete_Controls(void)
//===========================================================================
{
  int i, j;

  for ( i = 0 ; i < 4 ; i++ )
    for ( j = 0 ; j < 4 ; j++ ) {
      delete[] c_pts[i][j];
      c_pts[i][j] = NULL;
    }
}


//***************************************************************************
//
// * LU decomposition the matrix a, it also used in simtransform
//===========================================================================
static int
LU_Decomp(double **a, int n, int *indx, int *d, double *vv)
//===========================================================================
{
  int     i, imax, j, k;
  double  big, dum, sum, temp;
  
  *d = 1;
  for ( i = 0 ; i < n ; i++ )   {
    big = 0.0;
    for ( j = 0 ; j < n ; j++ )
      if ( ( temp = fabs(a[i][j]) ) > big )
	big = temp;
    if ( big == 0.0 )       {
      fprintf(stderr, "Singular matrix in LU_Decomp\n");
      return 0;
    }
    vv[i] = 1.0 / big;
  }
  for ( j = 0 ; j < n ; j++ )    {
    for ( i = 0 ; i < j ; i++ )        {
      sum = a[i][j];
      for ( k = 0 ; k < i ; k++ )
	sum -= a[i][k] * a[k][j];
      a[i][j] = sum;
    }
    big = 0.0;
    for ( i = j ; i < n ; i++ )       {
      sum = a[i][j];
      for ( k = 0 ; k < j ; k++ )
	sum -= a[i][k] * a[k][j];
      a[i][j] = sum;
      if ( ( dum = vv[i] * fabs(sum) ) >= big ) {
	big = dum;
	imax = i;
      }
    }
    if ( j != imax )        {
      for ( k =0 ; k < n ; k++ )            {
	dum = a[imax][k];
	a[imax][k] = a[j][k];
	a[j][k] = dum;
      }
      *d = - (*d);
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if ( a[j][j] == 0.0 )
      a[j][j] = 1.0e-16;
    if ( j != n - 1 )        {
      dum = 1.0 / a[j][j];
      for ( i = j + 1 ; i < n ; i++ )
	a[i][j] *= dum;
    }
  }

  return 1;
}


//***************************************************************************
//
// * Use the LU back substitution to find the result in b
//===========================================================================
static void
LU_Back_Subst(double **a, int n, int *indx, double *b)
//===========================================================================
{
  int     i, ii = -1, ip, j;
  double  sum;

  for ( i = 0 ; i < n ; i++ )   {
    ip = indx[i];
    sum = b[ip];
    b[ip] = b[i];
    if ( ii != -1 )
      for ( j = ii ; j <= i - 1 ; j++ )
	sum -= a[i][j] * b[j];
    else if ( sum )
      ii = i;
    b[i] = sum;
  }
  for ( i = n - 1 ; i >= 0 ; i-- )    {
    sum = b[i];
    for ( j = i + 1 ; j < n ; j++ )
      sum -= a[i][j] * b[j];
    b[i] = sum / a[i][i];
  }
}



