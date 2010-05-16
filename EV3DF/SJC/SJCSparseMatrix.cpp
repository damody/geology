/************************************************************************
     Main File:

     File:        SJCSparseMatrix.h

 
     Author:
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
               
     Comment:    The sparse matrix for the projection free   

     Constructors:
                  1. 0 : the default contructor
                  2. 2 :
                  3. 3 : 
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2.
************************************************************************/

#include <SJCSparseMatrix.h>

const double SJCSparseMatrix::ZEROEPSILON = 0.0001;
const double SJCSparseMatrix::EPSILON     = 1.0e-2;
const double SJCSparseMatrix::EPSILON_SQ  = 1.0e-4;

//**************************************************************************
//
// * Default constructor
//==========================================================================
SJCSparseMatrix::
SJCSparseMatrix(void) 
//==========================================================================
{
  m_uDim = 0;
}

//**************************************************************************
//
// * Constructor to set up the sparse matrix
//==========================================================================
SJCSparseMatrix::SJCSparseMatrix(int n) 
//==========================================================================
{
  m_uDim = n;
}

//**************************************************************************
//
// * Copy constructor
//==========================================================================
SJCSparseMatrix::
SJCSparseMatrix(const SJCSparseMatrix& m)
//==========================================================================
{
  m_uDim         = m.m_uDim;
  m_VIndices     = m.m_VIndices;
  m_VValues      = m.m_VValues;
}


//**************************************************************************
//
// * Set up the dimension
//==========================================================================
void SJCSparseMatrix::SetSize(int n) 
//==========================================================================
{
  m_uDim = n;
}

//**************************************************************************
//
// * To construct the 2D sparse matrix
//==========================================================================
void SJCSparseMatrix::
Prepare2D(const uint w, const uint h, const double dx, const double dy,
	  SJCScalarField2d* bound, SJCBoundary bx, SJCBoundary by) 
//==========================================================================
{

  // Compute the square delta
  double inv_dx_sqr = 1.0 / (dx * dx);
  double inv_dy_sqr = 1.0 / (dy * dy);


  // Resize the data storage
  m_VValues.resize(m_uDim + 1);
  m_VIndices.resize(m_uDim + 1);
  
  // Print out the mesm_VValuesge
  // Set up current to the next available
  int curr      = m_uDim + 1;
  m_VIndices[0] = m_uDim + 1;

  for (uint j = 0; j < h; j++) {
    for (uint i = 0; i < w; i++) {

      //******************************************************************
      // Boundary element
      //******************************************************************
      if (i == 0 || i == w -1 || j == 0 || j == h - 1) {
	int orig    = curr;
	int x_count = 0;
	int y_count = 0;

	// The left boarder and its right element is not a boundary
	if(i == 0 && j != 0 && j != h -1 && 
	   (*bound)(i + 1, j) == 0.f) {

	  m_VValues.push_back(-inv_dx_sqr);
	  m_VIndices.push_back( Index2d(i + 1, j, w));
	  x_count++;
	  curr++;
	}

	// The right boarder
	if(i == w - 1 && j != 0 && j != h -1 &&
	   (*bound)(i - 1, j) == 0.f) {
	  m_VValues.push_back(-inv_dx_sqr);
	  m_VIndices.push_back( Index2d(i - 1, j, w));
	  x_count++;
	  curr++;
	}

	// The bottom boarder
	if(j == 0 && i != 0 && i != w - 1 &&
	   (*bound)(i, j + 1) == 0.f) {
	  m_VValues.push_back(-inv_dy_sqr);
	  m_VIndices.push_back( Index2d(i, j+1, w));
	  y_count++;
	  curr++;
	}

	// The top boarder
	if(j == h - 1 && i != 0 && i != w -1 && 
	   (*bound)(i, j - 1) == 0.f) {
	  m_VValues.push_back(-inv_dy_sqr);
	  m_VIndices.push_back( Index2d(i, j-1, w));
	  y_count++;
	  curr++;
	}
	
	// To make sure at least something inside
	int diff     = curr - orig;

	// Compute the accumulation
	double accum = (double)x_count*inv_dx_sqr + (double)y_count*inv_dy_sqr;
	m_VValues[Index2d(i, j, w)]      = diff > 1 ? accum : 1.f;
	m_VIndices[Index2d(i, j, w) + 1] = curr;
      }
      else {  // It is not a boundary point.
	int orig    = curr;
	int x_count = 0;
	int y_count = 0;
	
	// The left element exists
	if(i != 0 && (*bound)(i - 1, j) == 0.f) {
	  m_VValues.push_back(-inv_dx_sqr);
	  m_VIndices.push_back( Index2d(i-1, j, w));
	  x_count++;
	  curr++;
	}

	// The right element exists
	if(i != w - 1 && (*bound)(i + 1, j) == 0.f) {
	  m_VValues.push_back(-inv_dx_sqr);
	  m_VIndices.push_back( Index2d(i+1, j, w));
	  x_count++;
	  curr++;
	}

	// The bottom element exists
	if(j != 0 && (*bound)(i, j -1) == 0.f) {
	  m_VValues.push_back(-inv_dy_sqr);
	  m_VIndices.push_back( Index2d(i, j-1, w));
	  y_count++;
	  curr++;
	}

	// The top boarder element
	if(j != h - 1 && (*bound)(i, j + 1) == 0.f) {
	  m_VValues.push_back(-inv_dy_sqr);
	  m_VIndices.push_back( Index2d(i, j+1, w));
	  y_count++;
	  curr++;
	}

	// Compute the difference
	int diff = curr - orig;

	// Compute the coefficient for the center
	double accum = (double)x_count*inv_dx_sqr + (double)y_count*inv_dy_sqr;

	// Put data in
	m_VValues[Index2d(i, j, w)]      = diff > 1 ? accum : 1;

	m_VIndices[Index2d(i, j, w) + 1] = curr;
      } // end of else
    } // end of for i
  } // end of for j
}


//**************************************************************************
//
// * Prepare the sparse matrix for 3D
//==========================================================================
void SJCSparseMatrix::
Prepare3D(const uint w, const uint h, const uint d, 
	  const double dx, const double dy, const double dz, 
	  SJCScalarField3d* bound, SJCBoundary bx, SJCBoundary by,
	  SJCBoundary bz) 
//==========================================================================
{
  // Compute the inverse of dx, dy, dz
  double inv_dx_sqr = 1.0 / (dx * dx);
  double inv_dy_sqr = 1.0 / (dy * dy);
  double inv_dz_sqr = 1.0 / (dz * dz);

  // Resize the data storage
  m_VValues.resize(m_uDim + 1);
  m_VIndices.resize(m_uDim + 1);
  
  // Set up current to the next available
  int curr      = m_uDim + 1;
  m_VIndices[0] = m_uDim + 1;
  for (uint k = 0; k < d; k++) {
    for (uint j = 0; j < h; j++) {
      for (uint i = 0; i < w; i++) {

	// if(bound_cond == BOUNDARY_NOWRAP)
	if (i == 0 || i == w - 1 || 
	    j == 0 || j == h - 1 ||
	    k == 0 || k == d - 1) { // boundary element

	  int orig    = curr;
	  int x_count = 0;
	  int y_count = 0;
	  int z_count = 0;

	  // The left boarder face
	  if(i == 0 && j != 0 && j != h-1 && k != 0 && k != d-1) {
	    m_VValues.push_back(-inv_dx_sqr);
	    m_VIndices.push_back( Index3d(i + 1, j, k, w, h));
	    x_count++;
	    curr++;
	  }

	  // The right boarder face
	  if(i == w - 1 && j != 0 && j != h-1 && k != 0 && k != d-1) {
	    m_VValues.push_back(-inv_dx_sqr);
	    m_VIndices.push_back( Index3d(i - 1, j, k, w, h));
	    x_count++;
	    curr++;
	  }

	  // The bottom boarder
	  if(j == 0 && i != 0 && i != w -1 && k != 0 && k != d-1) {
	    m_VValues.push_back(-inv_dy_sqr);
	    m_VIndices.push_back( Index3d(i, j + 1, k, w, h));
	    y_count++;
	    curr++;
	  }

	  // The top boarder
	  if(j == h - 1 && i != 0 && i != w -1 && k != 0 && k != d-1) {
	    m_VValues.push_back(-inv_dy_sqr);
	    m_VIndices.push_back( Index3d(i, j - 1, k, w, h));
	    y_count++;
	    curr++;
	  }
	
	  // The back boarder
	  if(k == 0 && i != 0 && i != w -1 && j != 0 && j != h-1) {
	    m_VValues.push_back(-inv_dz_sqr);
	    m_VIndices.push_back( Index3d(i, j, k+1, w, h));
	    z_count++;
	    curr++;
	  }

	  // The front boarder
	  if(k == d - 1 && i != 0 && i != w-1 && j != 0 && j != h-1) {
	    m_VValues.push_back(-inv_dz_sqr);
	    m_VIndices.push_back( Index3d(i, j, k-1, w, h));
	    z_count++;
	    curr++;
	  }

	  // To make sure at least something inside
	  int diff     = curr - orig;
	  double accum = (double)x_count * inv_dx_sqr + 
	                 (double)y_count * inv_dy_sqr +
	                 (double)z_count * inv_dz_sqr;
	  
	  m_VValues[Index3d(i, j, k, w, h)] = diff > 1 ? accum : 1;
	  m_VIndices[Index3d(i, j, k, w, h) + 1] = curr;
	}
	else  { // non-boundary element

	  int orig    = curr;
	  int x_count = 0;
	  int y_count = 0;
	  int z_count = 0;

	  // The left element exists
	  if(i != 0) {
	    m_VValues.push_back(-inv_dx_sqr);
	    m_VIndices.push_back( Index3d(i-1, j, k, w, h));
	    x_count++;
	    curr++;
	  }

	  // The right element exists
	  if(i != w-1) {
	    m_VValues.push_back(-inv_dx_sqr);
	    m_VIndices.push_back( Index3d(i+1, j, k, w, h));
	    x_count++;
	    curr++;
	  }

	  // The bottom element
	  if(j != 0) {
	    m_VValues.push_back(-inv_dy_sqr);
	    m_VIndices.push_back( Index3d(i, j-1, k, w, h));
	    y_count++;
	    curr++;
	  }

	  // The top element
	  if(j != h - 1) {
	    m_VValues.push_back(-inv_dy_sqr);
	    m_VIndices.push_back( Index3d(i, j+1, k, w, h));
	    y_count++;
	    curr++;
	  }
	
	  // The back element
	  if(k != 0) {
	    m_VValues.push_back(-inv_dz_sqr);
	    m_VIndices.push_back( Index3d(i, j, k-1, w, h));
	    z_count++;
	    curr++;
	  }

	  // The front element
	  if(k != d - 1) {
	    m_VValues.push_back(-inv_dz_sqr);
	    m_VIndices.push_back( Index3d(i, j, k+1, w, h));
	    z_count++;
	    curr++;
	  }

	  // To make sure at least something inside
	  int    diff  = curr - orig;
	  double accum = (double)x_count * inv_dx_sqr + 
	                 (double)y_count * inv_dy_sqr +
	                 (double)z_count * inv_dz_sqr;

	  m_VValues[Index3d(i, j, k, w, h)]      = diff > 1 ? accum : 1;
	  m_VIndices[Index3d(i, j, k, w, h) + 1] = curr;
	} // end of else
      } // end of for i
    } // end of for j
  }// end of for k
}

//**************************************************************************
//
// * Multiply two sparse matrix
//==========================================================================
void SJCSparseMatrix::
Multiply(const vector<double> &src, vector<double> &dest) 
//==========================================================================
{
  // The incoming matrix is a regular matrix.  Each entry is multiplied against
  // a diagonal of the sparse matrix and its neighbors.
  
  // let's do each row in the vector
  double acc;
  for (uint i = 0 ; i < dest.size(); i++) {
    // Compute the diagonal
    acc = src[i] * m_VValues[i]; 

    // Do the valid column
    for (uint k = m_VIndices[i]; k <= m_VIndices[i + 1] - 1; k++) {
      acc += src[m_VIndices[k]] * m_VValues[k];
    }

    dest[i] = acc;
  } // end of for i
}

//**************************************************************************
//
// * Multiply two matrix with one transpose
//==========================================================================
void SJCSparseMatrix::
MultiplyTranspose(const vector<double> &src, vector<double> &dest) 
//==========================================================================
{
  // let's do each row in the vector
  uint i;
  
  for (i = 0; i < dest.size(); i++)
     dest[i] = 0;

  for (i = 0 ; i <dest.size(); i++) {
    // Compute the diagonal
    dest[i] += src[i] * m_VValues[i];

    for (uint k = m_VIndices[i]; k <= m_VIndices[i + 1] - 1; k++) {
      int j = m_VIndices[k];
      dest[j] += src[i] * m_VValues[k];
    }
  }
}


//**************************************************************************
//
// * Conjugate gradient to solve the system
//   From numerical recipe 90
//==========================================================================
void SJCSparseMatrix::
ConjGrad(const vector<double> &b, vector<double> &x, bool transpose)
//==========================================================================
{
  
  int count = 0;

  // Temporary variables
  vector<double> r(m_uDim), p(m_uDim), Apk(m_uDim);
  vector<double> temp(m_uDim), temp2(m_uDim);

  double alpha, beta , bnorm;

  // Handle the matrix according to wether transpose or not  
  if (transpose) {
    MultiplyTranspose(x, temp2);
    Multiply(temp2, temp);
  } 
  else {
    Multiply(x, temp);
  }
  //  PrintVec("Temp", temp);
  

  for (uint i = 0; i < m_uDim; i++) {
    r[i] = b[i] - temp[i];
    p[i] = r[i];
  }
  //  PrintVec("P", p);
  
  // Compute the norm of B
  bnorm = VecNorm(b);
  
  if (bnorm < EPSILON && VecNorm(r) < EPSILON || 
      bnorm > EPSILON && VecNorm(r) / bnorm < EPSILON ) {
    return;
  }	
  
  //  printf("Norm r and b %f %f\n", VecNorm(r), bnorm);
  
  //  getchar();
  

  while(1) {
    count++;
   
    double rkdot = 0, pkdotApk = 0;
    if (transpose) {
      MultiplyTranspose(p, temp2);
      Multiply(temp2, Apk);
    }
    else {
      Multiply(p, Apk);
    }
    
    //    PrintVec("Apk", Apk);
  
    
    for (uint i = 0; i < m_uDim; i++) {
      rkdot += r[i] * r[i];
      pkdotApk += p[i] * Apk[i]; 
    }
    alpha = rkdot / pkdotApk;
    
    
    // calc x
    for (uint i = 0; i < m_uDim; i++) {
      x[i] += alpha * p[i];
    }

    //    PrintVec("x", x);
     
    double rkdotnew = 0;
    for (uint i = 0; i < m_uDim; i++) {
      r[i] = r[i] - alpha * Apk[i];
      rkdotnew += r[i] * r[i];
    }
    

    //    PrintVec("r", r);
 
    beta = rkdotnew / rkdot;
    for (uint i = 0; i < m_uDim; i++) {
      p[i] = r[i] +  beta * p[i];	
    }
  
    //    PrintVec("p", p);
   
    double rnorm = VecNorm(r);
    double sqrt_norm = sqrt(rnorm);
    
    if ((bnorm < EPSILON && sqrt_norm < EPSILON) || 
	(sqrt_norm / bnorm < EPSILON) || 
	count > 1000 || 
	count > b.size()) {
      printf("Converge in %d\n", count);
      return;
    }	// end of if
  } // end of for
  
}



//**************************************************************************
//
// * Precondition conjugate gradient
//==========================================================================
void SJCSparseMatrix::
PreCondConjGrad(const vector<double> &b, vector<double> &x, 
		SJCSparseMatrix &PreConditioner)
//==========================================================================
{
  double innerProdOld = 0, innerProd = 0, beta,alpha;
  vector<double> z(b.size());
  vector<double> r(b.size());
  vector<double> p(b.size());
  vector<double> Ap(b.size());
  vector<double> temp(b.size());
  
  r = b;
  
  int k = 0;
  while((sqrt(VecNorm(r)) / VecNorm(b) )> EPSILON) {
    PreConditioner.BackSubSolve(r, z);	
    k++;

    innerProd = 0;
    for (uint i = 0; i < r.size(); i++) 
      innerProd += r[i] * z[i];
    
    if (k == 1) {
      p = z;
    } 
    else {		
      beta = innerProd / innerProdOld;
      for (uint i = 0; i < r.size(); i++) 
	p[i] = z[i] + beta * p[i] ;
    }

    Multiply(p, Ap);
    double temp1 = 0;
    for (uint i = 0; i < r.size(); i++) { 
      temp1 += p[i] * Ap[i];
    }

    alpha = innerProd / temp1 ;
    for (uint i = 0; i < r.size(); i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    
    innerProdOld = innerProd;
   
  }
}

//**************************************************************************
//
// * Using back substitute to solve the linear system
//==========================================================================
void SJCSparseMatrix::
BackSubSolve(const vector<double> &b, vector<double> &x)
//==========================================================================
{
  // do each row
  static vector<double> xt(m_uDim);

  for (uint i = 0; i < m_uDim; i++) {
    double old = b[i];
    for (uint k = m_VIndices[i]; k <= m_VIndices[i+1] - 1; k++) {
      uint j = m_VIndices[k];
      old -= xt[j] * m_VValues[k];
    }
    xt[i] = old / m_VValues[i];
  }
  
  
  for (uint i = m_uDim - 1; i >= 0; i--) {
    x[i] = xt[i] / m_VValues[i];
    
    int bound = m_VIndices[i+1] - 1;
    for (uint k = m_VIndices[i]; k <= bound; k++) {
      xt[m_VIndices[k]] -= x[i] * m_VValues[k];
    }
  }

}

//**************************************************************************
//
// * Get the vec Norm
//==========================================================================
double SJCSparseMatrix::VecNorm(const vector<double>& x)
//==========================================================================
{
  double sumSq = 0;
  std::vector<double>::const_iterator i = x.begin();
  for (; i != x.end(); ++i) {
    sumSq += *i * *i;
  }
  return sqrt(sumSq);
}

//**************************************************************************
//
// *
//==========================================================================
void SJCSparseMatrix::PrintVec(char *str,const vector<double> &x)
//==========================================================================
{
  printf("%s: ", str);
  for (unsigned int i = 0; i < x.size(); i++) {
    printf("%f ", x[i]);
  }
  printf("\n");
}

//**************************************************************************
//
// *
//==========================================================================
void SJCSparseMatrix::PrintVec(char *str,const vector<uint> &x)
//==========================================================================
{
  printf("%s: ", str);
  for (unsigned int i = 0; i < x.size(); i++) {
    printf("%d ", x[i]);
  }
  printf("\n");
}


//**************************************************************************
//
// *
//==========================================================================
void SJCSparseMatrix::Print(void) 
//==========================================================================
{
  //  PrintVec("Indices", m_VIndices);
  //  PrintVec("Values", m_VValues);

  printf("For each row\n");
  
  for(uint i = 0 ; i < m_uDim; i++){
    printf("row %d: (diag %3.2f) ", i, m_VValues[i]);

    uint start_ix = m_VIndices[i];
    uint end_ix   = m_VIndices[i+1];
    
    for(uint j = start_ix; j < end_ix; j++){
      printf(" %dth %3.2f ", m_VIndices[j], m_VValues[j]);
      
    }
    printf("\n");
    
  }
  
}

//**************************************************************************
//
// *
//==========================================================================
std::ostream& operator<<(std::ostream&o, const SJCSparseMatrix &v) 
//==========================================================================
{

  std::vector<double> row;
  row.resize(v.m_uDim);

  for(uint i = 0 ; i < v.m_uDim; i++){

    for(uint j = 0; j < v.m_uDim; j++)
      row[j] = 0;

    row[i] = v.m_VValues[i];
    uint start_ix = v.m_VIndices[i];
    uint end_ix   = v.m_VIndices[i+1];

    for(uint j = start_ix; j < end_ix; j++){
       row[v.m_VIndices[j]] = v.m_VValues[j];
    }
    for(uint j = 0; j < v.m_uDim; j++)
      o << row[j] << " " ;
    o << "\n";
  }


  return o;
}


//**************************************************************************
//
// * ??? Is this Steven's implementation or Leo's
//==========================================================================
void SJCSparseMatrix::IncompleteCholeskyFactor(void)
//==========================================================================
{

  /*
  int k,i;
  
  for (k = 0; k < m_uDim; k++) {
    printf("k=%d of %d\n", k, m_uDim);
    m_VDiag[k] = sqrt(m_VDiag[k]);
    
    for (i = k+1; i < m_uDim; i++) {
      int idx = FindIdx(i,k);
      if (idx != -1) {
	m_VDiag[idx] /= m_VDiag[k];
      }
    }
		
    for (i = k+1; i < m_uDim; i++) {
      for (int kk = m_VOffDiagInd[i]; kk < m_VOffDiagInd[i+1]; kk++) {
	if (m_VNeighborInd[kk] > k) {
	  m_VOffDiag[kk] -= GetVal(i,k)* GetVal(m_VNeighborInd[kk],k) ;
	}
      }
      
    }	
  }
  
  
  for (i=0; i < m_uDim; i++) {
    int bound = m_VOffDiagInd[i+1];
    for (k = m_VOffDiagInd[i]; k < bound; k++) {
      if (m_VNeighborInd[k] > i) m_VOffDiag[k] = 0.0;
    }
  }
  */  
}
