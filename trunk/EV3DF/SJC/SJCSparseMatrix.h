/************************************************************************
     Main File:

     File:        SJCSparseMatrix.h

 
     Author:
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
               
  
     Comment:     Sparse matrix for the operation

     Constructors:
                  1. 0 : the default contructor
                  2. 2 :
                  3. 3 : 
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2.
************************************************************************/

#ifndef _SJC_SPARSE_MATRIX
#define _SJC_SPARSE_MATRIX

#include <vector>
#include <math.h>
#include <iostream>
#include <iomanip>

#include <SJC/SJC.h>
#include <SJC/SJCScalarField2.h>
#include <SJC/SJCScalarField3.h>
#include <SJC/SJCBoundary.h>

using namespace std;

class SJCSparseMatrix {
 public:
  static const double ZEROEPSILON; // Use to identify whether the number is
                                   // close to zero
  static const double EPSILON;
  static const double EPSILON_SQ;

 public:
  // Constructor and destructor
  SJCSparseMatrix(void);
  SJCSparseMatrix(int dim);
  SJCSparseMatrix(const SJCSparseMatrix& m);
  
  // Set the size of the sparse matrix
  void SetSize(int dim);

  // These are to make a sparse matrix for smoke.  The diag length is w*h*d
  void Prepare2D(const uint w, const uint h, const double dx, const double dy,
		 SJCScalarField2d* boundary, SJCBoundary bx, SJCBoundary by);
  void Prepare3D(const uint w, const uint h, const uint d, 
		 const double dx, const double dy, const double dz, 
		 SJCScalarField3d* boundary, SJCBoundary bx, SJCBoundary by,
		 SJCBoundary bz);


  // Two matrix multiplication
  void Multiply(const vector<double> &src, vector<double> &dest);
  // Two matrix multiplication but source first rotate
  void MultiplyTranspose(const vector<double> &src, vector<double> &dest);

  // if transpose is true then we treat matrix to be AA' 
  // (i.e. for preconditioned conjugate subproblem)
  void ConjGrad(const vector<double> &b, vector<double> &x, 
		bool transpose = false); 
  void PreCondConjGrad(const vector<double> &b, vector<double> &x, 
		       SJCSparseMatrix &preConditioner);
	
  // solves AA' x = b if A is lower triangular
  void BackSubSolve(const vector<double> &b, vector<double> &x);
	

  void IncompleteCholeskyFactor(void);
  void Print(void);
  
  friend std::ostream& operator<<(std::ostream&o, const SJCSparseMatrix &v);

 private:
  // Get the array element index in 2D
  inline static int Index2d(const uint x, const uint y, const uint w) { 
    return y * w + x; }

  // Get the array element index in 3D
  inline static int Index3d(const uint x, const uint y, const int z, 
			    const uint w, const uint h) {
    return z * w * h + y * w + x; }

  private:
  // Get the normal of the vector x
  inline static double VecNorm(const vector<double>& x);
  // Print out the vector
  inline static void PrintVec(char *str, const vector<double> &x);
  // Print out the vector
  inline static void PrintVec(char *str, const vector<uint> &x);
  
  
 private:
  uint           m_uDim;          // Number of diaganol elements 
                                  // (square dimension)

  vector<uint>   m_VIndices;     // Index of off diaganol values, ija
  vector<double> m_VValues;
  
};


#endif
