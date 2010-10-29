/************************************************************************
     Main File:

     File:        ScalarField2D.h

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

#ifndef _SJCSCALARFIELD2D_H_
#define _SJCSCALARFIELD2D_H_

#include "SJCVector2.h"

#include <iostream>

class SJCScalarField2D {
 public:

  enum Boundary { BOUNDARY_WRAP, BOUNDARY_NOWRAP };

  // Contructor and destructor
  SJCScalarField2D(const uint x = 1, const uint y = 1, // Actual data size
		   const Boundary bx = BOUNDARY_WRAP,
		   const Boundary by = BOUNDARY_WRAP,
		   const float *d = 0);

  SJCScalarField2D(std::istream &f);
  SJCScalarField2D(const SJCScalarField2D &vf);
  
  ~SJCScalarField2D(void);
  
  // Assign operator  
  SJCScalarField2D&  operator=(const SJCScalarField2D &vf);
  
  // Operator in ()
  float 	operator()(const float x, const float y) { return value(x,y); }
  float 	operator()(const float x, const float y, const unsigned char r)
    { return value(x, y, r); }
  float&	operator()(const uint i, const uint j);
  
  // The scalar field at x, y position
  float	value(const float x, const float y);
  float	value(const float x, const float y, const unsigned char r);

  // Finite difference differential operators. Versions with the r argument
  // compute rotated versions, in 90 degree increments
  // The gradient of the scalar field at x, y position
  SJCVector2f	grad(const float x, const float y);
  SJCVector2f	grad(const float x, const float y, const unsigned char r);

  // The curl field of the scalar field at x, y position
  // v = curl * S.Z
  SJCVector2f	curl(const float x, const float y);
  SJCVector2f	curl(const float x, const float y, const unsigned char r);
  
  // The boundary type
  Boundary	BoundaryTypeX(void) const { return bound_cond[0]; }
  Boundary	BoundaryTypeY(void) const { return bound_cond[1]; }
  
  // Minimum of the 2D
  float    	MinX(void) const { return 0.0f; }
  float    	MinY(void) const { return 0.0f; }

  // Maximum of the 2D
  float    	MaxX(void) const { return nx - 1.0f; }
  float    	MaxY(void) const { return ny - 1.0f; }
  float    	DiffMinX(void) const;
  float    	DiffMinY(void) const;
  float    	DiffMaxX(void) const;
  float    	DiffMaxY(void) const;

  // Get the number of sample points in x, y direction
  uint    	NumX(void) const { return nx; }
  uint    	NumY(void) const { return ny; }

  // Write out the data to the output stream in binary format
  void    Write(std::ostream &o);
  
  // Write out the data to the output stream in binary format
  bool    Read(std::istream &in);

  // Output operator  
  friend std::ostream& operator<<(std::ostream &o,const SJCScalarField2D &vf);

 private:
  uint    	nx;            // Actual data size. Addressable might differ.
  uint    	ny;            // Which is the actual sample points in each
                               // Dimension

  Boundary	bound_cond[2]; // The boundary type
  float	       *data;
    
  // Clear the data conditions
  void	Destroy(void);
    
};

#endif
