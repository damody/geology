/************************************************************************
     Main File:

     File:        SJCScalarField2.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     Class to handle the scalar field in 2D

        ---------
        |*|*|*|*|  
        ---------
        |*|*|*|*|
        ---------
        |*|*|*|*|
        ---------

     Constructors:
                  1. 0 : the default contructor
                  2. 6 : constructor to set up all value by input parameters
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

#ifndef _SJCSCALARFIELD2_H_
#define _SJCSCALARFIELD2_H_

#include <iostream>
#include <iomanip>
#include <assert.h>

#include "SJC.h"

#include "SJCField2.h"
#include "SJCVector2.h"
#include "SJCBoundary.h"
#pragma warning(disable:4520)
class SJCScalarField2d : public SJCField2d {
 public:
  // Contructor and destructor
  SJCScalarField2d(void);

  // The main constructor
  SJCScalarField2d(const uint nx        = 1,   // The grid size
		   const uint ny        = 1,   
		   const double delta_x = 1.0, // The size of the voxel 
		   const double delta_y = 1.0,
		   const SJCBoundary bx = BOUNDARY_NOWRAP,
		   const SJCBoundary by = BOUNDARY_NOWRAP,
		   const double *d      = 0);

  // Input constructor
  SJCScalarField2d(std::istream &f);

  // Copy constructor
  SJCScalarField2d(const SJCScalarField2d &vf);

  // Destructor
  ~SJCScalarField2d(void);

 protected:    
  // Clear the data conditions
  void	Destroy(void);

  // Assign the scalar field from vf
  void  Assign(const SJCScalarField2d &vf);

 public:    
  // Assign operator  
  SJCScalarField2d&  operator=(const SJCScalarField2d &vf);

  // Set up information
  void Set(const uint nx        = 1,   // The grid size
	   const uint ny        = 1,   
	   const double delta_x = 1.0, // The size of the voxel 
	   const double delta_y = 1.0,
	   const SJCBoundary bx = BOUNDARY_NOWRAP,
	   const SJCBoundary by = BOUNDARY_NOWRAP,
	   const double *d      = 0);
   
  // Get the value by interpolation at x, y position
  double  operator()(const double x, const double y) { return Value(x, y);}

  // Get the value by interpolation at x, y position
  double  operator()(const SJCVector2d& pos) { return Value(pos.x(), pos.y());}

  // Get the reference of the index position
  double&  operator()(const uint index_x, const uint index_y);

  // The scalar field at x, y position
  double Value(const double x, const double y);

  // Set up the scalar field at index_x,y
  void   Value(const int index_x, const int index_y, double value) {
    m_VData[Index(index_x, index_y)] = value;
  }

  // Get the coordinate of the voxel center
  void Coord(const uint index_x, const uint index_y, SJCVector2d& pos){
    pos.set( (double)(index_x + .5) * m_dDX, 
	     (double)(index_y + .5) * m_dDY);
  }

  // * Finite difference differential operators. 
  //   Veriations with the r argument compute rotated versions, 
  //   in 90 degree increments
  //
  // * The gradient of the scalar field at x, y position
  SJCVector2d Grad(const double x, const double y);

  // The curl field of the scalar field at x, y position
  SJCVector2d Curl(const double x, const double y);
  
  // Write out the data to the output stream in binary format
  void    Write(std::ostream &o);
  
  // Output operator  
  friend std::ostream& operator<<(std::ostream &o, const SJCScalarField2d &vf);

 private:
  double*       m_VData;
};

#endif
