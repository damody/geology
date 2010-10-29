/************************************************************************
     Main File:

     File:        SJCField2.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
  
     Comment:     Base class for the field to handle the dimensional 
                  information
 
     Constructors:
                  1. 0 : default
                  2. 6 : constructor to set up all value by input parameters
                  3. 1 : Set up the data from input stream
                  4. 1 : copy contructor
                   
     Functions:  
                 1. = : Assign operator which copy the parameter of random
                 2. MinX, MinY, MaxX, MaxY: get the maximum and minimum value 
                    of X, y
                 3. DiffMaxX, DiffMaxY, DiffMinX, DiffMinY: get the max and
                    min value of X, Y
                 8. NumX, NumY: get the number of sample points in X, Y
                 9. Write: write out the data into stream in binary form
                10. >>: output in the ascii form

************************************************************************/

#ifndef _SJCFIELD2_H_
#define _SJCFIELD2_H_

#include <iostream>
#include <iomanip>

#include "SJC.h"
#include "SJCBoundary.h"
#pragma warning(disable:4520)
class SJCField2d : public  TCountPointTo {
 public:
  // Contructor and destructor
  SJCField2d(void){}

  // Main constructor
  SJCField2d(const uint   nx      = 1,   // The grid size
	     const uint   ny      = 1,   
	     const double dx      = 1.0, // The size of the voxel 
	     const double dy      = 1.0,
	     const SJCBoundary bx = BOUNDARY_NOWRAP,
	     const SJCBoundary by = BOUNDARY_NOWRAP);

  // Input constructor
  SJCField2d(std::istream &f);

  // Copy constructor
  SJCField2d(const SJCField2d &vf);

  // Destructor
  ~SJCField2d(void);

 protected:    
  // Clear the data conditions
  virtual void	Destroy(void);

  // Assign the scalar field from vf
  void  Assign(const SJCField2d &vf);

 public:
  // Calculate the index
  virtual uint Index(const uint index_x, const uint index_y){
    SJCAssert((index_x >= 0 && index_x < m_uNX &&
	       index_y >= 0 && index_y < m_uNY), "Fail in 2D field index");
    return index_y * m_uNX + index_x;
  }

 public:
  // Assign operator
  SJCField2d& operator=(const SJCField2d &vf);

  // Set up information
  void Set(const uint   nx      = 1,   // The grid size
	   const uint   ny      = 1,   
	   const double dx      = 1.0, // The size of the voxel 
	   const double dy      = 1.0,
	   const SJCBoundary bx = BOUNDARY_WRAP,
	   const SJCBoundary by = BOUNDARY_WRAP);

  // The boundary type
  SJCBoundary	BoundaryTypeX(void) const { return m_VBoundary[0]; }
  SJCBoundary	BoundaryTypeY(void) const { return m_VBoundary[1]; }
  

  // * Right now the MinX and BuoundMinX is the same for NoWrap
  // The value between minmum and maximum is the valid simulation area
  // The minimum of the valid simulation area
  double    	MinX(void) const { return m_dDX; }
  double    	MinY(void) const { return m_dDY; }

  // Maximum of the 2D in real dimension
  double    	MaxX(void) const { return m_dLengthX - m_dDX; }
  double    	MaxY(void) const { return m_dLengthY - m_dDY; }

  // The value between minmum and maximum is the valid simulation area
  // Get the boundary minimum which no meaning outside this boundary
  double    	BoundMinX(void) const;
  double    	BoundMinY(void) const;

  // Get the boundary maximum which no meaning outside this boundary
  double    	BoundMaxX(void) const;
  double    	BoundMaxY(void) const;

  // Get the number of sample points in x, y direction
  uint    	NumX(void) const { return m_uNX; }
  uint    	NumY(void) const { return m_uNY; }
  
  // Get the size between each sample point
  double        DX(void) const { return m_dDX;}
  double        DY(void) const { return m_dDY;}

  // Read from a binary file
  bool    Read(std::istream &f);

  // Write out the data to the output stream in binary format
  virtual void  Write(std::ostream &o);
  
  // Output operator  
  friend std::ostream& operator<<(std::ostream &o, const SJCField2d &vf);

 protected:
  uint    	m_uNX;       // Actual data size. Addressable might differ.
  uint    	m_uNY;       // Which is the actual sample points in each
                             // Dimension

  double        m_dDX;       // The actual size of each voxel in x, y dim
  double        m_dDY;

  double        m_dHalfDX;   // The half distance
  double        m_dHalfDY;

  double        m_dLengthX;   // The total length in x, y
  double        m_dLengthY;

  SJCBoundary	m_VBoundary[2]; // The boundary type
};

#endif
