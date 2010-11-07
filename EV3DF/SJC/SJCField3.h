/************************************************************************
     Main File:

     File:        SJCField3.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
  
     Comment:     Basic class for 3D field

     Constructors:
                  1. 0 : default
                  1. 6 : constructor to set up all value by input parameters
                  2. 1 : Set up the data from input stream
                  3. 1 : copy contructor
                   
     Functions:  
                 1. = : Assign operator which copy the parameter of random
                 2. MinX, MinY, MinZ, MaxX, MaxY, MaxZ: geometry boundary in 
                    X, Y, Z
                 3. BoundMaxX, BoundMaxY, BoundMaxZ, BoundMinX, BoundMinY,
                    BoundMinZ: real simulation boundary.
                 4. NumX, NumY, NumZ: get the number of sample points in 
                    X, Y, Z
                 5. Index(uint, uint, uint): get the index for this field
                 6. Write: write out the data into stream in binary form
                 7. >>: output in the ascii form

************************************************************************/

#ifndef _SJCFIELD3_H_
#define _SJCFIELD3_H_

#include <iostream>
#include <iomanip>

#include "SJC.h"
#include "SJCBoundary.h"
#pragma warning(disable:4520)
class SJCField3d {
 public:
  // Contructor and destructor
  SJCField3d(void){}
  // Main constructor
  SJCField3d(const uint nx        = 1,   // The grid size
	     const uint ny        = 1,   
	     const uint nz        = 1,   
	     const double dx      = 1.0, // The size of the voxel 
	     const double dy      = 1.0,
	     const double dz      = 1.0,
	     const SJCBoundary bx = BOUNDARY_NOWRAP,
	     const SJCBoundary by = BOUNDARY_NOWRAP,
	     const SJCBoundary bz = BOUNDARY_NOWRAP);

  // Input constructor
  SJCField3d(std::istream &f);

  // Copy constructor
  SJCField3d(const SJCField3d &vf);

  // Destructor
  ~SJCField3d(void);

 protected:    
  // Clear the data conditions
  virtual void	Destroy(void);

  // Assign the scalar field from vf
  void  Assign(const SJCField3d &vf);

 public:
  // Calculate the index
	 virtual uint Index(uint index_x, uint index_y, uint index_z) {
	     if (index_x<0) index_x=0;
	     if (index_x>=m_uNX) index_x=m_uNX-1;
	     if (index_y<0) index_y=0;
	     if (index_y>=m_uNX) index_y=m_uNY-1;
	     if (index_z<0) index_z=0;
	     if (index_z>=m_uNX) index_z=m_uNZ-1;
    return index_z * m_uNX * m_uNY + index_y * m_uNX + index_x; }

 public:    
  // Assign operator
  SJCField3d& operator=(const SJCField3d &vf);

  // Set up information
  void Set(const uint nx        = 1,   // The grid size
	   const uint ny        = 1,   
	   const uint nz        = 1,   
	   const double dx      = 1.0, // The size of the voxel 
	   const double dy      = 1.0,
	   const double dz      = 1.0,
	   const SJCBoundary bx = BOUNDARY_NOWRAP,
	   const SJCBoundary by = BOUNDARY_NOWRAP,
	   const SJCBoundary bz = BOUNDARY_NOWRAP);

  // The boundary type
  SJCBoundary	BoundaryTypeX(void) const { return m_VBoundary[0]; }
  SJCBoundary	BoundaryTypeY(void) const { return m_VBoundary[1]; }
  SJCBoundary	BoundaryTypeZ(void) const { return m_VBoundary[2]; }
  
  // * Right now the MinX and BuoundMinX is the same for NoWrap
  // The value between minmum and maximum is the valid simulation area
  // Minimum of the 3D  in real dimension
  double    	MinX(void) const { return m_dDX; }
  double    	MinY(void) const { return m_dDY; }
  double    	MinZ(void) const { return m_dDZ; }

  // Maximum of the 3D in real dimension
  double    	MaxX(void) const { return m_dLengthX - m_dDX; }
  double    	MaxY(void) const { return m_dLengthY - m_dDY; }
  double    	MaxZ(void) const { return m_dLengthZ - m_dDZ; }

  // Get the boundary minimum which no meaning outside this boundary
  double    	BoundMinX(void) const;
  double    	BoundMinY(void) const;
  double    	BoundMinZ(void) const;

  // Get the differential maximum
  double    	BoundMaxX(void) const;
  double    	BoundMaxY(void) const;
  double    	BoundMaxZ(void) const;

  // Get the number of sample points in x, y direction
  uint    	NumX(void) const { return m_uNX; }
  uint    	NumY(void) const { return m_uNY; }
  uint    	NumZ(void) const { return m_uNZ; }
  
  // Get the size between each sample point
  double        DX(void) const { return m_dDX;}
  double        DY(void) const { return m_dDY;}
  double        DZ(void) const { return m_dDZ;}

  // Write out the data to the output stream in binary format
  virtual void  Write(std::ostream &o);
  
  // Output operator  
  friend std::ostream& operator<<(std::ostream &o, const SJCField3d &vf);

  uint    	m_uNX;       // Actual data size. Addressable might differ.
  uint    	m_uNY;       // Which is the actual sample points in each
  uint    	m_uNZ;       // Dimension

  double        m_dDX;       // The actual size of each voxel in x, y dim
  double        m_dDY;
  double        m_dDZ;

  double        m_dHalfDX;   // The half distance
  double        m_dHalfDY;
  double        m_dHalfDZ;

  double        m_dLengthX;   // The total length in x, y
  double        m_dLengthY;
  double        m_dLengthZ;

  SJCBoundary	m_VBoundary[3]; // The boundary type
};

#endif
