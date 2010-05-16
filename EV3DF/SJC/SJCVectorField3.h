/************************************************************************
     Main File:

     File:        SJCVectorField3.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     Class to handle the scalar field in 3D

     Constructors:
                  1. 0 : the default contructor
                  2. 8 : constructor to set up all value by input parameters
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

#ifndef _SJCVECTORFIELD3_H_
#define _SJCVECTORFIELD3_H_

#include "SJC.h"

#include <iostream>
#include <iomanip>
#include <assert.h>

#include "SJCField3.h"
#include "SJCScalarField3.h"
#include "SJCVector3.h"
#include "SJCBoundary.h"

class SJCVelocityField3d;

class SJCVectorField3d : public SJCField3d {
 public:
  // Contructor and destructor
  SJCVectorField3d(void);
  // Contructor and destructor
  SJCVectorField3d(const uint nx        = 1,   // The grid size
		   const uint ny        = 1,   
		   const uint nz        = 1,   
		   const double delta_x = 1.0, // The size of the voxel 
		   const double delta_y = 1.0,
		   const double delta_z = 1.0,
		   const SJCBoundary bx = BOUNDARY_NOWRAP,
		   const SJCBoundary by = BOUNDARY_NOWRAP,
		   const SJCBoundary bz = BOUNDARY_NOWRAP,
		   const SJCVector3d *d = 0,
		   const int step	= 3);

  // Input constructor
  SJCVectorField3d(std::istream &f);

  // Copy constructor
  SJCVectorField3d(const SJCVectorField3d &vf);

  // Destructor
  ~SJCVectorField3d(void);

 private:    
  // Clear the data conditions
  void	Destroy(void);

  // Assign the scalar field from vf
  void  Assign(const SJCVectorField3d &vf);

 public:    
  // Assign operator  
  SJCVectorField3d&  operator=(const SJCVectorField3d &vf);

  // Set up information
  void Set(const uint nx        = 1,   // The grid size
	   const uint ny        = 1,   
	   const uint nz        = 1,   
	   const double delta_x = 1.0, // The size of the voxel 
	   const double delta_y = 1.0,
	   const double delta_z = 1.0,
	   const SJCBoundary bx = BOUNDARY_NOWRAP,
	   const SJCBoundary by = BOUNDARY_NOWRAP,
	   const SJCBoundary bz = BOUNDARY_NOWRAP,
	   const SJCVector3d *d      = 0);

  // Compute the vorticity
  void Vorticity(SJCVelocityField3d* velocity);
  // According to the vorticity to define the force
  void VorticityForce(SJCScalarField3d* bound,
		      SJCVectorField3d* vorticity, const double vortCons);
 
  // Get the value by interpolation at x, y position
  SJCVector3d  operator()(const double x, const double y, const double z) { 
    return Value(x, y, z); }
  SJCVector3d  operator()(const SJCVector3d& pos) { 
    return Value(pos.x(), pos.y(), pos.z()); }

  // Get the value by integer
  SJCVector3d& operator()(const uint index_x, const uint index_y, 
			  const uint index_z);
  
  // The scalar field at x, y position
  SJCVector3d	Value(const double x, const double y, const double z);

  // Set up the value at
  void          Value(const int index_x, const int index_y, 
		      const int index_z, const SJCVector3d& value) {
    m_VData[Index(index_x, index_y, index_z)] = value;
  }
 // Get the magnitute
  double Magnitute(const uint index_x, const uint index_y, const uint index_z){
    if(!m_VMagnitute)
      return 0.f;
    return m_VMagnitute[Index(index_x, index_y, index_z)];
  }
  
  // Write out the data to the output stream in binary format
  void          Write(std::ostream &o);
  
  // Output operator  
  friend std::ostream& operator<<(std::ostream &o,
				  const SJCVectorField3d &vf);

 private:

  SJCVector3d*  m_VData;
  double*       m_VMagnitute;

};

#endif
