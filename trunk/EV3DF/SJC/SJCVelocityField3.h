/************************************************************************
     Main File:

     File:        SJCVelocityField3.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     To handle the 2D velocity field for MAC format
                  V(i, j, k).x = Vx(i + 0.5, j, k);
                  V(i, j, k).y = Vy(i, j + 0.5, k);
  
     Constructors:
                  1. 0 : the default contructor
                  2. 4 : constructor to set up all value by input parameters
                  3. 1 : copy contructor
                  4. 1 : Set up from the scalar field
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2. (): Get the value of the scalar field
                 3. VX: get the reference of X component of i, j
                 4. VY: get the reference of Y component of i, j
                 5. MakeDivergenceFree: make the velocity field divergence free
                 6. MinX, MinY, MaxX, MaxY: get the maximum and minimum value 
                    of X, y
                 7. NumX, NumY: get the number of sample points in X, Y
                 8. >>: output in the ascii form
 ************************************************************************/
  
#ifndef _SJCVELOCITYFIELD3_H_
#define _SJCVELOCITYFIELD3_H_

#include <iostream>
#include <iomanip>
#include <assert.h>

#include "SJC.h"

#include "SJCField3.h"
#include "SJCScalarField3.h"
#include "SJCVectorField3.h"
#include "SJCVector3.h"
#include "SJCBoundary.h"

class SJCVelocityField3d : public SJCField3d {
 private:
  static const uint m_cuIntegralSteps;
  
 public:
  // The Contructor and destructor
  SJCVelocityField3d(void);
  
  // The main Contructor
  SJCVelocityField3d(const uint nx        = 1,   // The grid size
		     const uint ny        = 1,   
		     const uint nz        = 1,   
		     const double delta_x = 1.0, // The size of the voxel 
		     const double delta_y = 1.0,
		     const double delta_z = 1.0,
		     const SJCBoundary bx = BOUNDARY_NOWRAP,
		     const SJCBoundary by = BOUNDARY_NOWRAP,
		     const SJCBoundary bz = BOUNDARY_NOWRAP,
		     const SJCVector3d *d      = 0);
 
  // Computed by taking curl pf  
  SJCVelocityField3d(const SJCScalarField3d &pf);  

  // Copy constructor
  SJCVelocityField3d(const SJCVelocityField3d &vf); 
  
  // Destructor
  ~SJCVelocityField3d(void);

 private:
  // Clear the data
  void	Destroy(void);

  // Assign the scalar field from vf
  void  Assign(const SJCVelocityField3d &vf);

  // Calculate the index
  virtual uint Index(const uint index_x, const uint index_y, 
		     const uint index_z) { 
    SJCAssert(index_x >= 0 && index_x < m_uNVX &&
	      index_y >= 0 && index_y < m_uNVY &&
	      index_z >= 0 && index_z < m_uNVZ, "Fail in 3D velocity field");
    return index_z * m_uNVX * m_uNVY + index_y * m_uNVX + index_x; }

 public:
  // Assign operator
  SJCVelocityField3d&  operator=(const SJCVelocityField3d &vf);

  // Get the velocity in x, y, z
  uint NVX(void) { return m_uNVX;}
  uint NVY(void) { return m_uNVY;}
  uint NVZ(void) { return m_uNVZ;}

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

  // Get the value by interpolation at x, y position
  SJCVector3d  operator()(const double x, const double y, const double z) { 
    return Value(x, y, z); }

  // Get the value by interpolation at x, y position
  SJCVector3d  operator()(const SJCVector3d& pos) { 
    return Value(pos.x(), pos.y(), pos.z()); }

  // Get the value by integer
  SJCVector3d& operator()(const uint index_x, const uint index_y, 
			  const uint index_z) {
    return m_VData[Index(index_x, index_y, index_z)];
  }

  // Get the value
  SJCVector3d Value(const double x, const double y, const double z);
  // Set the value
  void        Value(const uint index_x, const uint index_y, 
		    const uint index_z, const SJCVector3d& value) {
    m_VData[Index(index_x, index_y, index_z)] = value;
  }
  
  // Put the velocity at the center of the voxel
  SJCVector3d   VoxelCenterVelocity(const uint index_x, const uint index_y,
				    const uint index_z);
      
  // Add the voticity confinement force
  void AddForce(SJCScalarField3d* boundary,
		SJCVectorField3d* force, const double dt);

  // Get the velocity at i, j, k
  double& VX(const uint index_x, const uint index_y, const uint index_z);
  double& VY(const uint index_x, const uint index_y, const uint index_z);
  double& VZ(const uint index_x, const uint index_y, const uint index_z);
  
  // Get the coordinate for the velocity
  void VXCoord(const uint index_x, const uint index_y, const uint index_z,
	       SJCVector3d& pos){
    pos.set((double)index_x * m_dDX, ((double)index_y + 0.5) * m_dDY, 
	    ((double)index_z + 0.5) * m_dDZ);
  }
  void VYCoord(const uint index_x, const uint index_y, const uint index_z,
	       SJCVector3d& pos){
    pos.set(((double)index_x + 0.5) * m_dDX, (double)index_y * m_dDY, 
	    ((double)index_z + 0.5) * m_dDZ);
  }
  void VZCoord(const uint index_x, const uint index_y, const uint index_z,
	       SJCVector3d& pos){
    pos.set(((double)index_x + 0.5) * m_dDX, ((double)index_y + 0.5) * m_dDY, 
	    ((double)index_z) * m_dDY);
  }

  // Trace the particle back  
  void    TraceParticle(const double dt, const SJCVector3d curr, 
			SJCVector3d& prev);

  // Get the divergence at voxel index (x,y,z)
  double  Divergence(const uint index_x, const uint index_y, 
		     const uint index_z);
  
  // Write out the data to the output stream in binary format
  void    Write(std::ostream &o);

  // Output operator
  friend std::ostream& operator<<(std::ostream &o, 
				  const SJCVelocityField3d &vf);
 private:
  uint          m_uNVX;           // the real dimension for velocity
  uint          m_uNVY;
  uint          m_uNVZ;

  SJCVector3d*  m_VData;

};

#endif
