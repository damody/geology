/************************************************************************
     Main File:

     File:        SJCVelocityField2.h

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
  
#ifndef _SJCVELOCITYFIELD2_H_
#define _SJCVELOCITYFIELD2_H_

#include <iostream>
#include <iomanip>
#include <assert.h>

#include "SJC.h"

#include "SJCField2.h"
#include "SJCScalarField2.h"
#include "SJCScalarField2D.h"
#include "SJCVectorField2.h"
#include "SJCVector2.h"
#include "SJCBoundary.h"

class SJCVelocityField2d : public SJCField2d {
 private:
  static const uint m_cuIntegralSteps;
  
 public:
  // The Contructor and destructor
  SJCVelocityField2d(void);
  
  // The main Contructor
  SJCVelocityField2d(const uint x = 1, 
		     const uint y = 1,
		     const double delta_x = 1.0, // The size of the voxel 
		     const double delta_y = 1.0,
		     const SJCBoundary bx = BOUNDARY_NOWRAP,
		     const SJCBoundary by = BOUNDARY_NOWRAP,
		     const SJCVector2d *d = 0);

  // Computed by taking curl pf  
  SJCVelocityField2d(const SJCScalarField2d &pf);  

  // Copy constructor
  SJCVelocityField2d(const SJCVelocityField2d &vf); 

  // Parser
  SJCVelocityField2d(std::istream &f); 
  
  // Destructor
  virtual ~SJCVelocityField2d(void);

 private:
  // Clear the data
  void	Destroy(void);

  // Assign the scalar field from vf
  void  Assign(const SJCVelocityField2d &vf);

  // Calculate the index
  virtual uint Index(const uint index_x, const uint index_y) const { 
    SJCAssert(index_x >= 0 && index_x < m_uNVX &&
	      index_y >= 0 && index_y < m_uNVY, "Fail in 2D velocity field");
    return index_y * m_uNVX + index_x; }

 public:

  // Assign operator
  SJCVelocityField2d&  operator=(const SJCVelocityField2d &vf);

  // Get the velocity in x, y, z
  uint NVX(void) { return m_uNVX;}
  uint NVY(void) { return m_uNVY;}

  void Set(const uint x = 1, 
	   const uint y = 1,
	   const double delta_x = 1.0, // The size of the voxel 
	   const double delta_y = 1.0,
	   const SJCBoundary bx = BOUNDARY_NOWRAP,
	   const SJCBoundary by = BOUNDARY_NOWRAP,
	   const SJCVector2d *d = 0);

  // Computed by taking curl pf  this needs to be checke
  void Set( SJCScalarField2D &pf);  

  // Set up the boundary velocity by copy the nearest cell
  void SetBoundaryByCopy(void);
  

  // Get the x, y value
  SJCVector2d	operator()(const double x, const double y){
    return Value(x, y);}

  SJCVector2d	operator()(const double x, const double y, 
			   const unsigned char r){
    return Value(x, y, r);}
 
  SJCVector2d	operator()(const SJCVector2d& pos){
    return Value(pos.x(), pos.y());}

  // Get the reference at index_x, index_y
  SJCVector2d&  operator()(const uint index_x, const uint index_y){
    return m_VData[Index(index_x, index_y)];
  }
  const SJCVector2d& operator()(const uint index_x, const uint index_y) const {
    return m_VData[Index(index_x, index_y)];
  }
  

  // Put the velocity at the center of the voxel
  SJCVector2d   VoxelCenterVelocity(const uint index_x, const uint index_y);

  // Add the voticity confinement force
  void AddForce(SJCScalarField2d* boundary, 
		SJCVectorField2d* force,
		const double dt);

   // Get the value
  SJCVector2d Value(const double x, const double y);

  // Get the value
  SJCVector2d Value(const double x, const double y, const unsigned char r);

  // Set the value
  void        Value(const uint index_x, const uint index_y, 
		    const SJCVector2d& value) {
    m_VData[Index(index_x, index_y)] = value;
  }
 
  // Get the velocity at i, j
  double& VX(const uint index_x, const uint index_y);
  double& VY(const uint index_x, const uint index_y);
 
  // Get the coordinate for the velocity
  void VXCoord(const uint index_x, const uint index_y, SJCVector2d& pos){
    pos.set((double)index_x * m_dDX, ((double)index_y + 0.5) * m_dDY);
  }
  void VYCoord(const uint index_x, const uint index_y, SJCVector2d& pos){
    pos.set(((double)index_x + 0.5) * m_dDX, (double)index_y * m_dDY);
  }
  
  // Trace the particle back  
  void    TraceParticle(const double dt, const SJCVector2d curr, 
			SJCVector2d& prev);

  // Get the divergence at voxel index (x,y)
  double  Divergence(const uint index_x, const uint index_y);
  
  // Write out the data to the output stream in binary format
  void    Write(std::ostream &o);
  // Read from a binary file
  bool    Read(std::istream &f);

  // Output operator
  friend std::ostream& operator<<(std::ostream &o, 
				  const SJCVelocityField2d &vf);

 private:
  // Read from a binary file
  void    ReadCommon(std::istream &f);

 private:
  uint          m_uNVX;           // the real dimension for velocity
  uint          m_uNVY;
  
  SJCVector2d*  m_VData;

};

#endif
