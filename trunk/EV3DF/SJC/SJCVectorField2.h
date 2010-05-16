/************************************************************************
     Main File:

     File:        SJCVectorField2.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
  
     Comment:     To handle the 2D std::vector field which value is stored at center

     Constructors:
                  1. 0 : the default contructor
                  2. 6 : constructor to set up all value by input parameters
                  3. 1 : set up the class by using the scalar field
                  4. 1 : copy contructor
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2. (): Get the value of the scalar field
                 3. >>: output in the ascii form
 ************************************************************************/
  
#ifndef _SJC_VECTORFIELD2_H_
#define _SJC_VECTORFIELD2_H_

#include "SJC.h"

#include <iostream>
#include <iomanip>
#include <assert.h>
#include <string.h>

#include "SJCField2.h"
#include "SJCScalarField2.h"
#include "SJCVector2.h"
#include "SJCBoundary.h"

class SJCVelocityField2d;

class SJCVectorField2d : SJCField2d {
 public:
  // Contructor and destructor
  SJCVectorField2d(void);
  // The main constructor
  SJCVectorField2d(const uint   nx      = 1, 
		   const uint   ny      = 1,
		   const double delta_x = 1.0,
		   const double delta_y = 1.0,
		   const SJCBoundary bx = BOUNDARY_NOWRAP,
		   const SJCBoundary by = BOUNDARY_NOWRAP,
		   const SJCVector2d* d = 0);

  // Input constructor
  SJCVectorField2d(std::istream &f);

  // Vortecity
  SJCVectorField2d(const SJCScalarField2d &vf);

  // Copy constructor
  SJCVectorField2d(const SJCVectorField2d &vf);

  // Destructor
  ~SJCVectorField2d(void);
  


 private:

  // Clear the data conditions
  void  Destroy(void);

  // Assign the scalar field from vf
  void  Assign(const SJCVectorField2d &vf);

 public:  
  // Assign operator
  SJCVectorField2d&  operator=(const SJCVectorField2d &vf);

  // Set up information
  void Set(const uint nx        = 1,   // The grid size
	   const uint ny        = 1,   
	   const double delta_x = 1.0, // The size of the voxel 
	   const double delta_y = 1.0,
	   const SJCBoundary bx = BOUNDARY_NOWRAP,
	   const SJCBoundary by = BOUNDARY_NOWRAP,
	   const SJCVector2d* d = 0);

  // Compute the vorticity from velocity field
  void Vorticity(SJCVelocityField2d* velocity);

  // According to the vorticity to define the force
  void VorticityForce(SJCScalarField2d* bound,
		      SJCVectorField2d* vorticity, const double vortCons);
  
  // Get the reference at index_x, index_y
  SJCVector2d&	operator()(const uint index_x, const uint index_y){
    return m_VData[Index(index_x, index_y)]; 
  } 

  // Get the std::vector at x, y
  SJCVector2d   operator()(const double x, const double y) {
    return Value(x, y);
  }
  SJCVector2d   operator()(const SJCVector2d& pos) {
    return Value(pos.x(), pos.y());
  }

  // The scalar field at x, y position
  SJCVector2d Value(const double x, const double y);

  // Set up the scalar field at index_x,y
  void Value(const uint index_x, const uint index_y, const SJCVector2d& value){
    m_VData[Index(index_x, index_y)] = value;
  }
 
  // Get the magnitute
  double Magnitute(const uint index_x, const uint index_y) {
    if(!m_VMagnitute)
      return 0.f;
    return m_VMagnitute[Index(index_x, index_y)];
  }
  
  // Write out the data to the output stream in binary format
  void    Write(std::ostream &o);

  // Output operator
  friend std::ostream& operator<<(std::ostream &o, const SJCVectorField2d &vf);

  // Reset to 0
  void Reset(void){ memset(m_VData, 0, sizeof(SJCVector2d)); }
  

 private:
  SJCVector2d*  m_VData;        // The data
  double*       m_VMagnitute;   // The magnitute for the data
};

#endif
