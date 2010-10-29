/************************************************************************
Main File:   


File:		LevelSet.h

Author:    
		Yu-Chi Lai, yu-chi@mail.ntust.edu.tw
Comment:     

Compiler:	Microsoft Visual Studio

Platform:	Window
*************************************************************************/

#ifndef LEVELSET_H_
#define LEVELSET_H_
#pragma warning(disable:4127)
#include "SJC.h"
#include "SJCScalarField3.h"
#include "SJCVelocityField3.h"
#include "SJCVectorField3.h"

// Pre-declared
class CParticle;
class CParticleSet;

//*************************************************************************
//
// This levelset will be defined on the 8 corners of each voxel
//
//**************************************************************************
class CLevelSet3d : public SJCScalarField3d
{
    public:
	// Constructor
	CLevelSet3d(int nx, int ny, int nz) 
		: SJCScalarField3d( nx,  ny,  nz, 
				   1.f, 1.f, 1.f, 
				   BOUNDARY_NOWRAP,
				   BOUNDARY_NOWRAP,
				   BOUNDARY_NOWRAP,
				   0) {}
	// 
	void Move(SJCVelocityField3d *pVelocity, double dt, CLevelSet3d *pOut);

	// 
	void Fix(CParticleSet *pParticleSet);

	/*
	void   Coord(const uint index_x, const uint index_y, const uint index_z, SJCVector3d& pos){
		pos.set( (double)(index_x) * m_dDX, 
			 (double)(index_y) * m_dDY,
			 (double)(index_z) * m_dDZ );
			}
	*/
    private:
        // Fix the positive boundary
	void FixPositive(CParticle *pParticle);

	// Fix the negative boundary
	void FixNegative(CParticle *pParticle);
};

#endif
