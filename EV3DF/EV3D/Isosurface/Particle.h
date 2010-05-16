/************************************************************************
	Main File:   


	File:		Particle.h

	Author:    
			Yu-Chi Lai, yu-chi@mail.ntust.edu.tw
	Comment:     
		Class CParticle: for particle levelset operation
		constructor:
		(x, y, z, dRadius, *pPrev, *pNext): set up the position 
		        radius and prev and next particle
		(x, y, z, dRadius): add to the end with x, y, z, r
		(pPrev, pNext): add a particle to previous and end

		Functions
		1. Next(): Get the next particle
		2. Prev(): Get the previous particle
		3. Radius(): Retrieve the radius
		4. SetRadius(r): Set up the radius
		5. SetNext(CParticle*), SetPrev(CParticle*): set up the previous and next pointer
		6. Phi(x, y, z), Phi(SJCVector3d): Compute the phi which is the level set value

	Compiler:	Microsoft Visual Studio

	Platform:	Window
*************************************************************************/


#include <vector>

#include "SJC.h"
#include "LevelSet.h"

#ifndef PARTICLE_H_
#define PARTICLE_H_

// Convention: negative radius for interior points, positive radius for exterior

class CLevelSet;

//****************************************************************************
//
// *
//
//****************************************************************************
class CParticle	:	public SJCVector3d
{
    public:
        // Constructor
	CParticle(double x, double y, double z, double dRadius, 
		  CParticle *pPrev, CParticle *pNext);
	
	// Constructor
	CParticle(double x, double y, double z, double dRadius);

	// Constructor
	CParticle(CParticle *pPrev, CParticle *pNext);

	// Default contructor
	CParticle(void);

	// Destructor
	~CParticle(void);

    public:
        // Get the previous particle
	CParticle *Next(void){return m_pNext;}

	// Get the next particle
	CParticle *Prev(void){return m_pPrev;}

	// Retrieve the radius
	double     Radius(void){return m_dRadius;}

	// Set up the radius
	void	   SetRadius(double dRad){m_dRadius=dRad;}

	// Set up the proper pointer
	void	   SetNext(CParticle *pNext){m_pNext=pNext;};
	void	   SetPrev(CParticle *pPrev){m_pPrev=pPrev;};

	// Compute the phi which is the level set value
	double	   Phi(double x, double y, double z);
	double	   Phi(SJCVector3d &pos);
private:
	CParticle	*m_pNext;    // Pointer to form a double linked list
	CParticle	*m_pPrev;
	double		 m_dRadius;  // Radius of the particle
};

//****************************************************************************
//
// * Set for the particles
//
//****************************************************************************
class CParticleSet
{
    public:
        // Default Constructor and destructor
 	CParticleSet(void);
	~CParticleSet(void);

	// Add new particle into the set at the end
	void       Add(CParticle *particle);
	void       Add(double  x, double y, double z, double dRad);

	// Insert the particle next to the current one
	void       Insert(CParticle *particle);
	void       Insert(double x, double y, double z, double dRad);

	// Move the particle according 
	void	   Move(SJCVelocityField3d *velocity, double dt);

	// Get the current active particle 
	CParticle *Current(void){return m_pCur;}

	// Grab the next particle on the list
	CParticle *Next(void){m_pCur = m_pCur->Next(); return m_pCur;}

	// Grab the previous particle on the list
	CParticle *Prev(void){m_pCur = m_pCur->Prev(); return m_pCur;}

	// Go back to the first particle and return it
	CParticle *Reset(void){m_pCur = m_pFirst; return m_pCur;}

	// Find the nearest particle 
	CParticle *FindNearest(CParticle &particle);
	CParticle *FindNearest(SJCVector3d &pos);
	CParticle *FindNearest(double x, double y, double z);

	// This function isn't really filled in, since I'm confused about 
	// why you'd use it
	// void ReInitialize(CLevelSet3d *pLevelSet);
	// Set up the radius of each particle to the radius of the particle in pLevelset
	void	   ReInitialize(CLevelSet3d *pLevelSet);

	// Initialize the particles in the particleset according to the levelset
	void	   Initialize(CLevelSet3d *pLevelSet);

	// Grab the first particle
	CParticle *GetFirstParticle(void){ return m_pFirst;}
    private:
	CParticle *m_pFirst; // The first particle
	CParticle *m_pLast;  // The last particle
	CParticle *m_pCur;   // The current particle
};

#endif //PARTICLE_H_