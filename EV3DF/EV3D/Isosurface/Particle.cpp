
#include "Particle.h"
#include "LevelSet.h"

#include <vector>

using std::max;
using std::min;
//****************************************************************************
//
// * Default  constructor
//============================================================================
CParticle::
CParticle(void)
//============================================================================
{
	m_pPrev = NULL;
	m_pNext = NULL;
}

//****************************************************************************
//
// * Constructor
//============================================================================
CParticle::
CParticle(CParticle *pPrev, CParticle *pNext)
//============================================================================
{
	m_pPrev = pPrev;
	m_pNext = pNext;
}

//****************************************************************************
//
// * Constructor
//============================================================================
CParticle::
CParticle(double x, double y, double z, double dRadius, 
	  CParticle *pPrev, CParticle *pNext)
//============================================================================
{
	set(x, y, z);
	m_dRadius=dRadius;
	m_pPrev=pPrev;
	m_pNext=pNext;
}

//****************************************************************************
//
// * Constructor
//============================================================================
CParticle::
CParticle(double x, double y, double z, double dRadius)
//============================================================================
{
	set(x, y, z);
	m_dRadius = dRadius;
	m_pPrev = NULL;
	m_pNext = NULL;
}

//****************************************************************************
//
// * Destructor
//============================================================================
CParticle::
~CParticle(void)
//============================================================================
{
	if (m_pPrev)
		m_pPrev->SetNext(m_pNext);
	if (m_pNext)
		m_pNext->SetPrev(m_pPrev);
}

//****************************************************************************
//
// * Compute the phi which is the level set value
//============================================================================
double CParticle::
Phi(SJCVector3d &pos)
//============================================================================
{
	if (Radius() > 0)
		return Radius() - distance(pos);
	else
		return Radius() + distance(pos);
}

//****************************************************************************
//
// *
//============================================================================
double CParticle::
Phi(double x, double y, double z)
//============================================================================
{
	SJCVector3d pt(x, y, z);
	return Phi(pt);
}

//****************************************************************************
//
// * Constructor to reset everything to null
//============================================================================
CParticleSet::
CParticleSet(void)
//============================================================================
{
	m_pFirst = NULL;
	m_pLast  = NULL;
	m_pCur   = NULL;
}

//****************************************************************************
//
// * Destructor clear up the data
//============================================================================
CParticleSet::
~CParticleSet(void)
//============================================================================
{
	m_pCur=m_pFirst;
	CParticle *pTmp;
	while(m_pCur!=NULL) {
		pTmp   = m_pCur;
		m_pCur = m_pCur->Next();
		delete pTmp;
	}
}

//****************************************************************************
//
// * Add a particle at the end
//============================================================================
void CParticleSet::
Add(CParticle *pParticle)
//============================================================================
{
	if (m_pLast)
		m_pLast->SetNext(pParticle);
	else {
		m_pFirst=pParticle;
		m_pCur=m_pFirst;
	}

	pParticle->SetPrev(m_pLast);
	pParticle->SetNext(NULL);
	m_pLast = pParticle;
}

//****************************************************************************
//
// * Add a particle at the end
//============================================================================
void CParticleSet::
Add(double x, double y, double z, double dRadius)
//============================================================================
{
	CParticle *pNew=new CParticle(x, y, z, dRadius);
	Add(pNew);
}

//****************************************************************************
//
// * Insert a particle after the current one
//============================================================================
void CParticleSet::
Insert(CParticle *pParticle)
//============================================================================
{
	if (m_pCur) {
		m_pCur->SetNext(pParticle);
		m_pCur->Next()->SetPrev(pParticle);
	}
	else {
		m_pFirst=pParticle;
		m_pLast=m_pFirst;
	}

	m_pCur=pParticle;
}

//****************************************************************************
//
// * Create a particle and insert next to the current one
//============================================================================
void CParticleSet::
Insert(double x, double y, double z, double dRadius)
//============================================================================
{
	CParticle *pNew = new CParticle(x, y, z, dRadius);
	Insert(pNew);
}

//****************************************************************************
//
// * Move the particle according to the std::vector field and the time
//============================================================================
void CParticleSet::
Move(SJCVelocityField3d *pVel, double dt)
//============================================================================
{
	CParticle *pTmp = m_pFirst;
	CParticle pNew;
	while(pTmp) {
		pVel->TraceParticle(dt, *pTmp, *pTmp);
		pTmp=pTmp->Next();
	}
}

//****************************************************************************
//
// * Find the particle which is closest to the current one
//============================================================================
CParticle *CParticleSet::
FindNearest(SJCVector3d &pos)
//============================================================================
{
	CParticle *pTmp = m_pFirst;
	CParticle *pMin = NULL;		//Particle to return
	double     dMin = -1;
	double     dDist;

	// Go through the entire list to find the one with minimum distance
	while(pTmp != NULL){
		dDist = pTmp->distance(pos);

		if (dMin < 0 || dDist < dMin) {
			dMin = dDist;
			pMin = pTmp;
		}
		pTmp = pTmp->Next();
	}
	return pMin;
}

//****************************************************************************
//
// * Find a particle which is closest to the current one
//============================================================================
CParticle *CParticleSet::
FindNearest(CParticle &particle)
//============================================================================
{
	SJCVector3d pos = (SJCVector3d)particle;
	return FindNearest(pos);
}

//****************************************************************************
//
// * Find a particle that is closest to the position x, y, z
//============================================================================
CParticle *CParticleSet::
FindNearest(const double x, const double y, const double z)
//============================================================================
{
	SJCVector3d pos(x, y, z);
	return FindNearest(pos);
}

//****************************************************************************
//
// * Initialize the partileset
//============================================================================
void CParticleSet::
Initialize(CLevelSet3d *pLevelSet)
//============================================================================
{
	for(float x = pLevelSet->MinX(); x < pLevelSet->MaxX(); x += pLevelSet->DX()){
		for(float y = pLevelSet->MinY(); y < pLevelSet->MaxY();	y += pLevelSet->DY()){
			for(float z = pLevelSet->MinZ(); z < pLevelSet->MaxZ();	z += pLevelSet->DZ()){
				if (fabs(pLevelSet->Value(x,y,z)) < 3. * pLevelSet->DX()) {
					//randomly scatter some particles inside this voxel
					for(int i = 0; i < 4; i++) {
						double dx=((double) rand() / RAND_MAX - .5f) * pLevelSet->DX();
						double dy=((double) rand() / RAND_MAX - .5f) * pLevelSet->DY();
						double dz=((double) rand() / RAND_MAX - .5f) * pLevelSet->DZ();

						double px = max(x+dx, pLevelSet->BoundMinX());
						px = min(px, pLevelSet->BoundMaxX());

						double py = max(y+dy, pLevelSet->BoundMinY());
						py = min(py, pLevelSet->BoundMaxY());
						
						double pz = max(z+dz, pLevelSet->BoundMinZ());
						pz = min(pz, pLevelSet->BoundMaxZ());

						Add(px, py, pz, pLevelSet->Value(px, py, pz));
					} // end of for i
				} // end of if
			} // end of for z
		} // end of for y
	} // end of for x
}

//****************************************************************************
//
// * Reinitialize the particle set with the corresponding radius
//============================================================================
void CParticleSet::
ReInitialize(CLevelSet3d *pLevelSet)
//============================================================================
{
	CParticle *pCur = m_pFirst;
	while(pCur != NULL) {
		pCur->SetRadius((*pLevelSet)(*pCur));
		pCur = pCur->Next();
	}
}

