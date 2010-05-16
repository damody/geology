#include "LevelSet.h"
#include "Particle.h"

//****************************************************************************
//
// *
//============================================================================
void CLevelSet3d::
Move(SJCVelocityField3d *pVelocity, double dt, CLevelSet3d *pOut)
//============================================================================
{
	// Current and previous position
	SJCVector3d curr;
	SJCVector3d prev;

	// *pOut = *this;
	// SJCVectorField3d grad=this->GradField();
	for(uint i=0;i<this->NumX();i++) {
		for(uint j=0;j<this->NumY();j++) {
			for(uint k=0;k<this->NumZ();k++) {
				// ( dphi/dt + u . del(phi) = 0 )
				// ( dphi/dt = - u . del(phi)
				// (*pOut)(i,j,k)+=-((*pVelocity)(i,j,k)*grad(i,j,k))*dt;
				this->Coord(i,j,k,curr);
				pVelocity->TraceParticle(-dt, curr, prev);
				/*
				if (prev.z()<this->BoundMinZ()) {
					cout << "{" << curr.x() <<","<<curr.y()<<","<<curr.z()<<"} => ";
					cout << "{" << prev.x() <<","<<prev.y()<<","<<prev.z()<<"}"<<endl;
				}
				*/
				(*pOut)(i,j,k)=(*this)(prev);
			} // end of k
		} // end of j
	}// end of i
}

//****************************************************************************
//
// *
//============================================================================
void CLevelSet3d::
Fix(CParticleSet *pParticleSet)
//============================================================================
{
	CParticle *pParticle=pParticleSet->Reset();
	while(pParticle!=NULL) {
		if (pParticle->Radius()<0) {
			if ((*this)(*pParticle)>0) //We've got an escapee!
				FixNegative(pParticle);
		}
		else {
			if ((*this)(*pParticle)<0)
				FixPositive(pParticle);
		}
		pParticle=pParticle->Next();
	}

    /*
    // Now smooth:

    double x, y, z, phi, gradmag, smoothed_phi;
    double timestep_sqr = 1.0;

    for (int iterations = 0; iterations < 1; iterations++) {
        for (int i = 2; i < NumX() - 2; i++) {
            for (int j = 2; j < NumY() - 2; j++) {
                for (int k = 2; k < NumZ() - 2; k++) {
                    x = MinX() + i * DX();
                    y = MinY() + j * DY();
                    z = MinZ() + k * DZ();
                    phi = Value(x, y, z);

                    if (fabs(phi) < 3.0) {
                        gradmag = Grad(x, y, z).length();
                        smoothed_phi = (phi / ((phi*phi) + timestep_sqr)) * (gradmag - 1.0);

                        this->Value(i,j,k, smoothed_phi);
                    }
                }
            }
        }
    }
    */
}

//****************************************************************************
//
// *
//============================================================================
void CLevelSet3d::
FixPositive(CParticle *pParticle)
//============================================================================
{
	double		x, y, z;
	SJCVector3d	pos;
	const double	*dPos = pParticle->get();
	uint startX = (uint) (dPos[0] - pParticle->Radius()) / this->DX();
	uint startY = (uint) (dPos[1] - pParticle->Radius()) / this->DY();
	uint startZ = (uint) (dPos[2] - pParticle->Radius()) / this->DZ();
	uint endX   = (uint) (dPos[0] + pParticle->Radius()) / this->DX() + 1;
	uint endY   = (uint) (dPos[1] + pParticle->Radius()) / this->DY() + 1;
	uint endZ   = (uint) (dPos[2] + pParticle->Radius()) / this->DZ() + 1;
	startX      = max(startX, (uint) 0);
	startY      = max(startY, (uint) 0);
	startZ      = max(startZ, (uint) 0);
	endX        = min(endX, NumX());
	endY        = min(endY, NumY());
	endZ        = min(endZ, NumZ());
	for (uint i=startX; i < endX; i++) {
		for (uint j=startY;j<endY;j++){
			for (uint k=startZ;k<endZ;k++)	{
				x = i * DX();
				y = j * DY();
				z = k * DZ();
		                this->Value(i,j,k, max(pParticle->Phi(x,y,z), (*this)(i,j,k)));
			} // end of for k
		} // end of for j
	} // end of for i
}

//****************************************************************************
//
// *
//============================================================================
void CLevelSet3d::
FixNegative(CParticle *pParticle)
//============================================================================
{
	uint i,j,k;
	double x, y, z;
	SJCVector3d pos;
	const double *dPos=pParticle->get();
	uint startX=(uint) (dPos[0]+pParticle->Radius())/this->DX();
	uint startY=(uint) (dPos[1]+pParticle->Radius())/this->DY();
	uint startZ=(uint) (dPos[2]+pParticle->Radius())/this->DZ();
	uint endX=(uint) (dPos[0]-pParticle->Radius())/this->DX()+1;
	uint endY=(uint) (dPos[1]-pParticle->Radius())/this->DY()+1;
	uint endZ=(uint) (dPos[2]-pParticle->Radius())/this->DZ()+1;
	startX=max(startX, (uint) 0);
	startY=max(startY, (uint) 0);
	startZ=max(startZ, (uint) 0);
	endX=min(endX, NumX());
	endY=min(endY, NumY());
	endZ=min(endZ, NumZ());
	for (i=startX;i<endX;i++)
	{
		for (j=startY;j<endY;j++)
		{
			for (k=startZ;k<endZ;k++)
			{
                x = i * DX();
                y = j * DY();
                z = k * DZ();
                this->Value(i,j,k, min(pParticle->Phi(x,y,z), (*this)(i,j,k)));
			}
		}
	}
}