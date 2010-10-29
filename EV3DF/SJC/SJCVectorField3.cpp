/************************************************************************
Main File:

File:        SJCVectorField3.cpp

Author:     
Yu-Chi Lai, yu-chi@cs.wisc.edu
Steven Chenney, schenney@cs.wisc.edu

Comment:     Class to handle the scalar field

Constructors:
1. 0 : the default contructor
2. 2 :
3. 3 : 

Functions: 
1. = : Assign operator which copy the parameter of random
2.
************************************************************************/

#include "SJCVectorField3.h"
#include "SJCVelocityField3.h"

//****************************************************************************
//
// * Contructor to set up all value
// * step on use for increasing not 1
// * step 用在非1遞增的位移中，一次跳一個double的大小
//============================================================================
SJCVectorField3d::
SJCVectorField3d(const uint nx,        const uint ny,       const uint nz,
		 const double dx,      const double dy,     const double dz,
		 const SJCBoundary bx, const SJCBoundary by, 
		 const SJCBoundary bz, const SJCVector3d *d, const int step)
		 : SJCField3d(nx, ny, nz, dx, dy, dz, bx, by, bz)
		 //============================================================================
{
	m_VMagnitute = 0;
	m_VData = new SJCVector3d[m_uNX * m_uNY * m_uNZ];
	if (d) {
		double* pd = (double*) d;
		for ( uint i = 0 ; i < m_uNX * m_uNY * m_uNZ ; i++ )
			memcpy(&m_VData[i], &pd[i*step], sizeof(double)*3);
	}
}


//****************************************************************************
//
// * Get the scalar field from input
//============================================================================
SJCVectorField3d::
SJCVectorField3d(std::istream &f)
: SJCField3d(f)
//============================================================================
{
	m_VMagnitute = 0;
	m_VData = new SJCVector3d[m_uNX * m_uNY * m_uNZ];
	f.read((char*)m_VData, m_uNX * m_uNY * m_uNZ * sizeof(SJCVector3d));
}



//****************************************************************************
//
// * Copy contructor
//============================================================================
SJCVectorField3d::
SJCVectorField3d(const SJCVectorField3d &vf)
: SJCField3d(vf)
//============================================================================
{
	Assign(vf);
}

//****************************************************************************
//
// * Copy contructor
//============================================================================
void SJCVectorField3d::
Assign(const SJCVectorField3d &vf)
//============================================================================
{
	SJCField3d::Assign(vf);

	m_VData = new SJCVector3d[m_uNX * m_uNY * m_uNZ];
	for ( uint i = 0 ; i < m_uNX * m_uNY * m_uNZ ; i++ )
		m_VData[i] = vf.m_VData[i];
	if(m_VMagnitute){
		m_VMagnitute = new double[m_uNX * m_uNY * m_uNZ];
		for ( uint i = 0 ; i < m_uNX * m_uNY * m_uNZ ; i++ )
			m_VMagnitute[i] = vf.m_VMagnitute[i];
	}
}



//****************************************************************************
//
// * Destructor
//============================================================================
SJCVectorField3d::
~SJCVectorField3d(void)
//============================================================================
{
	Destroy();
}


//****************************************************************************
//
// * Clear the m_VData
//============================================================================
void SJCVectorField3d::
Destroy(void)
//============================================================================
{
	SJCField3d::Destroy();

	// Release the data
	delete [] m_VData;
	m_VData   = 0;

	if(m_VMagnitute)
		delete [] m_VMagnitute;
	m_VMagnitute = 0;

}

//****************************************************************************
//
// * Assign operator
//============================================================================
SJCVectorField3d& SJCVectorField3d::
operator=(const SJCVectorField3d &vf)
//============================================================================
{
	// Clear data
	Destroy();

	// Do the copy
	Assign(vf);

	return *this;
}
//****************************************************************************
//
// * Set up the value
//============================================================================
void SJCVectorField3d::
Set(const uint nx, const uint ny, const uint nz,
    const double dx, const double dy, const double dz,
    const SJCBoundary bx, const SJCBoundary by, const SJCBoundary bz,
    const SJCVector3d *d)  
    //============================================================================
{
	Destroy();

	SJCField3d::Set(nx, ny, nz, dx, dy, dz, bx, by, bz);
	m_VData = new SJCVector3d[m_uNX * m_uNY * m_uNZ];
	if (d) {
		for ( uint i = 0 ; i < m_uNX * m_uNY * m_uNZ; i++ )
			m_VData[i] = d[i];
	}
	m_VMagnitute = 0;
}

//****************************************************************************
//
// * Set up the value from the velocity field
//============================================================================
void SJCVectorField3d::
Vorticity(SJCVelocityField3d* velocity)
//============================================================================
{
	// Check whether the magniture has been generated
	if(!m_VMagnitute){
		m_VMagnitute = new double[m_uNX * m_uNY * m_uNZ];
		for( uint i = 0 ; i < m_uNX * m_uNY * m_uNZ; i++){
			m_VMagnitute[i] = 0;
		}
	}


	// Go through all valid element 
	for(uint k = 1; k < m_uNZ - 1; k++){
		for(uint j = 1; j < m_uNY - 1; j++){
			for(uint i = 1; i < m_uNX - 1; i++){

				double vx = 
					( velocity->VoxelCenterVelocity(i, j+1, k).z() - 
					velocity->VoxelCenterVelocity(i, j-1, k).z()   ) / m_dDY * .5 
					- ( velocity->VoxelCenterVelocity(i, j, k+1).y() - 
					velocity->VoxelCenterVelocity(i, j, k-1).y()   ) / m_dDZ * .5;

				double vy = 
					( velocity->VoxelCenterVelocity(i, j, k+1).x() - 
					velocity->VoxelCenterVelocity(i, j, k-1).x()   ) / m_dDZ * .5 
					- ( velocity->VoxelCenterVelocity(i+1, j, k).z() -
					velocity->VoxelCenterVelocity(i-1, j, k).z()   ) / m_dDX * .5;

				double vz = 
					( velocity->VoxelCenterVelocity(i+1, j, k).y() -
					velocity->VoxelCenterVelocity(i-1, j, k).y()   ) / m_dDX * .5 
					- ( velocity->VoxelCenterVelocity(i, j+1, k).x()-
					velocity->VoxelCenterVelocity(i, j-1, k).x()   ) / m_dDY * .5;

				m_VData[Index(i, j, k)].set(vx, vy, vz);
				m_VMagnitute[Index(i, j, k)] = m_VData[Index(i, j, k)].length();
			} // end of for i
		} // end of for j
	}// end of for k
}
//****************************************************************************
//
// * Set up the force value for vorticity
//============================================================================
void SJCVectorField3d::
VorticityForce(SJCScalarField3d* bound, 
	       SJCVectorField3d* vorticity, const double vortCons)
	       //============================================================================
{
	// Go through the valid element
	for(uint k = 1; k < m_uNZ - 1; k++){
		for(uint j = 1; j < m_uNY - 1; j++){
			for(uint i = 1; i < m_uNX -1; i++){
				if ((*bound)(i, j, k) <= 0) {

					// Compute the Normal component which gradient of the 
					double x = (vorticity->Magnitute(i+1, j, k) - 
						vorticity->Magnitute(i-1, j, k)) * 0.5 / m_dDX;
					double y = (vorticity->Magnitute(i ,j+1, k) - 
						vorticity->Magnitute(i, j-1, k)) * 0.5 / m_dDY;
					double z = (vorticity->Magnitute(i, j, k+1) - 
						vorticity->Magnitute(i, j, k-1)) * 0.5 / m_dDZ;

					double norm = sqrt(x * x + y * y + z * z);
					if (norm != 0) {
						SJCVector3d N(x / norm, y / norm, z / norm);
						m_VData[Index(i, j, k)] = vortCons * (N % (*vorticity)(i, j, k));
					}
					else {
						m_VData[Index(i, j, k)].set( 0.f, 0.f, 0.f);
					}
				} // end of if boudn
				else {
					m_VData[Index(i, j, k)].set( 0.f, 0.f, 0.f);
				} // end of else 
			} // end of for i
		} // end of for j
	}// end of for k


}

//****************************************************************************
//
// * According to the boundary get the value at x, y, z
//============================================================================
SJCVector3d SJCVectorField3d::
Value(const double x_pos, const double y_pos, const double z_pos)
//============================================================================
{
	// Check whether the position is out of bound
	if(x_pos < MinX() || x_pos > MaxX() || 
		y_pos < MinY() || y_pos > MaxY() ||
		z_pos < MinZ() || z_pos > MaxZ())  {
			SJCWarning("X, Y, Z out of bound in getting value");

			return SJCVector3d(0.f, 0.f, 0.f);
	}

	// Compute the index position
	double x = (x_pos - m_dHalfDX) / m_dDX;
	double y = (y_pos - m_dHalfDY) / m_dDY;
	double z = (z_pos - m_dHalfDZ) / m_dDZ;

	uint	   index_xl, index_xr;  // the left and right index in x
	uint	   index_yb, index_yt;  // The top and bottom index in y
	uint	   index_zb, index_zf;  // The front and back index in z
	double   partial_x, partial_y, partial_z; // The partial in x, y, z

	// Here should have judgement when x_pos and y_pos are close to index 
	// position
	switch ( m_VBoundary[0] )   { // Check and set up the correct value for x,y 
		// according the boundary conditions
    case BOUNDARY_NOWRAP: 
    case BOUNDARY_NOWRAP_FREE:
	    if ( x == m_uNX - 1.f) {
		    index_xr  = (uint)m_uNX - 1;
		    index_xl  = index_xr - 1;
		    partial_x = 1.0f;
	    }
	    else {
		    index_xl  = (uint)floor(x);
		    index_xr  = index_xl + 1;
		    partial_x = x - (double)index_xl;
	    }
	    break;

    case BOUNDARY_WRAP: 
	    double	xp = fmod(x, (double)m_uNX);
	    if ( xp < 0.0 )
		    xp = xp + (int)m_uNX;

	    index_xl  = (uint)floor(xp);
	    index_xr  = (index_xl + 1) % m_uNX;
	    partial_x = xp - (double)index_xl;
	    break;
	} // end of switch

	switch ( m_VBoundary[1] )   {
    case BOUNDARY_NOWRAP:
    case BOUNDARY_NOWRAP_FREE:
	    if ( y == m_uNY - 1.f ) {
		    index_yt  = (uint)m_uNY - 1;
		    index_yb  = index_yt - 1;
		    partial_y = 1.0f;
	    }
	    else {
		    index_yb  = (uint)floor(y);
		    index_yt  = index_yb + 1;
		    partial_y = y - (double)index_yb;
	    }
	    break;

    case BOUNDARY_WRAP: 
	    double 	yp = fmod(y, (double)m_uNY);
	    if ( yp < 0.0 )
		    yp = yp + (int)m_uNY;

	    index_yb  = (uint)floor(yp);
	    index_yt  = (index_yb + 1) % m_uNY;
	    partial_y = yp - (double)index_yb;
	    break;
	}
	switch ( m_VBoundary[2] )   {
    case BOUNDARY_NOWRAP_FREE:
    case BOUNDARY_NOWRAP:
	    if ( z == m_uNZ - 1.f ) {
		    index_zf  = (uint)m_uNZ - 1;
		    index_zb  = index_zf - 1;
		    partial_z = 1.0f;
	    }
	    else {
		    index_zb  = (uint)floor(z);
		    index_zf  = index_zb + 1;
		    partial_z = z - (double)index_zb;
	    }
	    break;

    case BOUNDARY_WRAP: 
	    double 	zp = fmod(z, (double)m_uNZ);
	    if ( zp < 0.0 )
		    zp = zp + (int)m_uNZ;

	    index_zb  = (uint)floor(zp);
	    index_zf  = (index_zb + 1) % m_uNZ;
	    partial_z = zp - (double)index_zb;
	    break;
	}

	SJCVector3d vLeftBottomBack   = m_VData[Index(index_xl, index_yb, index_zb)];
	SJCVector3d vLeftBottomFront  = m_VData[Index(index_xl, index_yb, index_zf)];
	SJCVector3d vLeftTopBack      = m_VData[Index(index_xl, index_yt, index_zb)];
	SJCVector3d vLeftTopFront     = m_VData[Index(index_xl, index_yt, index_zf)];
	SJCVector3d vRightBottomBack  = m_VData[Index(index_xr, index_yb, index_zb)];
	SJCVector3d vRightBottomFront = m_VData[Index(index_xr, index_yb, index_zf)];
	SJCVector3d vRightTopBack     = m_VData[Index(index_xr, index_yt, index_zb)];
	SJCVector3d vRightTopFront    = m_VData[Index(index_xr, index_yt, index_zf)];

	return (1.f-partial_x)*(1.f-partial_y)*(1.f-partial_z) * vLeftBottomBack+
		(1.f-partial_x)*(1.f-partial_y)*partial_z       * vLeftBottomFront +
		(1.f-partial_x)*partial_y      *(1.f-partial_z) * vLeftTopBack +
		(1.f-partial_x)*partial_y      *partial_z       * vLeftTopFront +
		partial_x      *(1.f-partial_y)*(1.f-partial_z) * vRightBottomBack+
		partial_x      *(1.f-partial_y)*partial_z       * vRightBottomFront+
		partial_x      *partial_y      *(1.f-partial_z) * vRightTopBack +
		partial_x      *partial_y      *partial_z       * vRightTopFront ;
}

//****************************************************************************
//
// * Access the value at (x, y)
//============================================================================
SJCVector3d& SJCVectorField3d::
operator()(const uint index_x, const uint index_y, const uint index_z)
//============================================================================
{
	return m_VData[Index(index_x, index_y, index_z)];
}



//****************************************************************************
//
// * Output operator
//============================================================================
std::ostream&
operator<<(std::ostream &o, const SJCVectorField3d &vf)
//============================================================================
{
	o << (SJCField3d&)vf;

	for(uint k = 0; k < vf.m_uNZ - 1; k++) {
		for ( uint j = 0; j < vf.m_uNY - 1 ; j++ ) {
			for ( uint i = 0 ; i < vf.m_uNX ; i++ ) {
				o << std::setw(8) 
					<< vf.m_VData[i * vf.m_uNY * vf.m_uNZ + j * vf.m_uNZ + k] << " ";
			} // end of for i
		} // end of for k
	} // end of for j
	o << std::endl;
	return o;
}


//****************************************************************************
//
// * Write out the data in binary format
//============================================================================
void
SJCVectorField3d::Write(std::ostream &o)
//============================================================================
{
	SJCField3d::Write(o);

	o.write((char*)m_VData, m_uNX * m_uNY * m_uNZ * sizeof(SJCVector3d));
}


