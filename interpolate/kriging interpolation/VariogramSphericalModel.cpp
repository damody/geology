
#include <math.h>
#include <vector>
#include "VariogramSphericalModel.h"
//****************************************************************************
//
// *  spherical variogram model: according to the distance to compute the
//    dissimilarity from spherical variogram model
// 
// 1. n [in] nugget
// 2. s [in] sill
// 3. r [in] range
// 4. d [in] distance
// 
// return dissimilarity
//============================================================================
double Spherical(double n, double s, double r, double p, double d)
//============================================================================
{
	if (d > r)
		return n + s;
	double sp = fabs(d) / r;
	return n + s * (3.0 / 2.0 * sp  - pow(sp, 3.0) / 2);
}


//****************************************************************************
//
// * Set up the vector b
//   a: nuggest, sill, range, power
//============================================================================
void SphericalValueDerivative(const  double dist, 
							  vectord& a, double& yFit, vectord& dyda)
//============================================================================
{
	// Compute the estimate value
	yFit = Spherical(a[0], a[1], a[2], a[3], dist);

	if(dist > a[2]){
		dyda[0] = 1.0;  // df / d(nugget)
		dyda[1] = 1.0;  // df / d(sill)
		dyda[2] = 0.0;  // df / d(range)
	}
	else{
		double sp = fabs(dist) / a[2];
		double tp = pow(sp, 3.0);
		dyda[0] = 1.0;											  // df / d(nugget)
		dyda[1] = 1.5 * sp - 0.5 * tp ;					  // df / d(sill)
		dyda[2] = (1.5 * sp - 1.5 * tp) / a[2];  // df / d(range)

	}
}

void SphericalValueDerivative( const DP dist, Vec_I_DP& a, DP& yFit, Vec_O_DP& dyda )
{
	// Compute the estimate value
	yFit = Spherical(a[0], a[1], a[2], a[3], dist);

 	if(dist > a[2]){
		dyda[0] = 1.0;  // df / d(nugget)
		dyda[1] = 1.0;  // df / d(sill)
		dyda[2] = 0.0;  // df / d(range)
	}
	else{
		double sp = fabs(dist) / a[2];
		double tp = pow(sp, 3.0);
		dyda[0] = 1.0;											  // df / d(nugget)
		dyda[1] = 1.5 * sp - 0.5 * tp ;					  // df / d(sill)
		dyda[2] = a[1]*(-1.5 * sp + 1.5 * tp) / a[2];  // df / d(range)

	}
}

