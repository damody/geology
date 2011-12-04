

#include <math.h>
#include <vector>
#include "VariogramGaussianModel.h"

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
double Gaussian(double n, double s, double r, double power, double d)
//============================================================================
{
	double ratio = d / r;
	double t = ratio * ratio;
	if(t > 700.0)
		return n + s;
	double res = n + s * (1.0 - exp(-t));
	return res;
}


//****************************************************************************
//
// * Set up the vector b
//   a: nuggest, sill, range, power
//============================================================================
void GaussianValueDerivative(const  double dist, 
							 vectord& a, double& yFit, vectord& dyda)
//============================================================================
{
	// Compute the estimate value
	yFit = Gaussian(a[0], a[1], a[2], a[3], dist);

	double ratio = dist / a[2];
	double t = 3 * ratio * ratio;

	// Compute the exponential value
	double e = exp(-t);

	dyda[0] = 1.0;												// df / d(nugget)
	dyda[1] = 1.0 - e;										    // df / d(sill)
	dyda[2] = -a[1] * 6.0 * e * pow(ratio, 2) / a[2];			// df / d(range)

}

void GaussianValueDerivative( const DP dist, Vec_I_DP& a, DP& yFit, Vec_O_DP& dyda )
{
	// Compute the estimate value
	yFit = Gaussian(a[0], a[1], a[2], a[3], dist);

	double ratio = dist / a[2];
	double t = ratio * ratio;

	// Compute the exponential value
	double e = exp(-t);

	dyda[0] = 1.0;												// df / d(nugget)
	dyda[1] = 1.0 - e;										    // df / d(sill)
	dyda[2] = -a[1] * 2.0 * e * pow(ratio, 2) / a[2];			// df / d(range)
}

