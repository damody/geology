#include <math.h>
#include <vector>
#include "VariogramStableModel.h"


//****************************************************************************
//
// * Stable model variograms: get the estimated dissimilarity value using
//   stable model
// 1. n [in] nugget
// 2. s [in] sill
// 3. r [in] range
// 4. p [in] power coefficient of stable models
// 5. d [in] distance
// 6. return dissimilarity
//============================================================================
double Stable(double n, double s, double r, double power, double dist)
//============================================================================
{
	double t = fabs(pow(dist / r, power));
// 	if (t >= 700.0)
// 		return n + s;
	return n + s * (1.0 - exp(-t));
}

//****************************************************************************
//
// * Set up the vector b
//   a: nuggest, sill, range, power
//============================================================================
void StableValueDerivative(const  double dist, 
						   vectord& a, double& yFit, vectord& dyda)
//============================================================================
{
	// Compute the estimate value
	yFit = Stable(a[0], a[1], a[2], a[3], dist);

	// Compute the exponential value
	double ratio = dist / a[2];
	double e = exp(-pow(ratio, a[3]));

	dyda[0] = 1.0;												// df / d(nugget)
	dyda[1] = 1.0 - e;										    // df / d(sill)
	dyda[2] = - a[1] * a[3] * pow(ratio, a[3]) / a[2] * e;      // df / d(range)
}

void StableValueDerivative( const DP dist, Vec_I_DP& a, DP& yFit, Vec_O_DP& dyda )
{
	// Compute the estimate value
	yFit = Stable(a[0], a[1], a[2], a[3], dist);

	// Compute the exponential value
	double ratio = fabs(dist / a[2]);
	double e = exp( -pow(ratio, a[3]) );

	dyda[0] = 1.0;												// df / d(nugget)
	dyda[1] = 1.0 - e;										    // df / d(sill)
	dyda[2] = - a[1] * a[3] * pow(ratio, a[3]-1) / a[2] * e;      // df / d(range)

}