
#ifndef	_VARIOGRAM_MODEL_STABLE_H_
#define	_VARIOGRAM_MODEL_STABLE_H_

#include <vector>
#include <nr.h>
typedef std::vector<double> vectord;

double Stable(double n, double s, double r, double power, double d);
void StableValueDerivative(const  double dist, 
						   vectord& a, double& yFit, vectord& dyda);
void StableValueDerivative(const  DP dist, Vec_I_DP& a, DP& yFit, 
						   Vec_O_DP& dyda);

#endif


