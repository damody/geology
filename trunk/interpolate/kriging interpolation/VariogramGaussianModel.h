
#ifndef _VARIOGRAM_MODEL_GAUSSIAN_H
#define	_VARIOGRAM_MODEL_GAUSSIAN_H

#include <vector>
#include <nr.h>
typedef std::vector<double> vectord;

double Gaussian(double n, double s, double r, double power, double d);
void GaussianValueDerivative(const  double dist, 
							 vectord& a, double& yFit, vectord& dyda);

void GaussianValueDerivative(const  DP dist, Vec_I_DP& a, 
							   DP& yFit, Vec_O_DP& dyda);

#endif
