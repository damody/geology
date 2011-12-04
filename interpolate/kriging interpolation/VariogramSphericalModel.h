
#ifndef	LYC_VARIOGRAM_MODEL_SPHERICAL_H
#define	LYC_VARIOGRAM_MODEL_SPHERICAL_H

#include <vector>
#include <nr.h>
typedef std::vector<double> vectord;
double Spherical(double n, double s, double r, double p, double d);
void   SphericalValueDerivative(const  double dist, 
								vectord& a, double& yFit, vectord& dyda);
void   SphericalValueDerivative(const  DP dist, Vec_I_DP& a, 
								DP& yFit, Vec_O_DP& dyda);
#endif
