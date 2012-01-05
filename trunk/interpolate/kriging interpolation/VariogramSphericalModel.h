// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#ifndef	LYC_VARIOGRAM_MODEL_SPHERICAL_H
#define	LYC_VARIOGRAM_MODEL_SPHERICAL_H

#include <vector>
#include <nr.h>
typedef std::vector<double> vectord;

//variogram Spherical model function
double Spherical(double n, double s, double r, double p, double d);

//variogram Spherical model derivative function for normal c++
void   SphericalValueDerivative(const  double dist, 
								vectord& a, double& yFit, vectord& dyda);

//variogram Spherical model derivative function for normal Numerical Recipes
void   SphericalValueDerivative(const  DP dist, Vec_I_DP& a, 
								DP& yFit, Vec_O_DP& dyda);
#endif

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)
