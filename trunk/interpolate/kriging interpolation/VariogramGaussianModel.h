// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#ifndef _VARIOGRAM_MODEL_GAUSSIAN_H
#define	_VARIOGRAM_MODEL_GAUSSIAN_H

#include <vector>
#include <nr.h>
typedef std::vector<double> vectord;

//variogram Gaussian model function
double Gaussian(double n, double s, double r, double power, double d);

//variogram Gaussian model derivative function for normal c++
void GaussianValueDerivative(const  double dist, 
							 vectord& a, double& yFit, vectord& dyda);

//variogram Gaussian model derivative function for normal Numerical Recipes
void GaussianValueDerivative(const  DP dist, Vec_I_DP& a, 
							   DP& yFit, Vec_O_DP& dyda);

#endif

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)
