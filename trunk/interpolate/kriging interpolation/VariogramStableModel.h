// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#ifndef	_VARIOGRAM_MODEL_STABLE_H_
#define	_VARIOGRAM_MODEL_STABLE_H_

#include <vector>
#include <nr.h>
typedef std::vector<double> vectord;

//variogram stable model function
double Stable(double n, double s, double r, double power, double d);

//variogram stable model derivative function for normal c++
void StableValueDerivative(const  double dist, 
						   vectord& a, double& yFit, vectord& dyda);

//variogram stable model derivative function for normal Numerical Recipes
void StableValueDerivative(const  DP dist, Vec_I_DP& a, DP& yFit, 
						   Vec_O_DP& dyda);

#endif

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)