/************************************************************************
     Main File:

     File:          SimProbability.h

     Author:     
                    Yu-Chi Lai, yu-chi@cs.wisc.edu
  
     Comment:       The PDF evaluation class
                   
     Functions: 
                    1. LogNormalProb: Use the value x to calculate the 
                       probability in log with mean and std
                    2. LogStdNormalProb: Use the valuse x to calcualte the 
                       probability with normal distribution
                    3. LogNormalCircularProb: The circular vector's 
                       probability in deviation
                    4. LogStdNormalCircularProb: The circular vector's 
                       probability in standard deviation
                    5. LogNormalSphericalProb: The spherical vector's 
                       probability in deviation
                    6. LogStdNormalSphericalProb: The spherical vector's 
                       probability in standard deviation
                    7. NormalCumulativeDistribution:
                    8. UniformCumulativeDistribution
                    9. ProbSameNormal 
                   10. ProbSameUniformCircular
  

     Compiler:      g++
 
     Platform:      Linux
*************************************************************************/

#ifndef _SJCPROBABILITY_H_
#define _SJCPROBABILITY_H_

#include <SJC/SJC.h>

// C++ library
#include <ostream>
#include <vector>

#include <SJC/SJCVector3.h>

class SJCProbability {
 private:
  static const float LogSqrt2PI;

 public:
  //*******************************************************************
  // Constructor and destructor
  //*******************************************************************
  SJCProbability(void) {}
  ~SJCProbability(void) { }
    
  // Use the value x to calculate the probability in log with mean and std
  static float LogNormalProb(float x, float mean, float std_dev);
  
  // Use the valuse x to calcualte the probability with normal distribution
  static float LogStdNormalProb(float x);

  // The circular vector's probability in deviation
  static float LogNormalCircularProb(SJCVector3f v, float mean, float std_dev);

  // The circular vector's probability in standard deviation
  static float LogStdNormalCircularProb(SJCVector3f v);

  // The spherical vector's probability in deviation
  static float LogNormalSphericalProb(SJCVector3f v, float mean, 
				      float std_dev);
  
  // The spherical vector's probability in standard deviation
  static float LogStdNormalSphericalProb(SJCVector3f v);

  static float NormalCumulativeDistribution(float x, float mean, 
					    float std_dev);
  
  static float UniformCumulativeDistribution(float x, float min, float max);
  
  static float ProbSameNormal(float *values, uint nvalues, float mean, 
			      float std_dev, float &dist);
  
  static float ProbSameUniformCircular(float *values, uint nvalues, 
				       float &dist);

  // Output operator
  friend std::ostream& operator<<(std::ostream &o, const SJCProbability &r);
};


#endif

