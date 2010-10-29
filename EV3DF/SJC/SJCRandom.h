/************************************************************************
     Main File:

     File:        SimRandom.h

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     The random variable operation

     Constructors:
                  1. 1 : Set up the idum and calculate the following
                  2. 5 : Set up all preset variable
                  3. 1 : Set up assignment
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2. <<: output operator
                 3. seed(): return the seed
                 4. long Uniform(min, max): return the long nuber between min 
                    and max which is uniformly distributed
                 5. double Uniform(min, max): return the double nuber between 
                    min and max which is uniformly distributed
                 6. SimVector Circular(radius): return a std::vector with radius 
                    in 2D
                 7. NormalCircular: Generate the 2D direction with normalize 
                    distribution
                 8. SimVector Spherical(radius): return a std::vector with 
                    radius in 3D
                 9. NormalSpherical: Generate the 3D direction with normalize 
                    distribution
                10. Real Normal(mean, std_dev): return a normal distribution 
                    number
                11. RandomPermutation: Randomly permutate the number 
                    from 0 ~ permute.size()-1
                12. RandomSelect: Randomly select n number from the 0 ~ num
                13. RandomProbability: According to probabilities, randomly 
                    generate a number between start to  size - 1
                14. RandomProbability: According to probabilities, randomly 
                    generate a number between start to size - 1 and it will 
                    count for those with probability  below threshold
                15. RandomProbability: According to probabilities, randomly 
                    generate a number between start to  size - 1
                16. ProbabilityPermutation: Permutation according to 
                    probabilities 
                17. ProbabilityPermutation: Permutation according to 
                    probabilities but it will kick out those below threshold 
                    thus permute.size <= probs.size()
                18. HemisphereNormalTheta: Samples the hemisphere according to 
                    normal distribution 
                19. SphereUniformHypercube: A simple way to randomly (uniform) 
                    distribute points on sphere is called the "hypercube 
                    rejection method". to apply this to a unit cube at the 
                    origin, choose coordinates (x,y,z) each uniformly 
                    distributed on the interval [-1,1]. If the length of this 
                    std::vector is greater than 1 then reject it, otherwise 
                    normalise it and use it as a sample.
      source: http://astronomy.swin.edu.au/~pbourke/geometry/spherepoints/
                20. SphereUniform: Samples the hemisphere according to a 
                    uniform distribution i.e.  proportional to hemisphere area,
                    if will generate random variable itself.
                21. HemisphereUniform: Samples the hemisphere according to a 
                    uniform distribution i.e. proportional to hemisphere area, 
                    if will generate random variable itself.
                22. HemisphereCosTheta: Samples the hemisphere according to a 
                    cos_theta distribution 
                23. HemisphereCosNTheta: Samples the hemisphere according to a 
                    cos_theta ^ n  distribution 
                24. GetNrDivisions:  Defines a nice grid for stratified 
                    sampling where divs1 * divs2 = samples and  divs1 as close 
                    to divs2 as possible
                25. << : Output operator
   
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#ifndef _SJCRANDOM_H_
#define _SJCRANDOM_H_

#include "SJC.h"
#include "SJCConstants.h"


#include <algorithm>
#include <ostream>
#include <vector>

#include "SJCVector3.h"
#include "SJCVector2.h"
#include "SJCQuaternion.h"
#include "SJCRotateMatrix.h"

class SJCDLL SJCRandomd {
 private:
  long        	idum;    // The seed
  long	        iy;
  long 	        iv[32];  // Initial std::vector
  int       	iset;
  double    	gset;
  
  static const long 	IA;
  static const long 	IM;
  static const double	AMA;
  static const long 	IQ;
  static const long 	IR;
  static const long 	NTAB;
  static const long 	NDIVA;
  static const double EPS;
  static const double	RNMX;

 public:
  //*******************************************************************
  // Constructor and destructor
  //*******************************************************************
  SJCRandomd(const long seed = 1);
  SJCRandomd(const long idum_in, const long iy_in, const long iv_in[32],
	    const int iset_in, const double gset_in);
  SJCRandomd(const SJCRandomd &s);
  ~SJCRandomd(void) { }

  // Assign operator
  SJCRandomd& operator=(const SJCRandomd &s);

  // Return the seed
  long 	Seed(void) const { return idum; }
  void	Seed(const long seed);

  // Uniform distribution, possible from Long_min to long_max - 1
  long        Uniform(const long min = LONG_MIN, const long max = LONG_MAX);
  double      Uniform(const double min = 0.0, const double max = 1.0);

  // Genereate the 2D normalized direction with uniform distribution
  SJCVector2d  Circular(const double radius = 1.0);
  // Generate the 2D direction with normalize distribution
  SJCVector3d   NormalCircular(const double radius = 1.0) ; 

  // Generate the 3D direction with uniform distribution
  SJCVector3d   Spherical(const double radius = 1.0);
  // Generate the 3D direction with normalized distribution
  SJCVector3d   NormalSpherical(void) ; 

  double      Normal(const double mean, const double std_dev);


  template<class T> void	Permute(std::vector<T> &v) {
    for ( uint i = v.size() - 1 ; i > 0 ; i-- )     {
      uint	ind = Uniform(0l, (long)i + 1);
      swap(v[ind], v[i]);
    }
  }
  // Randomly permutate the number from 0 ~ permute.size()-1
  void   RandomPermutation(std::vector<uint>& permute);
  
  // Randomly select n number from the 0 ~ num
  void   RandomSelect(std::vector<uint>& select, uint uselect, uint num);
  
  // According to probabilities, randomly generate a number between start to 
  // size - 1
  uint   RandomProbability(std::vector<double>& probs, uint start);
  
  // According to probabilities, randomly generate a number 
  // between start to size - 1 and it will count for those with probability 
  // below threshold
  uint   RandomProbability(std::vector<double>& probs, uint start,
			   double threshold);
  
  // According to probabilities, randomly generate a number between start to 
  // size - 1
  uint   RandomProbability(const double* probs, uint prob_size, uint start);
  
  // Permutation according to probabilities
  void   ProbabilityPermutation(std::vector<uint>& permute,
				std::vector<double>& probs);
  
  // Permutation according to probabilities but it will kick out those
  // below threshold thus permute.size <= probs.size()
  void   ProbabilityPermutation(std::vector<uint>&  permute,
				std::vector<double>& probs,
				double threshold);
 public: // for the rendering
  // * Samples the hemisphere according to normal distribution 
  SJCVector3d   HemisphereNormalTheta(SJCVector3d& direction,
				      double mean,
				      double std_dev);
 
  // * A simple way to randomly (uniform) distribute points on sphere is 
  //   called the "hypercube rejection method". to apply this to a unit cube 
  //   at the origin, choose coordinates (x,y,z) each uniformly distributed 
  //   on the interval [-1,1]. If the length of this std::vector is greater 
  //   than 1 then reject it, otherwise normalise it and use it as a sample.
  //   source: http://astronomy.swin.edu.au/~pbourke/geometry/spherepoints/
  SJCVector3d SphereUniformHypercube(double& pdf_value);
  
  // * Samples the hemisphere according to a uniform distribution i.e. 
  //   proportional to hemisphere area, if will generate random variable 
  //   itself.
  SJCVector3d SphereUniform(SJCVector3d& dump1, 
			    SJCVector3d& dump2,
			    double& pdf_value);
  
  // * Samples the hemisphere according to a uniform distribution 
  //   i.e. proportional to hemisphere area, if will generate random
  //   variable itself.
  SJCVector3d HemisphereUniform(SJCVector3d& leavingsurfInDir,
			        SJCVector3d& normal,
			        double& pdf_value);
  
    // * Samples the hemisphere according to a cos_theta distribution 
  SJCVector3d HemisphereCosTheta(SJCVector3d& leavingsurfInDir,
			         SJCVector3d& normal, 
			         double& pdf_value);
  
  // * Samples the hemisphere according to a cos_theta ^ n  distribution 
  SJCVector3d HemisphereCosNTheta(SJCVector3d& normal, 
				  SJCVector3d& idealOut, 
				  double nOrder,
				  double& pdf_value);

    // * Defines a nice grid for stratified sampling where 
    //   divs1 * divs2 = samples and  divs1 as close to divs2 as possible
    void GetNrDivisions(int samples, int& divs1, int& divs2);
  
  friend std::ostream&    operator<<(std::ostream &o, const SJCRandomd &r);
};


class SJCDLL SJCRandomf {
 private:
  long        	idum;    // The seed
  long	        iy;
  long 	        iv[32];  // Initial std::vector
  int       	iset;
  float    	gset;
  
  static const long 	IA;
  static const long 	IM;
  static const float	AMA;
  static const long 	IQ;
  static const long 	IR;
  static const long 	NTAB;
  static const long 	NDIVA;
  static const float EPS;
  static const float	RNMX;

 public:
  //*******************************************************************
  // Constructor and destructor
  //*******************************************************************
  SJCRandomf(const long seed = 1);
  SJCRandomf(const long idum_in, const long iy_in, const long iv_in[32],
	    const int iset_in, const float gset_in);
  SJCRandomf(const SJCRandomf &s);
  ~SJCRandomf(void) { }

  // Assign operator
  SJCRandomf& operator=(const SJCRandomf &s);

  // Return the seed
  long 	Seed(void) const { return idum; }
  void	Seed(const long seed);

  // Uniform distribution, possible from Long_min to long_max - 1
  long       Uniform(const long min = LONG_MIN, const long max = LONG_MAX);
  float      Uniform(const float min = 0.0, const float max = 1.0);

  // Genereate the 2D normalized direction with uniform distribution
  SJCVector2f  Circular(const float radius = 1.0);

  // Generate the 2D direction with normalize distribution
  SJCVector3f   NormalCircular(const float radius = 1.0) ; 

  // Generate the 3D direction with uniform distribution
  SJCVector3f   Spherical(const float radius = 1.0);
  // Generate the 3D direction with normalized distribution
  SJCVector3f   NormalSpherical(void) ; 

  float      Normal(const float mean, const float std_dev);


  template<class T> void	Permute(std::vector<T> &v) {
    for ( uint i = v.size() - 1 ; i > 0 ; i-- )     {
      uint	ind = Uniform(0l, (long)i + 1);
      swap(v[ind], v[i]);
    }
  }
  // Randomly permutate the number from 0 ~ permute.size()-1
  void   RandomPermutation(std::vector<uint>& permute);
  
  // Randomly select n number from the 0 ~ num
  void   RandomSelect(std::vector<uint>& select, uint uselect, uint num);
  
  // According to probabilities, randomly generate a number between start to 
  // size - 1
  uint   RandomProbability(std::vector<float>& probs, uint start);
  
  // According to probabilities, randomly generate a number 
  // between start to size - 1 and it will count for those with probability 
  // below threshold
  uint   RandomProbability(std::vector<float>& probs, uint start,
			   float threshold);
  
  // According to probabilities, randomly generate a number between start to 
  // size - 1
  uint   RandomProbability(const float* probs, uint prob_size, uint start);
  
  // Permutation according to probabilities
  void   ProbabilityPermutation(std::vector<uint>& permute,
				std::vector<float>& probs);
  
  // Permutation according to probabilities but it will kick out those
  // below threshold thus permute.size <= probs.size()
  void   ProbabilityPermutation(std::vector<uint>&  permute,
				std::vector<float>& probs,
				float threshold);
 public: // for the rendering
 
  // * Samples the hemisphere according to normal distribution 
  SJCVector3f   HemisphereNormalTheta(SJCVector3f& direction,
				      float mean,
				      float std_dev);
 
  // * A simple way to randomly (uniform) distribute points on sphere is 
  //   called the "hypercube rejection method". to apply this to a unit cube 
  //   at the origin, choose coordinates (x,y,z) each uniformly distributed 
  //   on the interval [-1,1]. If the length of this std::vector is greater 
  //   than 1 then reject it, otherwise normalise it and use it as a sample.
  //   source: http://astronomy.swin.edu.au/~pbourke/geometry/spherepoints/
  SJCVector3f SphereUniformHypercube(float& pdf_value);
  
  // * Samples the hemisphere according to a uniform distribution i.e. 
  //   proportional to hemisphere area, if will generate random variable 
  //   itself.
  SJCVector3f SphereUniform(SJCVector3f& dump1, 
			    SJCVector3f& dump2,
			    float& pdf_value);
  
  // * Samples the hemisphere according to a uniform distribution 
  //   i.e. proportional to hemisphere area, if will generate random
  //   variable itself.
  SJCVector3f HemisphereUniform(SJCVector3f& leavingsurfInDir,
			        SJCVector3f& normal,
			        float& pdf_value);
  
    // * Samples the hemisphere according to a cos_theta distribution 
  SJCVector3f HemisphereCosTheta(SJCVector3f& leavingsurfInDir,
			         SJCVector3f& normal, 
			         float& pdf_value);
  
  // * Samples the hemisphere according to a cos_theta ^ n  distribution 
  SJCVector3f HemisphereCosNTheta(SJCVector3f& normal, 
				  SJCVector3f& idealOut, 
				  float nOrder,
				  float& pdf_value);

    // * Defines a nice grid for stratified sampling where 
    //   divs1 * divs2 = samples and  divs1 as close to divs2 as possible
    void GetNrDivisions(int samples, int& divs1, int& divs2);
  
  friend std::ostream&    operator<<(std::ostream &o, const SJCRandomf &r);
};


#endif

