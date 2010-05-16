/************************************************************************
     Main File:

 
     File:        SJCProbability.cpp


     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu

   
     Comment:     The PDF evaluation class


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

     Compiler:    g++


     Platform:    Linux
*************************************************************************/

#include <math.h>
#include <SJCProbability.h>

const float     SJCProbability::LogSqrt2PI = -0.5 * log(M_PI*2.0);

//***************************************************************************
//
// * Calculate the probability in log with x and mean, standard deviaiton
//==========================================================================
float SJCProbability::
LogNormalProb(float x, float mean, float std_dev)
//==========================================================================
{
  // This *may* return a constant multiple of the probability,  or a 
  // constant added to the log of the probability, as the case may be.
  if ( std_dev <= 0.0 )
    return ( WithinEpsilon(mean, x) ? static_cast<float>(0.0)
	     : static_cast<float>(-1.e20) );

  return static_cast<float>(LogSqrt2PI - log(std_dev) - 
			     0.5 * (x - mean) * (x - mean)
			     / (std_dev * std_dev));
}

//***************************************************************************
//
// * Calculate the log probability with mean 0 and std 1
//==========================================================================
float SJCProbability::
LogStdNormalProb(float x) 
//==========================================================================
{
  return LogNormalProb(x, 0, 1);
}

//***************************************************************************
//
// * 
//==========================================================================
float SJCProbability::LogNormalCircularProb(SJCVector3f v, float mean, 
					float std_dev) 
//==========================================================================
{
  return LogNormalProb(v.length(), mean, std_dev);
}

//***************************************************************************
//
// * 
//==========================================================================
float SJCProbability::
LogStdNormalCircularProb(SJCVector3f v) 
//==========================================================================
{
  return LogNormalProb(v.length(), 0, 1);
}

//***************************************************************************
//
// * 
//==========================================================================
float SJCProbability::
LogNormalSphericalProb(SJCVector3f v, float mean, float std_dev) 
//==========================================================================
{
  return LogNormalProb(v.length(), mean, std_dev);
}


//***************************************************************************
//
// * 
//==========================================================================
float SJCProbability::
LogStdNormalSphericalProb(SJCVector3f v) 
//==========================================================================
{
  return LogNormalProb(v.length(), 0, 1);
}


//==========================================================================
// Values of the normal cumulative distribution function from x=0 to x=4 in
// 0.01 steps. Actually, you need to add 0.5 to all the values to take into
// account the left half of the function. This is done in the access function
// below.
//==========================================================================
static float cumul_distrib_samples[] = {
  0.00000f, 0.00399f, 0.00798f, 0.01197f, 0.01595f,
  0.01994f, 0.02392f, 0.02790f, 0.03188f, 0.03586f,
  0.03983f, 0.04380f, 0.04776f, 0.05172f, 0.05567f,
  0.05962f, 0.06356f, 0.06749f, 0.07142f, 0.07535f,
  0.07926f, 0.08317f, 0.08706f, 0.09095f, 0.09483f,
  0.09871f, 0.10257f, 0.10642f, 0.11026f, 0.11409f,
  0.11791f, 0.12172f, 0.12552f, 0.12930f, 0.13307f,
  0.13683f, 0.14058f, 0.14431f, 0.14803f, 0.15173f,
  0.15542f, 0.15910f, 0.16276f, 0.16640f, 0.17003f,
  0.17364f, 0.17724f, 0.18082f, 0.18439f, 0.18793f,
  0.19146f, 0.19497f, 0.19847f, 0.20194f, 0.20540f,
  0.20884f, 0.21226f, 0.21566f, 0.21904f, 0.22240f,
  0.22575f, 0.22907f, 0.23237f, 0.23565f, 0.23891f,
  0.24215f, 0.24537f, 0.24857f, 0.25175f, 0.25490f,
  0.25804f, 0.26115f, 0.26424f, 0.26730f, 0.27035f,
  0.27337f, 0.27637f, 0.27935f, 0.28230f, 0.28524f,
  0.28814f, 0.29103f, 0.29389f, 0.29673f, 0.29955f,
  0.30234f, 0.30511f, 0.30785f, 0.31057f, 0.31327f,
  0.31594f, 0.31859f, 0.32121f, 0.32381f, 0.32639f,
  0.32894f, 0.33147f, 0.33398f, 0.33646f, 0.33891f,
  0.34134f, 0.34375f, 0.34614f, 0.34849f, 0.35083f,
  0.35314f, 0.35543f, 0.35769f, 0.35993f, 0.36214f,
  0.36433f, 0.36650f, 0.36864f, 0.37076f, 0.37286f,
  0.37493f, 0.37698f, 0.37900f, 0.38100f, 0.38298f,
  0.38493f, 0.38686f, 0.38877f, 0.39065f, 0.39251f,
  0.39435f, 0.39617f, 0.39796f, 0.39973f, 0.40147f,
  0.40320f, 0.40490f, 0.40658f, 0.40824f, 0.40988f,
  0.41149f, 0.41308f, 0.41466f, 0.41621f, 0.41774f,
  0.41924f, 0.42073f, 0.42220f, 0.42364f, 0.42507f,
  0.42647f, 0.42785f, 0.42922f, 0.43056f, 0.43189f,
  0.43319f, 0.43448f, 0.43574f, 0.43699f, 0.43822f,
  0.43943f, 0.44062f, 0.44179f, 0.44295f, 0.44408f,
  0.44520f, 0.44630f, 0.44738f, 0.44845f, 0.44950f,
  0.45053f, 0.45154f, 0.45254f, 0.45352f, 0.45449f,
  0.45543f, 0.45637f, 0.45728f, 0.45818f, 0.45907f,
  0.45994f, 0.46080f, 0.46164f, 0.46246f, 0.46327f,
  0.46407f, 0.46485f, 0.46562f, 0.46638f, 0.46712f,
  0.46784f, 0.46856f, 0.46926f, 0.46995f, 0.47062f,
  0.47128f, 0.47193f, 0.47257f, 0.47320f, 0.47381f,
  0.47441f, 0.47500f, 0.47558f, 0.47615f, 0.47670f,
  0.47725f, 0.47778f, 0.47831f, 0.47882f, 0.47932f,
  0.47982f, 0.48030f, 0.48077f, 0.48124f, 0.48169f,
  0.48214f, 0.48257f, 0.48300f, 0.48341f, 0.48382f,
  0.48422f, 0.48461f, 0.48500f, 0.48537f, 0.48574f,
  0.48610f, 0.48645f, 0.48679f, 0.48713f, 0.48745f,
  0.48778f, 0.48809f, 0.48840f, 0.48870f, 0.48899f,
  0.48928f, 0.48956f, 0.48983f, 0.49010f, 0.49036f,
  0.49061f, 0.49086f, 0.49111f, 0.49134f, 0.49158f,
  0.49180f, 0.49202f, 0.49224f, 0.49245f, 0.49266f,
  0.49286f, 0.49305f, 0.49324f, 0.49343f, 0.49361f,
  0.49379f, 0.49396f, 0.49413f, 0.49430f, 0.49446f,
  0.49461f, 0.49477f, 0.49492f, 0.49506f, 0.49520f,
  0.49534f, 0.49547f, 0.49560f, 0.49573f, 0.49585f,
  0.49598f, 0.49609f, 0.49621f, 0.49632f, 0.49643f,
  0.49653f, 0.49664f, 0.49674f, 0.49683f, 0.49693f,
  0.49702f, 0.49711f, 0.49720f, 0.49728f, 0.49736f,
  0.49744f, 0.49752f, 0.49760f, 0.49767f, 0.49774f,
  0.49781f, 0.49788f, 0.49795f, 0.49801f, 0.49807f,
  0.49813f, 0.49819f, 0.49825f, 0.49831f, 0.49836f,
  0.49841f, 0.49846f, 0.49851f, 0.49856f, 0.49861f,
  0.49865f, 0.49869f, 0.49874f, 0.49878f, 0.49882f,
  0.49886f, 0.49889f, 0.49893f, 0.49896f, 0.49900f,
  0.49903f, 0.49906f, 0.49910f, 0.49913f, 0.49916f,
  0.49918f, 0.49921f, 0.49924f, 0.49926f, 0.49929f,
  0.49931f, 0.49934f, 0.49936f, 0.49938f, 0.49940f,
  0.49942f, 0.49944f, 0.49946f, 0.49948f, 0.49950f,
  0.49952f, 0.49953f, 0.49955f, 0.49957f, 0.49958f,
  0.49960f, 0.49961f, 0.49962f, 0.49964f, 0.49965f,
  0.49966f, 0.49968f, 0.49969f, 0.49970f, 0.49971f,
  0.49972f, 0.49973f, 0.49974f, 0.49975f, 0.49976f,
  0.49977f, 0.49978f, 0.49978f, 0.49979f, 0.49980f,
  0.49981f, 0.49981f, 0.49982f, 0.49983f, 0.49983f,
  0.49984f, 0.49985f, 0.49985f, 0.49986f, 0.49986f,
  0.49987f, 0.49987f, 0.49988f, 0.49988f, 0.49989f,
  0.49989f, 0.49990f, 0.49990f, 0.49990f, 0.49991f,
  0.49991f, 0.49992f, 0.49992f, 0.49992f, 0.49992f,
  0.49993f, 0.49993f, 0.49993f, 0.49994f, 0.49994f,
  0.49994f, 0.49994f, 0.49995f, 0.49995f, 0.49995f,
  0.49995f, 0.49995f, 0.49996f, 0.49996f, 0.49996f,
  0.49996f, 0.49996f, 0.49996f, 0.49997f, 0.49997f,
  0.49997f, 0.49997f, 0.49997f, 0.49997f, 0.49997f,
  0.49997f, 0.49998f, 0.49998f, 0.49998f, 0.49998f 
};

//***************************************************************************
//
// * Return the value of the cumulative normal distrubution function at a given
//   point.
//==========================================================================
float SJCProbability::
NormalCumulativeDistribution(float x, float mean, float std_dev)
//==========================================================================
{
  // Convert x into standard normal
  float   y = ( x - mean ) / std_dev;
  
  // Take absolute value.
  float   py = fabs(y);
  
  // Check for out of range
  if ( py > 4.09 )
    return y < 0.0 ? 0.0 : 1.0;
  
  int	min = (int)floor(py * 100.0);
  int	max = (int)ceil(py * 100.0);
  
  if ( min == max )
    return y < 0.0 ? 0.5 - cumul_distrib_samples[min]
      : 0.5 + cumul_distrib_samples[min];
  
  float   fmin = min / 100.0;
  float   fmax = max / 100.0;
  
  float   r = ( py - fmin ) * 100.0 * cumul_distrib_samples[max]
    + ( fmax - py ) * 100.0 * cumul_distrib_samples[min];
  
  return y < 0.0 ? 0.5 - r : 0.5 + r;
}

//***************************************************************************
//
// * 
//==========================================================================
float SJCProbability::
UniformCumulativeDistribution(float x, float min, float max)
//==========================================================================
{
  // Convert x into uniform on 0-1
  return ( x - min ) / ( max - min );
}

//***************************************************************************
//
// * 
//==========================================================================
static int DoubleSort(const void *a, const void *b)
//==========================================================================
{
  float   x = *(float*)a;
  float   y = *(float*)b;
  
  if ( x < y )
    return -1;
  else if ( x > y )
    return 1;
  return 0;
}

//***************************************************************************
//
// * 
//==========================================================================
static float ProbKS(float alam)
//==========================================================================
{
  float   a2 = -2.0 * alam * alam;
  float   fac = 2.0;
  float   sum = 0.0;
  float   termbf = 0.0;
  for ( int j = 1 ; j < 100 ; j++ )   {
    float	term = fac * exp(a2 * j*j);
    sum += term;
    if ( fabs(term) <= 0.001 * termbf
	 || fabs(term) < 1.0e-8 * sum )
      return sum;
    fac = -fac;
    termbf = fabs(term);
  }
  return 1.0;
}

//***************************************************************************
//
// * 
//==========================================================================
static float ProbKP(float alam)
//==========================================================================
{
  if ( alam < 0.4 )
    return 1.0;
  
  float   a2 = alam * alam;
  float   sum = 0.0;
  float   termbf = 0.0;
  for ( int j = 1 ; j < 200 ; j++ )   {
    float term = 2.0 * ( 4.0 * j * j * a2 - 1 ) * exp(-2.0 * a2 * j*j);
    sum += term;
    if ( fabs(term) <= 0.001 * termbf
	 || fabs(term) < 1.0e-8 * sum )
      return sum;
    termbf = fabs(term);
  }
  return 1.0;
}

//***************************************************************************
//
// * Compute the probability that an observed distribution differs from
//   a particular normal distribution.
//==========================================================================
float SJCProbability::
ProbSameNormal(float *values, uint nvalues, float mean,
	       float std_dev, float &dist)
//==========================================================================
{
  qsort(values, nvalues, sizeof(float), DoubleSort);
  
  float   en = (float)nvalues;
  float   fo = 0.0;
  dist = 0.0f;
  for ( uint j = 0 ; j < nvalues ; j++ ) {
    float	fn = j / en;
    float	ff = NormalCumulativeDistribution(values[j], mean, std_dev);
    float	dt = SJCMax(fabs(fo - ff), fabs(fn - ff));
    if ( dt > dist ) 
      dist = dt;
    fo = fn;
  }
  en = sqrt(en);
  return ProbKS((en + 0.12 + 0.11/en)*dist);
}

//***************************************************************************
//
// * Compute the probability that an observed distribution differs from
//   a uniform distribution on a circle.
//==========================================================================
float SJCProbability::
ProbSameUniformCircular(float *values, uint nvalues, float &dist)
//==========================================================================
{
  qsort(values, nvalues, sizeof(float), DoubleSort);
  
  float   en = (float)nvalues;
  float   fo = 0.0;
  float   d1 = 0.0f;
  float   d2 = 0.0f;
  for ( uint j = 0 ; j < nvalues ; j++ )  {
    float	fn = j / en;
    float	ff = UniformCumulativeDistribution(values[j], -M_PI, M_PI);
    float	dt = SJCMax(fo - ff, fn - ff);
    if ( dt > d1 )
      d1 = dt;
    dt = SJCMax(ff - fo, ff - fn);
    if ( dt > d2 )
      d2 = dt;
    fo = fn;
  }
  en = sqrt(en);
  dist = d1 + d2;
  return ProbKP((en + 0.155 + 0.24/en)*dist);
}

//****************************************************************************
//
// * Output operator
//=============================================================================
std::ostream& operator<<(std::ostream &o, const SJCProbability &r)
//=============================================================================
{
  return o;
}
