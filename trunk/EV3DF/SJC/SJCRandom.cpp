/************************************************************************
     Main File:

     File:        SJCRandom.cpp

     Author:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
                  Steven Chenney, schenney@cs.wisc.edu
  
     Comment:     The random variable operation

     * Should combine this with rand48 

     Functions:   
     
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include "SJCRandom.h"
#include <math.h>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const long 	SJCRandomd::IA = 16807;
const long 	SJCRandomd::IM = 2147483647;
const double 	SJCRandomd::AMA = ( 1.0 / IM );
const long 	SJCRandomd::IQ = 127773;
const long 	SJCRandomd::IR = 2836;
const long 	SJCRandomd::NTAB = 32;
const long  	SJCRandomd::NDIVA = ( 1 + ( IM - 1 ) / NTAB );
const double 	SJCRandomd::EPS = 1.e-16;
const double  	SJCRandomd::RNMX = ( 1.0 - EPS );

const long 	SJCRandomf::IA = 16807;
const long 	SJCRandomf::IM = 2147483647;
const float 	SJCRandomf::AMA = ( 1.0 / IM );
const long 	SJCRandomf::IQ = 127773;
const long 	SJCRandomf::IR = 2836;
const long 	SJCRandomf::NTAB = 32;
const long  	SJCRandomf::NDIVA = ( 1 + ( IM - 1 ) / NTAB );
const float 	SJCRandomf::EPS = 1.e-16;
const float  	SJCRandomf::RNMX = ( 1.0 - EPS );

//****************************************************************************
//
// * Constructor
//=============================================================================
SJCRandomd::
SJCRandomd(const long seed)
//=============================================================================
{
  long    k;
  
  idum = seed;
  if ( idum < 1 )
    idum = 1;
  
  iy = 0;
  for ( int j = NTAB + 7 ; j >= 0 ; j-- )   {
    k = idum / IQ;
    idum = IA * ( idum - k * IQ ) - IR * k;
    if ( idum < 0 )
      idum += IM;
    if ( j < NTAB )
      iv[j] = idum;
  }
  iy = iv[0];
  
  iset = 0;
}


//****************************************************************************
//
// * Constructor
//=============================================================================
SJCRandomd::SJCRandomd(const long idum_in, const long iy_in,
		     const long iv_in[32], const int iset_in,
		     const double gset_in)
//=============================================================================
{
    idum = idum_in;
    iy = iy_in;
    for ( uint i = 0 ; i < 32 ; i++ )
	iv[i] = iv_in[i];
    iset = iset_in;
    gset = gset_in;
}


//****************************************************************************
//
// * Constructor
//=============================================================================
SJCRandomd::SJCRandomd(const SJCRandomd &s)
//=============================================================================
{
  idum = s.idum;
  iy = s.iy;
  for ( uint i = 0 ; i < 32 ; i++ )
    iv[i] = s.iv[i];
  iset = s.iset;
  gset = s.gset;
}

//****************************************************************************
//
// * Set up the seed
//=============================================================================
void
SJCRandomd::Seed(const long seed)
//=============================================================================
{
  long    k;

  idum = seed;
  if ( idum < 1 )
    idum = 1;

  iy = 0;
  for ( int j = NTAB + 7 ; j >= 0 ; j-- )  {
    k = idum / IQ;
    idum = IA * ( idum - k * IQ ) - IR * k;
    if ( idum < 0 )
      idum += IM;
    if ( j < NTAB )
      iv[j] = idum;
  }
  iy = iv[0];
  
  iset = 0;
}


//****************************************************************************
//
// * Assign operator
//=============================================================================
SJCRandomd&
SJCRandomd::operator=(const SJCRandomd &s)
//=============================================================================
{
  idum = s.idum;
  iy = s.iy;
  for ( uint i = 0 ; i < 32 ; i++ )
    iv[i] = s.iv[i];
  iset = s.iset;
  gset = s.gset;
  
  return (*this);
}


//****************************************************************************
//
// * Output operator
//=============================================================================
std::ostream&
operator<<(std::ostream &o, const SJCRandomd &r)
//=============================================================================
{
  o << "[ " << r.idum << " " << r.iy << " ";
  for ( uint i = 0 ; i < 32 ; i++ )
    o << r.iv[i] << " ";
  o << r.iset << " " << r.gset << " ]";
  
  return o;
}


//****************************************************************************
//
// * Return a random number between min and max with uniform distribution
//=============================================================================
double
SJCRandomd::Uniform(const double min, const double max)
//=============================================================================
{
  int	  j;
  long    k;
  double  temp;
  
  k = idum / IQ;
  idum = IA * ( idum - k * IQ ) - IR * k;
  if ( idum < 0 )
    idum += IM;
  j = iy / NDIVA;
  iy = iv[j];
  iv[j] = idum;
  temp = AMA * iy;
  if ( temp > RNMX )
    return RNMX;
  
  return temp * ( max - min ) + min;
}


//****************************************************************************
//
// * Return a double between min and max - 1 with uniform distribution
//=============================================================================
long
SJCRandomd::Uniform(const long min, const long max)
//=============================================================================
{
  int	    j;
  long    k;
  
  k = idum / IQ;
  idum = IA * ( idum - k * IQ ) - IR * k;
  if ( idum < 0 )
    idum += IM;
  j = iy / NDIVA;
  iy = iv[j];
  iv[j] = idum;
  
  // This assumes that the range is quite small with respect to the range
  // of longs.
  return ( iy % ( max - min ) ) + min;
}


//****************************************************************************
//
// * randomly pick up a direction in 2D circle
//=============================================================================
SJCVector2d
SJCRandomd::Circular(const double radius)
//=============================================================================
{
  double  angle = Uniform(0.0, 2.0 * M_PI);
  
  return SJCVector2d(cos(angle), sin(angle));
}


//****************************************************************************
//
// * Return a std::vector in 3D with radius 1
//=============================================================================
SJCVector3d
SJCRandomd::Spherical(const double radius)
//=============================================================================
{
  double  phi = Uniform(0.0, 2.0 * M_PI);
  double  sin_theta = Uniform(-1.0,1.0);
  double  theta = asin(sin_theta);
  double  cos_theta = cos(theta);
  
  return SJCVector3d(cos(phi) * cos_theta, sin(phi) * cos_theta, sin_theta);
}


//****************************************************************************
//
// * Return a number which is normally distribution
//=============================================================================
double
SJCRandomd::Normal(const double mean, const double std_dev)
//=============================================================================
{
  double  fac, rsq, v1, v2;
  
  if ( ! iset )  {
    do {
      v1 = Uniform(-1.0, 1.0);
      v2 = Uniform(-1.0, 1.0);
      rsq = v1 * v1 + v2 * v2;
    } while ( rsq >= 1.0 || rsq == 0.0 );
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return ( v2 * fac * std_dev + mean );
  }
  iset = 0;
  return ( gset * std_dev + mean );
}


//***************************************************************************
//
// * Randomly permute the sequence from 0 to n-1 in the array
//==========================================================================
void SJCRandomd::
RandomPermutation(std::vector<uint>& permute)
//==========================================================================
{
  uint size = permute.size();
  for ( uint i = 0 ; i < size ; i++ )
    permute[i] = i;
  for ( uint i = 0 ; i < size - 1 ; i++ )   {
    uint  r = (uint)floor(Uniform(0.f, 1.f) * ( size - i )) + i;
    uint  tmp = permute[r];
    permute[r] = permute[i];
    permute[i] = tmp;
  } // end of for i
}


//***************************************************************************
//
// * Randomly permute the sequence from 0 to n-1 in the array
//==========================================================================
void SJCRandomd::
RandomSelect(std::vector<uint>& select, uint size, uint num)
//==========================================================================
{
  assert(size >= num);
  select.clear();

  for( ; ; ) {
    bool flag = true;
    long temp = Uniform((long)0, (long)size);
    for(uint i = 0; i < select.size(); i++) {
      if(select[i] == (uint)temp){
	flag = false;
	break;
      }// end of if
    }// end of for
    if(flag) { // not exist
      select.push_back((uint)temp);
    }
    if(select.size() >= num) // reach K
      return;
  }// end of for 
}

//***************************************************************************
//
// * Randomly select a number between 1 to size - 1 according to the 
//   probability
//==========================================================================
uint SJCRandomd::
RandomProbability(std::vector<double>& probs, uint start)
//==========================================================================
{
  std::vector<double> probs_normal(probs.size());
  
  double accum = 0.f;
  for(uint i = start; i < probs.size(); i++){
    accum += probs[i];
    probs_normal[i] = accum;
  }

  // Protect the probs contain nothing
  if(accum <= SJC_EPSILON)
    return start;
  
  // Normalize the values
  for(uint i = start; i < probs.size(); i++){
    probs_normal[i] = probs_normal[i] / accum;
  }
  // Generate the probability
  double probe = Uniform(0.f, 1.f);


  if(probe <= probs_normal[start])
    return start;

  for(uint i = start; i < probs.size() - 1; i++) {
    if(probe > probs_normal[i] && probe <= probs_normal[i + 1]) {
      return i + 1;
    }
  } // end of for

  return probs.size() - 1 ;
 
}
//***************************************************************************
//
// * Randomly select a number between 1 to size - 1 according to the 
//   probability and then reject the element with probability lowwer than
//==========================================================================
uint SJCRandomd::
RandomProbability(std::vector<double>& probs, uint start, double threshold)
//==========================================================================
{
  std::vector<double> probs_normal;
  std::vector<uint>  index;
  
  double accum = 0.f;
  for(uint i = start; i < probs.size(); i++){
    if(probs[i] >= threshold) {
      accum += probs[i];
      probs_normal.push_back(accum);
      index.push_back(i);
    } // end of if
  }// end of for
  
  // Protect the probs contain nothing
  if(accum <= SJC_EPSILON)
    return start;
  

  // Normalize the values 
  for(uint i = 0; i < probs_normal.size(); i++){
    probs_normal[i] = probs_normal[i] / accum;
  }

  // Generate the probability
  double probe = Uniform(0.f, 1.f);

  if(probe <= probs_normal[0])
    return index[0];

  for(uint i = 0; i < probs_normal.size() - 1; i++) {
    if(probe > probs_normal[i] && probe <= probs_normal[i + 1]) {
      return index[i + 1];
    }
  } // end of for

  return index[probs_normal.size() - 1] ;
 
}

//***************************************************************************
//
// * Randomly select a number between 1 to size - 1 according to the 
//   probability
//==========================================================================
uint SJCRandomd::
RandomProbability(const double* probs, uint prob_size, uint start)
//==========================================================================
{
  std::vector<double> probs_normal(prob_size);
  
  double accum = 0.f;
  for(uint i = start; i < prob_size; i++){
    accum += probs[i];
    probs_normal[i] = accum;
  }

  // Protect the probs contain nothing
  if(accum <= SJC_EPSILON)
    return start;
  
  // Normalize the values
  for(uint i = start; i < prob_size; i++){
    probs_normal[i] = probs_normal[i] / accum;
  }
  // Generate the probability
  double probe = Uniform(0.f, 1.f);


  if(probe <= probs_normal[start])
    return start;

  for(uint i = start; i < prob_size - 1; i++) {
    if(probe > probs_normal[i] && probe <= probs_normal[i + 1]) {
      return i + 1;
    }
  } // end of for

  return prob_size - 1 ;
 
}

//***************************************************************************
//
// * Permute the value according to the probability
//==========================================================================
void SJCRandomd::
ProbabilityPermutation(std::vector<uint>&  permute,
		       std::vector<double>& probs)
//==========================================================================
{
  
  uint size = probs.size();
  permute.resize(size);
  for ( uint i = 0 ; i < size ; i++ )
    permute[i] = i;

  // to swap to generate the permutation
  for ( uint i = 0 ; i < size - 1 ; i++ ) {
    uint  index    = RandomProbability(probs, i);
    uint  tmp      = permute[index];

    // Swap the index and the probability
    permute[index] = permute[i];
    permute[i]     = tmp;

    double tmp_f    = probs[index];
    probs[index]   = probs[i];
    probs[i]       = tmp_f;
  } // end of for i
}

//***************************************************************************
//
// * Permute the value according to the probability and it also kick out
//   those who is below threshold
//==========================================================================
void SJCRandomd::
ProbabilityPermutation(std::vector<uint>&  permute,
		       std::vector<double>& probs, 
		       double threshold)
//==========================================================================
{
  std::vector<double> probs_normal;
  std::vector<uint>  index;
  
  // Keep those up the threshold
  for(uint i = 0; i < probs.size(); i++){
    if(probs[i] >= threshold) {

      // push in the probability and index
      probs_normal.push_back(probs[i]);
      index.push_back(i);
    } // end of if
  }// end of for

  // Permute according to the prob_normals
  std::vector<uint> permute_index;
  ProbabilityPermutation(permute_index, probs_normal);

  // Resize the permute
  permute.resize(index.size());
  for(uint i = 0; i < permute_index.size(); i++){
    permute[i] = index[permute_index[i]];
  }
}




//***************************************************************************
// * Samples the hemisphere according to a uniform distribution 
//   i.e. proportional to hemisphere area, if will generate random variable 
//   itself.
//
//============================================================================
SJCVector3d SJCRandomd::
HemisphereNormalTheta(SJCVector3d& direction, double mean, double std_dev)
//============================================================================
{
  double phi, theta;	        // value for phi
  double cos_phi, sin_phi;	// cos and sin phi value
  double cos_theta, sin_theta;	// cos and sin theta value

  SJCVector3d zAxis(0., 0., 1.);
  SJCQuaterniond r(zAxis, direction);
 
 
  // Get the related value of phi
  phi = 2.0 * M_PI *  Uniform((double)0., (double)1.);
  cos_phi = cos(phi);
  sin_phi = sin(phi);
	
  // Get the related value of theta from SIGGRAPH Course 2001 #29 p.p. 25
  theta     = Normal(mean, std_dev);
  cos_theta = cos(theta);
  sin_theta = sin(theta);

  // Create the normal in Z coordinate
  SJCVector3d outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);

  // Rotate to align with the out direction
  outDir = r * outDir;

  // Normalize the direction  
  outDir.normalize();

  return outDir;
 
}


//****************************************************************************
//
// * A simple way to randomly (uniform) distribute points on sphere is called 
//   the "hypercube rejection method". To apply this to a unit cube at the
//   origin, choose coordinates (x, y, z) each uniformly distributed on the 
//   interval [-1,1]. If the length of this std::vector is greater than 1 then 
//   reject it, otherwise normalise it  and use it as a sample.
//   source: http://astronomy.swin.edu.au/~pbourke/geometry/spherepoints/
//
//============================================================================
SJCVector3d SJCRandomd::
SphereUniformHypercube(double& pdf_value)
//============================================================================
{

  double randVal1, randVal2, randVal3;
  double sum;
  
  do{
    randVal1 = Uniform((double)-1.0,(double) 1.0);
    randVal2 = Uniform((double)-1.0,(double) 1.0);
    randVal3 = Uniform((double)-1.0,(double) 1.0);
    
    sum = randVal1 * randVal1 + randVal2 * randVal2 + randVal3 * randVal3;
 } while(sum > 1.0);
  
  // Create an direction
  SJCVector3d outDir(randVal1, randVal2, randVal3);

  // normailize
  outDir.normalize();
  
  // Return the pdf 
  pdf_value = 1.0/6.0/M_PI;
  
  return outDir;
}

//***************************************************************************
//
// * Samples the sphere according to a uniform distribution 
//   i.e. proportional to sphere area, if will generate random variable itself.
//=============================================================================
SJCVector3d SJCRandomd::
SphereUniform(SJCVector3d& dump1, SJCVector3d& dump2, double& pdf_value)
//=============================================================================
{
  double uniformRandom1 = Uniform((double)0.0, (double)1.0);
  double uniformRandom2 = Uniform((double)0.0, (double)1.0);
  
  double phi, theta;	        // value for phi
  double cos_phi, sin_phi;	// cos and sin phi value
  double cos_theta, sin_theta;	// cos and sin theta value
  
  // Get the related value of phi
  phi = 2.0 * M_PI * uniformRandom1;
  cos_phi = cos(phi);
  sin_phi = sin(phi);
  
  // Get the related value of theta
  theta = M_PI * uniformRandom2; //theta is from -PI/2 to PI/2

  // z component is from +1 to -1 
  cos_theta = cos(theta);
  sin_theta = sin(theta);
  
  // Create direction
  SJCVector3d outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta); 
  
  // normailize
  outDir.normalize();

  // Return the pdf 
  pdf_value = 0.25 / M_PI;
  
  return outDir;
}

//***************************************************************************
//
// * Samples the hemisphere according to a uniform distribution 
//   i.e. proportional to hemisphere area, if will generate random variable 
//   itself.
//============================================================================
SJCVector3d SJCRandomd::
HemisphereUniform(SJCVector3d& leavingsurfInDir, SJCVector3d& normal,
		  double& pdf_value)
//============================================================================
{
  double uniformRandom1 = Uniform((double)0.0, (double)1.0);
  double uniformRandom2 = Uniform((double)0.0, (double)1.0);

  double phi, theta;	     // value for phi
  double cos_phi, sin_phi;			// cos and sin phi value
	double cos_theta, sin_theta;	// cos and sin theta value
  
	// Get the related value of phi
  phi = 2.0 * M_PI * uniformRandom1;
  cos_phi = cos(phi);
  sin_phi = sin(phi);
	
  // Get the related value of theta from SIGGRAPH Course 2001 #29 p.p. 25
  theta = 0.5 * M_PI * uniformRandom2;
  cos_theta = cos(theta);
  sin_theta = sin(theta);

  // Rotation to align with the current normal direction
  SJCVector3d X = leavingsurfInDir % normal;
  SJCVector3d Y = normal % X;

  // This matrix is from world to X, y, normal
  SJCRotateMatrixd rotation(X, Y, normal);

  // From x, y, normal to world
  SJCRotateMatrixd toWorld = rotation.inverse();
  
  // Create the normal in Z coordinate
  SJCVector3d outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);

  // Rotate to align with the out direction
  outDir = toWorld * outDir;
  outDir.normalize();

  // Return the pdf 
  pdf_value = 1.0 / (2.0 * M_PI);
  return outDir;
  
}


//****************************************************************************
// * Samples the hemisphere according to a cos_theta distribution 
//   We need to align the Z-axis with the normal direction which is cosTheta
//   about If this sampling is used as a pdf over the hemisphere, the 
//   corresponding pdf evaluation is:  cos_theta / pi  this function passed 
//   the test -- 12/31/03 
//   The input leavingsurfInDir is the input ray expressed using the direction 
//   of leaving the surface
//============================================================================
SJCVector3d SJCRandomd::
HemisphereCosTheta(SJCVector3d& leavingsurfInDir, SJCVector3d& normal, 
		   double& pdf_value)
//=============================================================================
{
  double uniformRandom1 = Uniform((double)0.0, (double)1.0);
  double uniformRandom2 = Uniform((double)0.0, (double)1.0);

  double phi;			// value for phi
  double cos_phi, sin_phi;	// cos and sin phi value
  double cos_theta, sin_theta;	// cos and sin theta value

  // Get the related value of phi
  phi = 2.0 * M_PI * uniformRandom1;
  cos_phi = cos(phi);
  sin_phi = sin(phi);
	
  // Get the related value of theta
  cos_theta = sqrt(1.0- uniformRandom2);
  sin_theta = sqrt(uniformRandom2);
  
  
  // Rotation to align with the normal direction
  SJCVector3d X = leavingsurfInDir % normal;
  SJCVector3d Y = normal % X;
  // This matrix is from world to X, y, normal
  SJCRotateMatrixd rotation(X, Y, normal);

  // From x, y, normal to world
  SJCRotateMatrixd toWorld = rotation.inverse();
  

  // Create the normal in Z coordinate
  SJCVector3d outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta); 

  // Rotate to align with the out direction
  outDir = toWorld * outDir;
  outDir.normalize();
  
  if((outDir * normal)<0){
	
    printf("how can diffuse sampling get this out direction?\n");	
    printf("rand1 %f, rand2 %f\n", uniformRandom1, uniformRandom2);
    printf("normal:   ");      std::cout << normal << std::endl;
    printf("inDir:   ");       std::cout << leavingsurfInDir << std::endl;
    printf("leaving_out:   "); std::cout << outDir << std::endl;
    printf("cosOutNormal %f, cosInNormal  %f\n", outDir * normal, 
	   leavingsurfInDir * normal );
  }
  
  // pdf value
  pdf_value =  cos_theta / M_PI;
  return outDir;
} 


//*****************************************************************************
// * Samples the hemisphere according to a cos_theta ^ n  distribution 
//	 Phong Specular situation
// * align the Z-axis with the idealOut
//
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// It seems still something wrong with this function == 11/12/03
// refer outDir = sampleHemisphereCosNTheta(normal, idealDir, specularPower,
//  uniformRandom1, uniformRandom2, specularProbability);
// in the file material.cpp 
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//===========================================================================
SJCVector3d SJCRandomd::
HemisphereCosNTheta(SJCVector3d& normal, SJCVector3d& idealOut, 
		    double nOrder, double& pdf_value)
//=============================================================================
{
  double uniformRandom1 = Uniform((double)0.0, (double)1.0);
  double uniformRandom2 = Uniform((double)0.0, (double)1.0);


  double phi;	                // value for phi
  double cos_phi, sin_phi;	// cos and sin phi value
  double cos_theta, sin_theta;	// cos and sin theta value

  // Get the related value of phi
  phi = 2.0 * M_PI * uniformRandom1;
  cos_phi = cos(phi);
  sin_phi = sin(phi);
	
  // Get the related value of theta
  cos_theta = pow((double)1 - uniformRandom2, (double)1.0 / (nOrder + 1));
  sin_theta = sqrt((double)1.0 - cos_theta * cos_theta);
  

  // Rotation to align with the current ideal specular direction
  SJCVector3d X = normal % idealOut;
  SJCVector3d Y = idealOut % X;

  // This matrix is from world to X, y, normal
  SJCRotateMatrixd rotation(X, Y, idealOut);

  // From x, y, normal to world
  SJCRotateMatrixd toWorld = rotation.inverse();
  
  SJCVector3d outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
  outDir = toWorld * outDir;
  outDir.normalize();
  
  pdf_value = (nOrder + 1.0) * pow (cos_theta, nOrder) / (2.0 * M_PI);
  
  return outDir;

}

//***************************************************************************
//
// * Makes a nice grid for stratified sampling 
//============================================================================
void SJCRandomd::
GetNrDivisions(int samples, int &divs1, int &divs2)
//=============================================================================
{
  // The valid sample should over 1
  if (samples <= 0) {
    divs1 = 0;
    divs2 = 0;
    return;
  }
  
  // Start will the square root value of samples and decrement of divs1
  // To find the matching point
  divs1 = (int)ceil(sqrt((double)samples));
  divs2 = samples / divs1;
  while (divs1 * divs2 != samples && divs1 > 1) {
    divs1 --;
    divs2 = samples / divs1;
  } // end of while
}


//****************************************************************************
//
// * Constructor
//=============================================================================
SJCRandomf::
SJCRandomf(const long seed)
//=============================================================================
{
  long    k;
  
  idum = seed;
  if ( idum < 1 )
    idum = 1;
  
  iy = 0;
  for ( int j = NTAB + 7 ; j >= 0 ; j-- )   {
    k = idum / IQ;
    idum = IA * ( idum - k * IQ ) - IR * k;
    if ( idum < 0 )
      idum += IM;
    if ( j < NTAB )
      iv[j] = idum;
  }
  iy = iv[0];
  
  iset = 0;
}


//****************************************************************************
//
// * Constructor
//=============================================================================
SJCRandomf::SJCRandomf(const long idum_in, const long iy_in,
		     const long iv_in[32], const int iset_in,
		     const float gset_in)
//=============================================================================
{
    idum = idum_in;
    iy = iy_in;
    for ( uint i = 0 ; i < 32 ; i++ )
	iv[i] = iv_in[i];
    iset = iset_in;
    gset = gset_in;
}


//****************************************************************************
//
// * Constructor
//=============================================================================
SJCRandomf::SJCRandomf(const SJCRandomf &s)
//=============================================================================
{
  idum = s.idum;
  iy = s.iy;
  for ( uint i = 0 ; i < 32 ; i++ )
    iv[i] = s.iv[i];
  iset = s.iset;
  gset = s.gset;
}

//****************************************************************************
//
// * Set up the seed
//=============================================================================
void
SJCRandomf::Seed(const long seed)
//=============================================================================
{
  long    k;

  idum = seed;
  if ( idum < 1 )
    idum = 1;

  iy = 0;
  for ( int j = NTAB + 7 ; j >= 0 ; j-- )  {
    k = idum / IQ;
    idum = IA * ( idum - k * IQ ) - IR * k;
    if ( idum < 0 )
      idum += IM;
    if ( j < NTAB )
      iv[j] = idum;
  }
  iy = iv[0];
  
  iset = 0;
}


//****************************************************************************
//
// * Assign operator
//=============================================================================
SJCRandomf&
SJCRandomf::operator=(const SJCRandomf &s)
//=============================================================================
{
  idum = s.idum;
  iy = s.iy;
  for ( uint i = 0 ; i < 32 ; i++ )
    iv[i] = s.iv[i];
  iset = s.iset;
  gset = s.gset;
  
  return (*this);
}


//****************************************************************************
//
// * Output operator
//=============================================================================
std::ostream&
operator<<(std::ostream &o, const SJCRandomf &r)
//=============================================================================
{
  o << "[ " << r.idum << " " << r.iy << " ";
  for ( uint i = 0 ; i < 32 ; i++ )
    o << r.iv[i] << " ";
  o << r.iset << " " << r.gset << " ]";
  
  return o;
}


//****************************************************************************
//
// * Return a random number between min and max with uniform distribution
//=============================================================================
float
SJCRandomf::Uniform(const float min, const float max)
//=============================================================================
{
  int	  j;
  long    k;
  float  temp;
  
  k = idum / IQ;
  idum = IA * ( idum - k * IQ ) - IR * k;
  if ( idum < 0 )
    idum += IM;
  j = iy / NDIVA;
  iy = iv[j];
  iv[j] = idum;
  temp = AMA * iy;
  if ( temp > RNMX )
    return RNMX;
  
  return temp * ( max - min ) + min;
}


//****************************************************************************
//
// * Return a float between min and max - 1 with uniform distribution
//=============================================================================
long
SJCRandomf::Uniform(const long min, const long max)
//=============================================================================
{
  int	    j;
  long    k;
  
  k = idum / IQ;
  idum = IA * ( idum - k * IQ ) - IR * k;
  if ( idum < 0 )
    idum += IM;
  j = iy / NDIVA;
  iy = iv[j];
  iv[j] = idum;
  
  // This assumes that the range is quite small with respect to the range
  // of longs.
  return ( iy % ( max - min ) ) + min;
}


//****************************************************************************
//
// * randomly pick up a direction in 2D circle
//=============================================================================
SJCVector2f
SJCRandomf::Circular(const float radius)
//=============================================================================
{
  float  angle = Uniform((float)0.0, (float)2.0 * M_PI);
  
  return SJCVector2f((float)cos(angle), (float)sin(angle));
}


//****************************************************************************
//
// * Return a std::vector in 3D with radius 1
//=============================================================================
SJCVector3f
SJCRandomf::Spherical(const float radius)
//=============================================================================
{
  float  phi       = Uniform((float)0.0, (float)2.0 * M_PI);
  float  sin_theta = Uniform((float)-1.0, (float)1.0);
  float  theta     = (float)asin(sin_theta);
  float  cos_theta = (float)cos(theta);
  
  return SJCVector3f((float)cos(phi) * cos_theta, 
		     (float)sin(phi) * cos_theta, sin_theta);
}


//****************************************************************************
//
// * Return a number which is normally distribution
//=============================================================================
float
SJCRandomf::Normal(const float mean, const float std_dev)
//=============================================================================
{
  float  fac, rsq, v1, v2;
  
  if ( ! iset )  {
    do {
      v1 = Uniform((float)-1.0, (float)1.0);
      v2 = Uniform((float)-1.0, (float)1.0);
      rsq = v1 * v1 + v2 * v2;
    } while ( rsq >= 1.0 || rsq == 0.0 );
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return ( v2 * fac * std_dev + mean );
  }
  iset = 0;
  return ( gset * std_dev + mean );
}


//***************************************************************************
//
// * Randomly permute the sequence from 0 to n-1 in the array
//==========================================================================
void SJCRandomf::
RandomPermutation(std::vector<uint>& permute)
//==========================================================================
{
  uint size = permute.size();
  for ( uint i = 0 ; i < size ; i++ )
    permute[i] = i;
  for ( uint i = 0 ; i < size - 1 ; i++ )   {
    uint  r = (uint)floor(Uniform(0.f, 1.f) * ( size - i )) + i;
    uint  tmp = permute[r];
    permute[r] = permute[i];
    permute[i] = tmp;
  } // end of for i
}


//***************************************************************************
//
// * Randomly permute the sequence from 0 to n-1 in the array
//==========================================================================
void SJCRandomf::
RandomSelect(std::vector<uint>& select, uint size, uint num)
//==========================================================================
{
  assert(size >= num);
  select.clear();

  for( ; ; ) {
    bool flag = true;
    long temp = Uniform((long)0, (long)size);
    for(uint i = 0; i < select.size(); i++) {
      if(select[i] == (uint)temp){
	flag = false;
	break;
      }// end of if
    }// end of for
    if(flag) { // not exist
      select.push_back((uint)temp);
    }
    if(select.size() >= num) // reach K
      return;
  }// end of for 
}

//***************************************************************************
//
// * Randomly select a number between 1 to size - 1 according to the 
//   probability
//==========================================================================
uint SJCRandomf::
RandomProbability(std::vector<float>& probs, uint start)
//==========================================================================
{
  std::vector<float> probs_normal(probs.size());
  
  float accum = 0.f;
  for(uint i = start; i < probs.size(); i++){
    accum += probs[i];
    probs_normal[i] = accum;
  }

  // Protect the probs contain nothing
  if(accum <= SJC_EPSILON)
    return start;
  
  // Normalize the values
  for(uint i = start; i < probs.size(); i++){
    probs_normal[i] = probs_normal[i] / accum;
  }
  // Generate the probability
  float probe = Uniform(0.f, 1.f);


  if(probe <= probs_normal[start])
    return start;

  for(uint i = start; i < probs.size() - 1; i++) {
    if(probe > probs_normal[i] && probe <= probs_normal[i + 1]) {
      return i + 1;
    }
  } // end of for

  return probs.size() - 1 ;
 
}
//***************************************************************************
//
// * Randomly select a number between 1 to size - 1 according to the 
//   probability and then reject the element with probability lowwer than
//==========================================================================
uint SJCRandomf::
RandomProbability(std::vector<float>& probs, uint start, float threshold)
//==========================================================================
{
  std::vector<float> probs_normal;
  std::vector<uint>  index;
  
  float accum = 0.f;
  for(uint i = start; i < probs.size(); i++){
    if(probs[i] >= threshold) {
      accum += probs[i];
      probs_normal.push_back(accum);
      index.push_back(i);
    } // end of if
  }// end of for
  
  // Protect the probs contain nothing
  if(accum <= SJC_EPSILON)
    return start;
  

  // Normalize the values 
  for(uint i = 0; i < probs_normal.size(); i++){
    probs_normal[i] = probs_normal[i] / accum;
  }

  // Generate the probability
  float probe = Uniform(0.f, 1.f);

  if(probe <= probs_normal[0])
    return index[0];

  for(uint i = 0; i < probs_normal.size() - 1; i++) {
    if(probe > probs_normal[i] && probe <= probs_normal[i + 1]) {
      return index[i + 1];
    }
  } // end of for

  return index[probs_normal.size() - 1] ;
 
}

//***************************************************************************
//
// * Randomly select a number between 1 to size - 1 according to the 
//   probability
//==========================================================================
uint SJCRandomf::
RandomProbability(const float* probs, uint prob_size, uint start)
//==========================================================================
{
  std::vector<float> probs_normal(prob_size);
  
  float accum = 0.f;
  for(uint i = start; i < prob_size; i++){
    accum += probs[i];
    probs_normal[i] = accum;
  }

  // Protect the probs contain nothing
  if(accum <= SJC_EPSILON)
    return start;
  
  // Normalize the values
  for(uint i = start; i < prob_size; i++){
    probs_normal[i] = probs_normal[i] / accum;
  }
  // Generate the probability
  float probe = Uniform(0.f, 1.f);


  if(probe <= probs_normal[start])
    return start;

  for(uint i = start; i < prob_size - 1; i++) {
    if(probe > probs_normal[i] && probe <= probs_normal[i + 1]) {
      return i + 1;
    }
  } // end of for

  return prob_size - 1 ;
 
}

//***************************************************************************
//
// * Permute the value according to the probability
//==========================================================================
void SJCRandomf::
ProbabilityPermutation(std::vector<uint>&  permute,
		       std::vector<float>& probs)
//==========================================================================
{
  
  uint size = probs.size();
  permute.resize(size);
  for ( uint i = 0 ; i < size ; i++ )
    permute[i] = i;

  // to swap to generate the permutation
  for ( uint i = 0 ; i < size - 1 ; i++ ) {
    uint  index    = RandomProbability(probs, i);
    uint  tmp      = permute[index];

    // Swap the index and the probability
    permute[index] = permute[i];
    permute[i]     = tmp;

    float tmp_f    = probs[index];
    probs[index]   = probs[i];
    probs[i]       = tmp_f;
  } // end of for i
}

//***************************************************************************
//
// * Permute the value according to the probability and it also kick out
//   those who is below threshold
//==========================================================================
void SJCRandomf::
ProbabilityPermutation(std::vector<uint>&  permute,
		       std::vector<float>& probs, 
		       float threshold)
//==========================================================================
{
  std::vector<float> probs_normal;
  std::vector<uint>  index;
  
  // Keep those up the threshold
  for(uint i = 0; i < probs.size(); i++){
    if(probs[i] >= threshold) {

      // push in the probability and index
      probs_normal.push_back(probs[i]);
      index.push_back(i);
    } // end of if
  }// end of for

  // Permute according to the prob_normals
  std::vector<uint> permute_index;
  ProbabilityPermutation(permute_index, probs_normal);

  // Resize the permute
  permute.resize(index.size());
  for(uint i = 0; i < permute_index.size(); i++){
    permute[i] = index[permute_index[i]];
  }
}




//***************************************************************************
// * Samples the hemisphere according to a uniform distribution 
//   i.e. proportional to hemisphere area, if will generate random variable 
//   itself.
//
//============================================================================
SJCVector3f SJCRandomf::
HemisphereNormalTheta(SJCVector3f& direction, float mean, float std_dev)
//============================================================================
{
  float phi, theta;	        // value for phi
  float cos_phi, sin_phi;	// cos and sin phi value
  float cos_theta, sin_theta;	// cos and sin theta value

  SJCVector3f zAxis(0., 0., 1.);
  SJCQuaternionf r(zAxis, direction);
 
 
  // Get the related value of phi
  phi = 2.0 * M_PI *  Uniform((float)0., (float)1.);
  cos_phi = cos(phi);
  sin_phi = sin(phi);
	
  // Get the related value of theta from SIGGRAPH Course 2001 #29 p.p. 25
  theta     = Normal(mean, std_dev);
  cos_theta = cos(theta);
  sin_theta = sin(theta);

  // Create the normal in Z coordinate
  SJCVector3f outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);

  // Rotate to align with the out direction
  outDir = r * outDir;

  // Normalize the direction  
  outDir.normalize();

  return outDir;
 
}


//****************************************************************************
//
// * A simple way to randomly (uniform) distribute points on sphere is called 
//   the "hypercube rejection method". To apply this to a unit cube at the
//   origin, choose coordinates (x, y, z) each uniformly distributed on the 
//   interval [-1,1]. If the length of this std::vector is greater than 1 then 
//   reject it, otherwise normalise it  and use it as a sample.
//   source: http://astronomy.swin.edu.au/~pbourke/geometry/spherepoints/
//
//============================================================================
SJCVector3f SJCRandomf::
SphereUniformHypercube(float& pdf_value)
//============================================================================
{

  float randVal1, randVal2, randVal3;
  float sum;
  
  do{
    randVal1 = Uniform((float)-1.0,(float) 1.0);
    randVal2 = Uniform((float)-1.0,(float) 1.0);
    randVal3 = Uniform((float)-1.0,(float) 1.0);
    
    sum = randVal1 * randVal1 + randVal2 * randVal2 + randVal3 * randVal3;
 } while(sum > 1.0);
  
  // Create an direction
  SJCVector3f outDir(randVal1, randVal2, randVal3);

  // normailize
  outDir.normalize();
  
  // Return the pdf 
  pdf_value = 1.0/6.0/M_PI;
  
  return outDir;
}

//***************************************************************************
//
// * Samples the sphere according to a uniform distribution 
//   i.e. proportional to sphere area, if will generate random variable itself.
//=============================================================================
SJCVector3f SJCRandomf::
SphereUniform(SJCVector3f& dump1, SJCVector3f& dump2, float& pdf_value)
//=============================================================================
{
  float uniformRandom1 = Uniform((float)0.0, (float)1.0);
  float uniformRandom2 = Uniform((float)0.0, (float)1.0);
  
  float phi, theta;	        // value for phi
  float cos_phi, sin_phi;	// cos and sin phi value
  float cos_theta, sin_theta;	// cos and sin theta value
  
  // Get the related value of phi
  phi = 2.0 * M_PI * uniformRandom1;
  cos_phi = cos(phi);
  sin_phi = sin(phi);
  
  // Get the related value of theta
  theta = M_PI * uniformRandom2; //theta is from -PI/2 to PI/2

  // z component is from +1 to -1 
  cos_theta = cos(theta);
  sin_theta = sin(theta);
  
  // Create direction
  SJCVector3f outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta); 
  
  // normailize
  outDir.normalize();

  // Return the pdf 
  pdf_value = 0.25 / M_PI;
  
  return outDir;
}

//***************************************************************************
//
// * Samples the hemisphere according to a uniform distribution 
//   i.e. proportional to hemisphere area, if will generate random variable 
//   itself.
//============================================================================
SJCVector3f SJCRandomf::
HemisphereUniform(SJCVector3f& leavingsurfInDir, SJCVector3f& normal,
		  float& pdf_value)
//============================================================================
{
  float uniformRandom1 = Uniform((float)0.0, (float)1.0);
  float uniformRandom2 = Uniform((float)0.0, (float)1.0);

  float phi, theta;	     // value for phi
  float cos_phi, sin_phi;			// cos and sin phi value
	float cos_theta, sin_theta;	// cos and sin theta value
  
	// Get the related value of phi
  phi = 2.0 * M_PI * uniformRandom1;
  cos_phi = cos(phi);
  sin_phi = sin(phi);
	
  // Get the related value of theta from SIGGRAPH Course 2001 #29 p.p. 25
  theta = 0.5 * M_PI * uniformRandom2;
  cos_theta = cos(theta);
  sin_theta = sin(theta);

  // Rotation to align with the current normal direction
  SJCVector3f X = leavingsurfInDir % normal;
  SJCVector3f Y = normal % X;

  // This matrix is from world to X, y, normal
  SJCRotateMatrixf rotation(X, Y, normal);

  // From x, y, normal to world
  SJCRotateMatrixf toWorld = rotation.inverse();
  
  // Create the normal in Z coordinate
  SJCVector3f outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);

  // Rotate to align with the out direction
  outDir = toWorld * outDir;
  outDir.normalize();

  // Return the pdf 
  pdf_value = 1.0 / (2.0 * M_PI);
  return outDir;
  
}


//****************************************************************************
// * Samples the hemisphere according to a cos_theta distribution 
//   We need to align the Z-axis with the normal direction which is cosTheta
//   about If this sampling is used as a pdf over the hemisphere, the 
//   corresponding pdf evaluation is:  cos_theta / pi  this function passed 
//   the test -- 12/31/03 
//   The input leavingsurfInDir is the input ray expressed using the direction 
//   of leaving the surface
//============================================================================
SJCVector3f SJCRandomf::
HemisphereCosTheta(SJCVector3f& leavingsurfInDir, SJCVector3f& normal, 
		   float& pdf_value)
//=============================================================================
{
  float uniformRandom1 = Uniform((float)0.0, (float)1.0);
  float uniformRandom2 = Uniform((float)0.0, (float)1.0);

  float phi;			// value for phi
  float cos_phi, sin_phi;	// cos and sin phi value
  float cos_theta, sin_theta;	// cos and sin theta value

  // Get the related value of phi
  phi = 2.0 * M_PI * uniformRandom1;
  cos_phi = cos(phi);
  sin_phi = sin(phi);
	
  // Get the related value of theta
  cos_theta = sqrt(1.0- uniformRandom2);
  sin_theta = sqrt(uniformRandom2);
  
  
  // Rotation to align with the normal direction
  SJCVector3f X = leavingsurfInDir % normal;
  SJCVector3f Y = normal % X;
  // This matrix is from world to X, y, normal
  SJCRotateMatrixf rotation(X, Y, normal);

  // From x, y, normal to world
  SJCRotateMatrixf toWorld = rotation.inverse();
  

  // Create the normal in Z coordinate
  SJCVector3f outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta); 

  // Rotate to align with the out direction
  outDir = toWorld * outDir;
  outDir.normalize();
  
  if((outDir * normal)<0){
	
    printf("how can diffuse sampling get this out direction?\n");	
    printf("rand1 %f, rand2 %f\n", uniformRandom1, uniformRandom2);
    printf("normal:   ");      std::cout << normal << std::endl;
    printf("inDir:   ");       std::cout << leavingsurfInDir << std::endl;
    printf("leaving_out:   "); std::cout << outDir << std::endl;
    printf("cosOutNormal %f, cosInNormal  %f\n", outDir * normal, 
	   leavingsurfInDir * normal );
  }
  
  // pdf value
  pdf_value =  cos_theta / M_PI;
  return outDir;
} 


//*****************************************************************************
// * Samples the hemisphere according to a cos_theta ^ n  distribution 
//	 Phong Specular situation
// * align the Z-axis with the idealOut
//
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// It seems still something wrong with this function == 11/12/03
// refer outDir = sampleHemisphereCosNTheta(normal, idealDir, specularPower,
//  uniformRandom1, uniformRandom2, specularProbability);
// in the file material.cpp 
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//===========================================================================
SJCVector3f SJCRandomf::
HemisphereCosNTheta(SJCVector3f& normal, SJCVector3f& idealOut, 
		    float nOrder, float& pdf_value)
//=============================================================================
{
  float uniformRandom1 = Uniform((float)0.0, (float)1.0);
  float uniformRandom2 = Uniform((float)0.0, (float)1.0);


  float phi;	                // value for phi
  float cos_phi, sin_phi;	// cos and sin phi value
  float cos_theta, sin_theta;	// cos and sin theta value

  // Get the related value of phi
  phi = 2.0 * M_PI * uniformRandom1;
  cos_phi = cos(phi);
  sin_phi = sin(phi);
	
  // Get the related value of theta
  cos_theta = pow((float)1 - uniformRandom2, (float)1.0 / (nOrder + 1));
  sin_theta = sqrt((float)1.0 - cos_theta * cos_theta);
  

  // Rotation to align with the current ideal specular direction
  SJCVector3f X = normal % idealOut;
  SJCVector3f Y = idealOut % X;

  // This matrix is from world to X, y, normal
  SJCRotateMatrixf rotation(X, Y, idealOut);

  // From x, y, normal to world
  SJCRotateMatrixf toWorld = rotation.inverse();
  
  SJCVector3f outDir(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
  outDir = toWorld * outDir;
  outDir.normalize();
  
  pdf_value = (nOrder + 1.0) * pow (cos_theta, nOrder) / (2.0 * M_PI);
  
  return outDir;

}

//***************************************************************************
//
// * Makes a nice grid for stratified sampling 
//============================================================================
void SJCRandomf::
GetNrDivisions(int samples, int &divs1, int &divs2)
//=============================================================================
{
  // The valid sample should over 1
  if (samples <= 0) {
    divs1 = 0;
    divs2 = 0;
    return;
  }
  
  // Start will the square root value of samples and decrement of divs1
  // To find the matching point
  divs1 = (int)ceil(sqrt((float)samples));
  divs2 = samples / divs1;
  while (divs1 * divs2 != samples && divs1 > 1) {
    divs1 --;
    divs2 = samples / divs1;
  } // end of while
}




