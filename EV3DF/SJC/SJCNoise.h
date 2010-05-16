
/************************************************************************
     Main File:

     File:        SJCNoise.h

     Author:     
                  Steven Chenney, schenney@cs.wisc.edu

     Modifier    
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
  
     Comment:     
     
     Contructor:
                 

     Function:
  
************************************************************************/
#ifndef _SJCNOISE_H_
#define _SJCNOISE_H_

#include <SJCVector3.h>
#include <ostream>

class SJCNoise {
 private:
  static const uint	PERM_SIZE;    //
  static const uint  	PERM_ARRAY[]; //

  //
  static float Grad(const int x, const int y, const int z,
		    const float dx, const float dy, const float dz) {
    int h = PERM_ARRAY[PERM_ARRAY[PERM_ARRAY[x]+y]+z] & 15;
    float	u = ( h < 8 || h == 12 || h == 13 ) ? dx : dy;
    float	v = ( h < 4 || h == 12 || h == 13 ) ? dy : dz;
    return ( ( h & 1 ) ? -u : u ) + ( ( h & 2 ) ? -v : v );
  }

  //
  static float NoiseWeight(const float u) {
    float	u_cube = u * u * u;
    float	u_four = u_cube * u;
    return 6.0f * u_four * u - 15.0f * u_four + 10.0f * u_cube;
  }
  
  //
  static float Eval(const float x, const float y, const float z);
  
 public:
  // Constructor and destructor
  SJCNoise(void) { }
  ~SJCNoise(void) { }
  
  //
  static float    Evaluate(const float x, const float y, const float z,
			   const int octaves = 1, const float scale = 0.5);
  //
  static float    Evaluate(const SJCVector3f &v,
			   const int octaves = 1, const float scale = 0.5) {
    return Evaluate(v.x(), v.y(), v.z(), octaves, scale); }
};


#endif

