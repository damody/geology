/************************************************************************
   Main File :

 
   File:           SJCArcLengthSpline.h

    
   Author:         Yu-Chi Lai, yu-chi@cs.wisc.edu

                
   Comment:        Based on cubic BSpline to find the arc length 
                   parameterization BSpline curve

   Functions:     
                   1. Constructor( loop, 3d)
                   2. Constructor( loop, b3d, size, SJCVector3d*)
                   3. Constructor( loop, b3d, vector<SJCVector3d>)
                   4. Destructor
                   5. GetSamples:  Get sample points from the path
                   6. EvaluatePoint: get the pos at the coeff
                   7. EvaluateDerivative: get the derivative at the coeff
                   8. Length: get the total length of entire path
                   9. Clear: clear all data 
                  10. LeastSquareFit: Use the least square error algorithm to 
                      find the fitting curve
                  11. MapWPPointToRP: Map the world coordinate to ribbon 
                      coordinate
                  12. MapRPPointToWP: Map the ribbon coordinate to the world 
                      coordinate
 
   Compiler:       g++

 
   Platform:       Linux
*************************************************************************/    

#ifndef _SJC_ARC_LENGTH_SPLINE_H_
#define _SJC_ARC_LENGTH_SPLINE_H_


// The global definition information
#include <SJC.h> 
#include <SJCConstants.h>

// General C library
#include <stdio.h>
#include <math.h> 
#include <stdlib.h>

// General C++ library
#include <iostream> 
#include <vector> 
#include <string>

// My own library
#include <SJCVector3.h> 
#include <SJCQuaternion.h>

#include <SJCErrorHandling.h>
using namespace std;

class SJCArcLengthSplinef : public SJCCubicBSplined
{
 protected:
  // The minimum number of control points for least square fitting algorithm
  static const uint   m_ucNumMinControlPoints;  

  // The average number of vertices we should put a control point
  static const uint   m_ucControlPointsDensity;

  // The total number of resample for arc length
  static const uint   m_ucNumArcLengthSample;
  
  // The increment evaluation in control coefficient
  static const float  m_dcEvaluateIncrement;

  static const float  m_dcEpsilon;
  

 public:
	 
  //********************************************************
  // Constructors and destructor
  //******************************************************** 
  SJCArcLengthSplinef(const bool loop_var = true, const bool b3d = false)
    : SJCCubicBSpline(loop_var, b3d){ 
    m_dLength = -1.f;
  }
  
  // Initilize spline with size and loop vars and control points
  SJCArcLengthSplinef(const bool loop_var, const bool b3d, int size,
		      SJCVector3f *control_points)
    : SJCCubicBSpline(loop_var, b3d, size, control_points) {
    m_dLength = -1.f;
  }
  
  // Initilize spline with dim and loop vars and control points 
  SJCArcLengthSplinef(const bool loop_var, const bool b3d, 
		      vector<SJCVector3f>& control_points)
    : SJCCubicBSpline(loop_var, b3d, size, control_points) {
    m_dLength = -1.f;
  }
		
  // Destructor
  virtual ~SJCArcLengthSplinef(void){ }

  //*************************************************************
  // Operator
  //************************************************************
  // Copy Operator
  SJCArcLengthSplinef & operator=(const SJCArcLengthSplinef &);

  //**************************************************************
  // Access function
  //**************************************************************
  // Get sample points from the path
  virtual void GetSamples(uint numSamples, vector<SJCVector3f>& samples);

  // Evaluate curve at a parameter and return result in SJCVector3f 
  // throws exception if parameters out of range, unless looping. 
  SJCVector3f PointAtArcLength(const float, SJCVector3f&); 
  
  //Evaluate derivitive at a given parameter and return result in SJCVector3f 
  //throws exception if parameters out of range, unless looping. 
  // dP / dt
  SJCVector3f DerivativeAtArcLength(const float, SJCVector3f&); 

  // dP / ds = dP / dt * dt / ds
  SJCVector3f DerivativeAtArcLengthToArc(const float arc_length, 
					 SJCVector3f& deriv);

  // dP2 / ds2 = dP2 / dt2 * dt2 / ds2
  SJCVector3f SecondDerivativeAtArcLengthToArc(const float arc_length, 
					       SJCVector3f& deriv);

  // Get 3 direction of at the arc_length
  void Direction(const float arc_length, SJCVector3f &d,
			SJCVector3f &o, SJCVector3f &l);

   
  // The total length of the path 
  virtual float Length(void);
  virtual void Clear(void);
  
  //******************************************************************* 
  // Find the fitting curve 
  //******************************************************************* 
  // Use the least square error algorithm to find the fitting curve
  void ArcLengthLeastSquareFit(vector<SJCVector3f>&);

  // Use the least square error algorithm to find the fitting curve
  void ArcLengthLeastSquareFit(vector<SJCVector3f>&, uint num_controls);
  
 public:

  // Map the world coordinate to ribbon coordinate
  SJCVector3f MapWPPointToRP(const SJCVector3f& point);

  // Map the ribbon coordinate to the world coordinate
  SJCVector3f MapRPPointToWP(const SJCVector3f& point);

 public: 
  // Output operator
  friend ostream& operator<<(ostream &o, const SJCArcLengthSplinef& g);

 protected:
  float    m_dLength;     // The total length for the arc length
  float    m_dArcToCoeff; // The ratio from arc length to coeff
  float    m_dCoeffToArc; // The ratio from coeff to arc length       
};

#endif
