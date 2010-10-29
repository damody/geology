/************************************************************************ 
   Main File :

 
   File:           SJCArcLengthSpline.cpp 

    
   Author:                       
                   Yu-Chi Lai, yu-chi@cs.wisc.edu 

 
   Comment:        Based on cubic BSpline to find the arc length 
                   parameterization BSpline curve


   Functions:     
                   1. Constructor( loop, 3d)
                   2. Constructor( loop, b3d, size, SJCVector3f*)
                   3. Constructor( loop, b3d, vector<SJCVector3f>)
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
 
    
    Compiler:                    g++


   Platform:                    linux
*************************************************************************/    
#include <SJCArcLengthSpline.h> 

const uint    SJCArcLengthSplinef::m_ucNumMinControlPoints  = 10;
const uint    SJCArcLengthSplinef::m_ucControlPointsDensity = 20;
const float   SJCArcLengthSplinef::m_dcEvaluateIncrement    = 0.01f;
const uint    SJCArcLengthSplinef::m_ucNumArcLengthSample   = 100;
const float   SJCArcLengthSplinef::m_dcEpsilon              = 0.0001f;

//************************************************************************* 
//
// Operator =
//========================================================================== 
SJCArcLengthSplinef & SJCArcLengthSplinef::
operator =(const SJCArcLengthSplinef & src)
//========================================================================== 
{ 
  // Check whether this's address and the src's address is the same 
  // or not, the same we don't need to do anything
  if (this  != &src){ 
    // Clear all the path
    Clear();

    // Set up the loop flag
    m_bLoop = src.m_bLoop;

    // Set up the 3D flag
    m_b3D          = src.m_b3D;

    // Set up boundary
    m_vMinBounding = src.m_vMinBounding;
    m_vMaxBounding = src.m_vMaxBounding;

    // Set up the boundary
    m_sLabel       = src.m_sLabel;

    // Copy the control point
    int size = src.m_VControls.size();  
    m_VControls.resize(size); 
    for(int i = 0; i < size; i++){ 
      m_VControls[i] = src.m_VControls[i]; 
    } 
    
    // Copy the instance	 
    size = src.m_VInstances.size(); 
    m_VInstances.resize(size); 
    for(int i = 0; i < size; i++){ 
      m_VInstances[i] = src.m_VInstances[i]; 
    } 
    
    m_dLength = -1.f;
  }// end of else 

  return (*this);
} 

//*********************************************************************** 
// 
// Get num samples from the path
//============================================================================ 
void SJCArcLengthSplinef::
GetSamples(uint numSamples, vector<SJCVector3f>& samples)
//============================================================================ 
{ 
  if(m_dLength < 0)
    Length();
  
  // Calculate the increment
  float increment = m_dLength / (float)numSamples;
  samples.resize(numSamples);

  // Get the sample point from vertices
  for(uint i = 0; i < numSamples; i++){
    EvaluatePoint((float)i * increment, samples[i]);
  }
}
 
//************************************************************************ 
// 
// Evalute the curve position 
//============================================================================ 
SJCVector3f SJCArcLengthSplinef::
PointAtArcLength(const float arc_length, SJCVector3f& point)
//============================================================================ 
{ 
  float t = arc_length * m_dArcToCoeff;
  return SJCCubicBSplinef(t, point);
}

//************************************************************************ 
// 
// Evalute the curve derivative
//=========================================================================== 
SJCVector3f SJCArcLengthSplinef::
DerivativeAtArcLength(const float arc_length, SJCVector3f& deriv)
//============================================================================ 
{
  float t = arc_length * m_dArcToCoeff;
  return EvaluateDerivative(t, deriv);
} 

//************************************************************************ 
// 
// Evalute the curve derivative
//=========================================================================== 
SJCVector3f SJCArcLengthSplinef::
DerivativeAtArcLengthToArc(const float arc_length, SJCVector3f& deriv)
//============================================================================ 
{
  float t = arc_length * m_dArcToCoeff;
  return EvaluateDerivative(t, deriv) * m_dCoeffToArc;
} 

//************************************************************************ 
// 
// Evalute the curve derivative
//=========================================================================== 
float SJCArcLengthSplinef::
Length(void)
//============================================================================ 
{
  if(m_dLength > 0)
    return m_dLength;

  m_dLength = 0.f;
  
  float last_two = (float)NumControls() - 3.f - 2.f * m_dcEvaluateIncrement;
  float t = 0.f;
  
  for( float t = 0.f; t <= last_two; t += m_dcEvaluateIncrement){
    SJCVector3f dump;
    SJCVector3d P1 = EvaluatePoint(t, dump); 
    SJCVector3d P2 = EvaluatePoint(t + m_dcEvaluateIncrement, dump); 
    SJCVector3d V12 = P2 - P1; 
    m_dLength += V12.length(); 
  } // end of for

  // The very last segment
  SJCVector3f dump;
  SJCVector3d P1 = SJCCubicBSplinef::EvaluatePoint(t, dump); 
  SJCVector3d P2 = 
    SJCCubicBSplinef::EvaluatePoint((float)NumControls() - 3.f - SJC_EPSILON,
				    dump); 
  SJCVector3d V12 = P2 - P1; 
  m_dLength += V12.length(); 
  return m_dLength;
}

//************************************************************************ 
// 
// * Clear all the point in the path
//========================================================================== 
void SJCArcLengthSplinef::
Clear(void)
//========================================================================== 
{
  m_VControls.clear();
  m_VInstances.clear();
  m_dLength = -1.f;
  
}

//***************************************************************************
//
// * Using the frame position to find the least square fit curve 
// * numCPS: total number of control points
// * coeff:
// * framePos:
//=========================================================================== 
void SJCArcLengthSplinef::
ArcLengthLeastSquareFit( vector<SJCVector3f>& framePos) 
//===========================================================================  
{ 
  // Compute the initial path
  LeastSquareFit(framePos);

  // Compute the length
  Length();
    
  // - increment is to avoid undefined
  float sample_length = (m_dLength - SJC_EPSILON) / 
                        (m_ucNumArcLengthSamples - 1);

  vector<SJCVector3f> arc_samples;
  arc_samples.resize(m_ucNumArcLengthSamples);
  for(uint i = 0; i < m_ucNumArcLengthSamples; i++){
    arc_samples[i] = FindPointAtArcLength(sample_length * (float)i);
  }
  Clear();
  
  // Fit in the new path
  SJCCubicBSplinef::LeastSquareFit(arc_samples);

  // Compute the length
  Length();

  m_dArcToCoeff = m_dLength / (float)(NumControls() - 3);
  m_dCoeffToArc = 1.f / m_dArcToCoeff;
  
} 


//***************************************************************************
//
// * Using the frame position to find the least square fit curve 
// * numCPS: total number of control points
// * coeff:
// * framePos:
//=========================================================================== 
void SJCArcLengthSplinef::
ArcLengthLeastSquareFit( vector<SJCVector3f>& framePos, uint numCPs) 
//===========================================================================  
{ 
  // Compute the initial path
  LeastSquareFit(framePos, numCPs);

  // Compute the length
  m_dLength = Length();
    
  // - increment is to avoid undefined
  float sample_length = (m_dLength - SJC_EPSILON) / 
                        (m_ucNumArcLengthSamples - 1);

  vector<SJCVector3f> arc_samples;
  arc_samples.resize(m_ucNumArcLengthSamples);
  for(uint i = 0; i < m_ucNumArcLengthSamples; i++){
    arc_samples[i] = FindPointAtArcLength(sample_length * (float)i);
  }
		     
  SJCCubicBSplinef::LeastSquareFit(arc_samples, numCPs);

  // Compute the length
  Length();

  m_dArcToCoeff = m_dLength / (float)(NumControls() - 3);
  m_dCoeffToArc = 1.f / m_dArcToCoeff;
} 

//****************************************************************************
//
// * Before we normalize the spline
//============================================================================
SJCVector3f SJCArcLengthSplinef::
FindPointAtArcLength(const double arc_length) 
//============================================================================
{
  if(arc_length <= 0.f)
    return StartPoint();
  if(arc_length > m_dLength)
    return EndPoint();
  
  float accum = 0.f; 
  float last_l = 0.f;
  float last_two = (float)NumControls() - 3.f - 2.f * m_dcEvaluateIncrement;
  float t = 0.f;
  bool bFound = false;
  
 
  for( float t = 0.f; t <= last_two; t += m_dcEvaluateIncrement){
    SJCVector3f dump;
    SJCVector3d P1 = EvaluatePoint(t, dump); 
    SJCVector3d P2 = EvaluatePoint(t + m_dcEvaluateIncrement,
						     dump); 
    SJCVector3d V12 = P2 - P1; 
    last_l = V12.length();
    accum += last_l; 
    if(accum == arc_length)
      return P2;
    else if(accum > arc_length){
      bFound = true;
      accum -= last_l;
      break;
    } // end of else
  } // end of for

  float t_next;
  if(bFound)
    t_next = t + m_dcEvaluateIncrement;
  else {
    t_next = (float)NumControls() - 3.f - SJC_EPSILON;
    last_l = m_dLength - accum;
  }
  
  // I just do the interpolation and I did not use the bisection because
  // I assume that the evaluation increment is small enough to avoid deviation
  // It is only good for my implementation because I use
  SJCVector3f dump;
  SJCVector3d P1 = EvaluatePoint(t, dump); 
  SJCVector3d P2 = EvaluatePoint(t_next, dump); 

  // Compute the ratio
  float alpha = (arc_length - accum) / last_l;
  
  return alpha * P2 + (1.f - alpha) * P1;
}


//***************************************************************************
//
// * Map the ribbon coordinate to the world coordinate
//=========================================================================== 
SJCVector3f SJCArcLengthSplinef::
MapRPPointToWP(const SJCVector3f& RP)
//===========================================================================  
{ 
  SJCVector3f dump;
  SJCVector3f d, o, l;
  Direction(RP.x(), d, o, l);
  
  return PointAtArcLength(RP.x(), dump) + RP.y() * o;
}

//***************************************************************************
//
// * Compute the 3 direction at arc length
//=========================================================================== 
void SJCArcLengthSplinef::
Direction(const float arc_length, SJCVector3f &d, SJCVector3f &o, 
	  SJCVector3f &l)
//===========================================================================  
{ 
  d = DerivativeAtArcLength(arc_length);
  d.normalize();
  o.set(-d.y(), d.x(), 0.f);
  l = SJCConstants::SJC_vZAxis3f;
}


//***************************************************************************
//
// * Map the ribbon coordinate to the world coordinate
//=========================================================================== 
SJCVector3f SJCArcLengthSplinef::
MapWPPointToRP(const SJCVector3f& WP, const float start_s)
//===========================================================================  
{ 

  float s1, s2, s3, s_star;
  float ds1, ds2, ds3, ds_star;
    
  if(start_s < 0.f){
    float min_l = 1000000.f;
    float min_t = 0;
    float t_max = (float)( NumControls() - 3);
    
    
    for(float t = 0.f; i < t_max; t += 1.f){
      SJCVector point;
      EvaluatePoint(t, point);
      float l2 = (WP - point).lengthSquare();
      if( l2 < min_l){
	min_l = l2;
	min_t = t;
      } // end of if
    } // end of for
    
    // The very last point
    SJCVector point;
    EvaluatePoint(t_max - SJC_EPSILON, point);
    float l2 = (WP - point).lengthSquare();
    if( l2 < min_l){
      s1 = (t_max - 2.f) * m_dCoeffToArc;
      s2 = (t_max - 1.f) * m_dCoeffToArc;
      s3 = (t_max - SJC_EPSILON);
    } // end of if
    else if (min_t == 0.f){
      s1 = 0.f;
      s2 = m_dCoeffToArc;
      s3 = 2.f * m_dCoeffToArc;
    }
    else{
      s1 = m_dCoeffToArc * (t_max - 1.f);
      s2 = m_dCoeffToArc * t_max;
      s3 = m_dCoeffToArc * (t_max + 1.f);
    }
  } // end of if
  else {
    s2 = start_s * m_dArcToCoeff;
    s1 = (float)int( s2);
    s3 = (float)int( s2 + 0.5f);
  }



  // Quadratic method  
  SJCVector point;
  PointAtArcLength(s1, point);
  ds1 = (WP - point).lengthSquare();
  
  PointAtArcLength(s2, point);
  ds2 = (WP - point).lengthSquare();

  PointAtArcLength(s3, point);
  ds3 = (WP - point).lengthSquare();

  for( uint i = 0; i < 4; i++){
    float s1_sqr = s1 * s1;
    float s2_sqr = s2 * s2;
    float s3_sqr = s3 * s3;
    
    float y12 = s1_sqr - s2_sqr;
    float y23 = s2_sqr - s3_sqr;
    float y31 = s3_sqr - s1_sqr;

    float s12 = s1 - s2;
    float s23 = s2 - s3;
    float s31 = s3 - s1;
    
    s_star = .5f * ( y23 * ds1 + y31 * ds2 + y12 * ds3) / 
                   ( s23 * ds1 + s31 * ds2 + s12 * ds3);
    
    PointAtArcLength(s_star, point);
    ds_start = (WP - point).lengthSquare();
    
    if( ds1 <= ds2 && ds1 <= ds3 && ds1 < ds_star)
      PutInOrder(s1, s2, s3, s2, s3, s_star, 
		 ds1, ds2, ds3, ds2, ds3, ds_star);
    else if( ds2 <= ds1 && ds2 <= ds3 && ds2 < ds_star)
      PutInOrder(s1, s2, s3, s1, s3, s_star, 
		 ds1, ds2, ds3, ds1, ds3, ds_star);
    else if( ds3 <= ds1 && ds3 <= ds2 && ds3 < ds_star)
      PutInOrder(s1, s2, s3, s1, s2, s_star, 
		 ds1, ds2, ds3, ds1, ds2, ds_star);
    else {
      break;
    } 
  }// end of for
  
  float s, s_new;
  if( ds1 <= ds2 && ds1 <=ds3)
    s = s1;
  else if ( ds2 <=ds3)
    s = s2;
  else
    s = s3;
  
  // Newton's method
  for(; ; ){
    // Get the point
    SJCVector3f p, diff; 
    PointAtArcLength( s, p);
    diff = p - WP;
           
    // Get the first derivative
    SJCVector3f derive;
    DerivativeAtArcLengthToArc(s, derive);

    // Get the second derivative
    SJCVector3f second_derive;
    SecondDerivativeAtArcLengthToArc(s, second_derive);

    float dDs_ds = 2.f * ( diff.x() * derive.x() + diff.y() * derive.y() +
			   diff.z() * derive.z());
    float dDs2_ds2 = derive.x() * derive.x() * second_derive.x() * diff.x() +
                     derive.y() * derive.y() * second_derive.y() * diff.y() +
                     derive.z() * derive.z() * second_derive.z() * diff.z();
    
    s_new = s - dDs_ds / dDs2_ds2;

    if(fabs(s_new - s ) < m_dcEpsilon) {
      s = s_new;
      break;
    }

    s = s_new;
    
  } // end of for 
  SJCVector3f near_p;
  SJCVector3f near_d;

  // Get the near point
  PointAtArcLength(s, near_p);

  // Get the near direction
  DerivativeAtArcLength(s, near_d);
  near_d.normalize();
  
  SJCVector3f near_diff = WP - near_p;
  
  return SJCVector3f(s, near_diff * near_d, 0.f);
  
}


//***************************************************************************
//
// * ???? Is order matter????
//=========================================================================== 
void SJCArcLengthSplinef::
PutInOrder(float &s1, float &s2, float &s3, 
	   const float new_s1, const float new_s2, const float new_s3,
	   float &ds1, float &ds2, float &ds3,
	   const float new_ds1, const float new_ds2, const float new_ds3)
//===========================================================================  
{ 
  if(new_s1 <= new_s2 && new_s1 <= new_s3){
    s1  = new_s1;
    ds1 = new_ds1;
    if(new_s2 <= new_s3){
      s2 = new_s2;
      s3 = new_s3;
      ds2 = new_ds2;
      ds3 = new_ds3;
      
    }
    else {
      s2 = new_s3;
      s3 = new_s2;
      ds2 = new_ds3;
      ds3 = new_ds2;
    }
  }
  else if(new_s2 <= new_s1 && new_s2 <= new_s3){
    s1  = new_s2;
    ds1 = new_ds2;

    if(new_s1 <= new_s3){
      s2 = new_s1;
      s3 = new_s3;
      ds2 = new_ds1;
      ds3 = new_ds3;
    }
    else {
      s2 = new_s3;
      s3 = new_s1;
      ds2 = new_ds3;
      ds3 = new_ds1;
    }
  }
  else {
    s1  = new_s3;
    ds1 = new_ds3;

    if(new_s1 <= new_s2){
      s2 = new_s1;
      s3 = new_s2;
      ds2 = new_ds1;
      ds3 = new_ds2;
    }
    else {
      s2 = new_s2;
      s3 = new_s1;
      ds2 = new_ds2;
      ds3 = new_ds1;
    } // end of else
  } // end of else
}
