/************************************************************************ 
   Main File :                  Main.cpp

 
   File:                        SJCCubicBSpline.cpp 

    
   Author:                       
                                Yu-Chi Lai, yu-chi@cs.wisc.edu 

 
   Comment:                     The BSpline curve 


   Functions:     
                   1. Constructor( loop, 3d)
                   2. Constructor( loop, b3d, size, SJCVector3d*)
                   3. Constructor( loop, b3d, std::vector<SJCVector3d>)
                   4. Destructor
                   5. Label: set and get the label of the curve
                   6. Set( loop, 3d): set up the loop and 3d flag
                   7. CalBound: Calculate the bound of the path
                   8. MinBound, MaxBound: get the max and min bound of path
                   9. StartPoint, StartDirection: get the start pos and dir
                  10. EndPoint, EndDirection: get the end pos and dir
                  11. NumControls: number of control points
                  12. NumInstances: number of instance points
                  13. GetControl: get the index control point
                  14. GetControls: get all control points
                  15. GetInstance: get the index instance point
                  16. GetInstances: get all instance points
                  17. GetSamples:  Get sample points from the path
                  18. EvaluatePoint: get the pos at the coeff
                  19. EvaluateDerivative: get the derivative at the coeff
                  20. Length: get the total length of entire path
                  21. Clear: clear all data 
                  22. SetControl: change the index control points
                  23. SetControls: change the entire control points
                  24. SetControlComponent: set all X, Y, or Z component for
                      certain values
                  25. AppendControl: Add a control point to the end of the 
                      spline 
                  26. InsertControl: Insert a control point.
                  27. DeleteControl: Remove a control point
                  28. MovePath:  Translate the entire path
                  29. Transform: Transform the entire path
                  30. RefineTolerance:  Check whether the new formed spline 
                      within certain threshold 
                  31. LeastSquareFit: Use the least square error algorithm to 
                      find the fitting curve
                  32. LeastSquareFitWithLimitAtStartAndEnd: Use the least 
                      square error algorithm to find the fitting curve
                  33. MapPointToPath: Given an arbitrary point ("A"), 
                      returns the nearest point ("P") on this path.
                  34. MapPathDistanceToPoint: Given a distance along the path, 
                      convert it to a point on the path
                  35. MapPathDistanceToDirection:  Given a distance along the 
                      path, convert it to a point on the path
                  35. MapPathDistanceToPositionDirection
                  36. MapPointToPathDistance: Given an arbitrary point, 
                      convert it to a distance along the path.
                  37. MapPointToPathSection: Given an arbitrary point ("A"), 
                      returns the nearest point ("P") on this path in the 
                      section with path length between start and end
                  38. Create2DPath: create a 2D circular path
                  39. Create3DPath: create a 3D circular path

    
    Compiler:                    g++


   Platform:                    linux
*************************************************************************/    
#include "SJCCubicBSpline.h"
#include <vector>

const uint    SJCCubicBSplined::m_ucNumMinControlPoints = 10;
const uint    SJCCubicBSplined::m_ucControlPointsDensity = 20;
const double  SJCCubicBSplined::m_dcEvaluateIncrement   = 0.01f;

const uint    SJCCubicBSplinef::m_ucNumMinControlPoints = 10;
const uint    SJCCubicBSplinef::m_ucControlPointsDensity = 20;
const float   SJCCubicBSplinef::m_dcEvaluateIncrement   = 0.01f;

//************************************************************************** 
//
// Constructor 
//=========================================================================== 
SJCCubicBSplined::
SJCCubicBSplined(const bool loop_var, const bool b3d, int size, 
		SJCVector3d *points)
//=========================================================================== 
{
  // Set up the loop flag 
  m_bLoop = loop_var; 

  // Set up 3D information
  m_b3D = b3d;
  
  // Read the data in 
  m_VControls.resize(size); 
  for(int i = 0; i < size; i++){
    m_VControls[i]  = points[i]; 
    m_VInstances[i] = points[i]; 
  }
} 

//**************************************************************************
// 
// Constructor 
//========================================================================== 
SJCCubicBSplined::
SJCCubicBSplined(const bool loop_var, const bool b3d,
	      std::vector<SJCVector3d>& points) 
//========================================================================== 
{ 
  // Set up the loop flag 
  m_bLoop = loop_var; 

  // Set up the 3d informatin
  m_b3D = b3d;

  // Get the size of the std::vector 
  int size = points.size(); 
  
  // Read the data in 
  m_VControls.resize(size); 
  for(int i = 0; i < size; i++){ 
    m_VControls[i]  = points[i]; 
    m_VInstances[i] = points[i]; 
  }
} 
 
 
//************************************************************************* 
//
// Operator =
//========================================================================== 
SJCCubicBSplined & SJCCubicBSplined::
operator =(const SJCCubicBSplined & src)
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
    m_b3D   = src.m_b3D;
    m_vMinBounding = src.m_vMinBounding;
    m_vMaxBounding = src.m_vMaxBounding;
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
    
  }// end of else 

  return (*this);
} 

//************************************************************************ 
// 
// Get the control point according to index 
//========================================================================== 
void SJCCubicBSplined::
CalBound(void)
//========================================================================== 
{ 
  m_vMinBounding.set(SJC_INFINITE, SJC_INFINITE, SJC_INFINITE);
  m_vMaxBounding.set(-SJC_INFINITE, -SJC_INFINITE, -SJC_INFINITE);
  if(!m_VControls.size())
    return;

  for(uint i = 0; i < m_VControls.size(); i++){
    if(m_VControls[i].x() < m_vMinBounding.x())
       m_vMinBounding.x( m_VControls[i].x());

    if(m_VControls[i].y() < m_vMinBounding.y())
       m_vMinBounding.y( m_VControls[i].y());

    if(m_VControls[i].z() < m_vMinBounding.z())
       m_vMinBounding.z( m_VControls[i].z());

    if(m_VControls[i].x() > m_vMaxBounding.x())
      m_vMaxBounding.x( m_VControls[i].x());

    if(m_VControls[i].y() > m_vMaxBounding.y())
      m_vMaxBounding.y( m_VControls[i].y());

    if(m_VControls[i].z() > m_vMaxBounding.z())
      m_vMaxBounding.z( m_VControls[i].z());
  }
} 

//************************************************************************ 
// 
// Get the start point
//========================================================================== 
SJCVector3d SJCCubicBSplined::
StartPoint(void)
//========================================================================== 
{
  double start_coeff = 0.f;
  SJCVector3d pos;
  EvaluatePoint(start_coeff, pos);
  return pos;
}

//************************************************************************ 
// 
// Get the start direction
//========================================================================== 
SJCVector3d SJCCubicBSplined::
StartDirection(void)
//========================================================================== 
{
  double start_coeff = 0.f;
  SJCVector3d deriv;
  EvaluateDerivative(start_coeff, deriv);
  deriv.normalize();
  return deriv;
}

//************************************************************************ 
// 
// Get the position of the end point 
//========================================================================== 
SJCVector3d SJCCubicBSplined::
EndPoint(void)
//========================================================================== 
{
  double end_coeff = (double)(m_VControls.size() - 3) - 0.00001;
  SJCVector3d pos;

  EvaluatePoint(end_coeff, pos);
  return pos;
}

//************************************************************************ 
// 
// Get the end direction of the path 
//========================================================================== 
SJCVector3d SJCCubicBSplined::
EndDirection(void)
//========================================================================== 
{
  double end_coeff = (double)(m_VControls.size() - 3) - 0.01;
  SJCVector3d deriv;
  EvaluateDerivative(end_coeff, deriv);
  deriv.normalize();
  return deriv;
}

//************************************************************************ 
// 
// Get the control point according to index 
//========================================================================== 
void SJCCubicBSplined::
GetControl(const unsigned short index, SJCVector3d& point)
//========================================================================== 
{ 
  // The index out of range
  if (index >= m_VControls.size())
    SJCError("SJCCubicBSplined::ControlPoint - Index out of range");
  
  // Return the point
  point = m_VControls[index];
}

//************************************************************************ 
// 
// Get the control point according to index 
//========================================================================== 
SJCVector3d SJCCubicBSplined::
GetControl(const unsigned short index)
//========================================================================== 
{ 
  // The index out of range
  if (index >= m_VControls.size())
    SJCError("SJCCubicBSplined::ControlPoint - Index out of range");
  
  // Return the point
  return m_VControls[index];
}

//************************************************************************ 
// 
// Get the instance point according to index 
//========================================================================== 
void SJCCubicBSplined::
GetInstance(const unsigned short index, SJCVector3d& point) 
//========================================================================== 
{ 
  // The index out of range 
  if (index >= m_VInstances.size()) 
    SJCError("SJCCubicBSplined::InstancePoint - Index out of range"); 
  
  // Return the point 
  point = m_VInstances[index]; 
} 

//*********************************************************************** 
// 
// Get num samples from the path
//============================================================================ 
void SJCCubicBSplined::
GetSamples(uint numSamples, std::vector<SJCVector3d>& samples)
//============================================================================ 
{ 
  // Calculate the increment
  double increment = (double)(m_VControls.size() - 3) / 
                     (double)numSamples;
  samples.resize(numSamples);

  // Get the sample point from vertices
  for(uint i = 0; i < numSamples; i++){
    EvaluatePoint((double)i * increment, samples[i]);
  }
}
 
//************************************************************************ 
// 
// Evalute the curve position 
//============================================================================ 
SJCVector3d SJCCubicBSplined::
EvaluatePoint(const double t, SJCVector3d& point)
//============================================================================ 
{ 
  // For the index in the curse
  int index; 
  int localIndex; 
  
  // For local parametric value
  double u;
  double uSquare;
  double uCube; 
  
  // Basic function
  double basis[4]; 

  //pick the previous control point 
  index = (int) floor(t);
  
  // Because we need for point to evaluate P0(n-4), P1(n-3), P2(n-2), P3(n-1)
  if ((index > (int)m_VControls.size() - 4) && (!m_bLoop)) 
    SJCError("SJCCubicBSplined::EvaluatePoint - Parameter out of range");
  
  // Calculate the parametric value
  u	  = t - index; 
  uSquare = u * u;
  uCube	  = u * uSquare;
  
  //evaluate basic function
  basis[0] =       -uCube + 3.0 * uSquare - 3.0 * u + 1.0;
  basis[1] =  3.0 * uCube - 6.0 * uSquare + 4.0;
  basis[2] = -3.0 * uCube + 3.0 * uSquare + 3.0 * u + 1.0;
  basis[3] = uCube;

  //sum up the control points * the basis function for each dimension
  // i =  control points, j = dimensino
  point.set(0, 0, 0); 
  

  for(int i = 0; i < 4; i++){ 
    // For loop calculation
    localIndex = (i + index) % m_VControls.size();

    point += m_VControls[localIndex] * basis[i];
  } 
  
  // For the average coefficient
  point /= 6.0;

  return point;
}

//************************************************************************ 
// 
// Evalute the curve derivative
//=========================================================================== 
SJCVector3d SJCCubicBSplined::
EvaluateDerivative(const double t, SJCVector3d& deriv)
//============================================================================ 
{
  // For the index in the curse 
  int	 index; 
  int	 localIndex; 
  
  // For local parametric value 
  double u; 
  double uSquare; 
  double uCube; 
  // Basic function 
  double basis[4]; 
  
  //pick the previous control point 
  index = (int) floor(t);
  
  // Because we need for point to evaluate P0(n-4), P1(n-3), P2(n-2), P3(n-1) 
  if ((index > (int)m_VControls.size() - 4) && (!m_bLoop))  
    SJCError("SJCCubicBSplined::EvaluatePoint - Parameter out of range"); 
  
  // Calculate the parametric value 
  u	  = t - index; 
  uSquare = u * u; 
  uCube	  = u * uSquare; 
  
  //eval the derivs of the basis functions
  basis[0] = -3.0 * uSquare +  6.0 * u - 3.0;
  basis[1] =  9.0 * uSquare - 12.0 * u;
  basis[2] = -9.0 * uSquare +  6.0 * u + 3.0;
  basis[3] =  3.0 * uSquare;
 
  //sum up the control points * the basis function for each dimension 
  // i =  control points, j = dimensino 
  deriv.set(0, 0, 0); 
  
  for(int i = 0; i < 4; i++){ 
    // For loop calculation 
    localIndex = (i + index) % m_VControls.size(); 
    deriv += m_VControls[localIndex] * basis[i]; 
  } 
  // For the average coefficient 
  deriv /= 6.0; 

  return deriv;
} 

//**************************************************************** 
// 
// Calculate the length of the B spline line 
//================================================================ 
double SJCCubicBSplined::
Length() 
//================================================================ 
{ 
  double l = 0; 
 
  for(uint i = 0; i < m_VInstances.size() - 1; i++){ 
    SJCVector3d P1 = m_VInstances[i]; 
    SJCVector3d P2 = m_VInstances[i + 1]; 
    SJCVector3d V12 = P2 - P1; 
    l += V12.length(); 
  } 
  return l; 
} 
//************************************************************************ 
// 
// * Clear all the point in the path
//========================================================================== 
void SJCCubicBSplined::
Clear(void)
//========================================================================== 
{
  m_VControls.clear();
  m_VInstances.clear();
}

//**************************************************************************** 
// 
// Set the control points
//============================================================================ 
void SJCCubicBSplined::
SetControls(std::vector<SJCVector3d>& points)
//============================================================================ 
{ 
  m_VControls.resize(points.size());
  m_VInstances.resize(points.size());

  for(uint i = 0; i < points.size(); i++){
    m_VControls[i] = points[i];
    m_VInstances[i] = points[i];
  }
 
} 

//**************************************************************************** 
// 
// Set the control point according to index 
//============================================================================ 
void SJCCubicBSplined::
SetControl(const unsigned short index, const SJCVector3d& point)
//============================================================================ 
{ 
  // Out of bound
  if (index >= m_VControls.size())
    SJCError("SJCCubicBSplined::SetControl - Index out of range");
  
  m_VControls[index] = point; 
} 

//*********************************************************************** 
// 
// Set x, y, or z = certain value 
//============================================================================ 
void SJCCubicBSplined::
SetControlComponent(int direction, double value) 
//============================================================================ 
{ 
  if(direction < 0 || direction > 0)
    SJCError("The indexing is wrong\n");

  for(uint i = 0; i < m_VControls.size(); i++) 
    m_VControls[i][direction] = value; 
} 

//************************************************************************ 
// 
// Append the control point at the end
//============================================================================ 
void SJCCubicBSplined::
AppendControl(const SJCVector3d& point)
//============================================================================ 
{
  m_VControls.push_back(point);
}

//************************************************************************ 
// 
// Insert a control point into the pos. Throws an exception  
// if the pos is beyond the end. 
//============================================================================ 
void SJCCubicBSplined::
InsertControl(const unsigned short index,  const SJCVector3d& point) 
//========================================================================== 
{ 
  // Out of range 
  if(index > m_VControls.size()) 
    SJCError("SJCCubicBSplined::ControlPoint - Index out of range"); 
  
  std::vector<SJCVector3d>::iterator insertP = m_VControls.begin() + index; 
  m_VControls.insert(insertP, point); 
} 

//************************************************************************ 
// 
// Delete the control point according to index 
//============================================================================ 
void SJCCubicBSplined::
DeleteControl(const unsigned short index)
//============================================================================ 
{ 
  // Out of bound
  if (index >= m_VControls.size())
    SJCError("SJCCubicBSplined::DeleteControl - Index out of range");
 
  std::vector<SJCVector3d>::iterator deleteP = m_VControls.begin() + index;
  m_VControls.erase(deleteP);
} 

//****************************************************************************
//
// Move the entire path to certain distance 
//=========================================================================== 
void SJCCubicBSplined::
MovePath(SJCVector3d& howfar)
//=========================================================================== 
{
  for (uint i = 0; i < m_VControls.size(); i++){
    m_VControls[i] = m_VControls[i] + howfar;
  }
  for (uint i = 0; i < m_VInstances.size(); i++){
    m_VInstances[i] = m_VInstances[i] + howfar;
  }
}

//****************************************************************************
//
// * Transform the entire path 
//=========================================================================== 
void SJCCubicBSplined::
Transform(SJCQuaterniond& r, SJCVector3d& t)
//=========================================================================== 
{
  for (uint i = 0; i < m_VControls.size(); i++){
    m_VControls[i] = r * m_VControls[i] + t;
  }
  for (uint i = 0; i < m_VInstances.size(); i++){
    m_VInstances[i] = r * m_VInstances[i] +t;
  }
}



//************************************************************************ 
// 
// Evalute the curve velocity, Problem whether the refinement will reserve 
// The first and end points' position 
//=========================================================================== 
void SJCCubicBSplined::
Refine(SJCCubicBSplined &result)
//============================================================================ 
{ 
  // Loop variable 
  int i, j; 
  std::vector<SJCVector3d> old; 
  
  // Get the size of original control elements 
  int numInstances = result.m_VInstances.size(); 
  
  // Resize the old one 
  old.resize(numInstances); 

  // Copy the information in 
  for(i = 0; i < numInstances; i++) 
    old[i] = result.m_VInstances[i]; 

  // Calc new drawing points count
  if (m_bLoop)
    result.m_VInstances.resize(numInstances * 2);
  else
    result.m_VInstances.resize(numInstances * 2 - 3);
  
  int newNumInstances = result.m_VInstances.size(); 
  
  // i is index at new array 
  // j is index at old array
  for(i = 0, j = 0; i < newNumInstances; i+=2, j++){
    
    // These are the indices of the points to average
    SJCVector3d p0 = old[j % numInstances];
    SJCVector3d p1 = old[(j + 1) % numInstances];
    SJCVector3d p2 = old[(j + 2) % numInstances];
    
    result.m_VInstances[i] = 0.5 * (p0 + p1);
    if(i + 1 < newNumInstances)
      result.m_VInstances[i+1] = 0.125 * (p0 + p1 * 6.0 + p2);
  } // Duplicate the result
}

//*************************************************************************** 
// 
// Refine the std::vector until it reach the tolerance 
//============================================================================ 
void SJCCubicBSplined::
RefineTolerance(SJCCubicBSplined &result, const double tolerance)
//============================================================================ 
{
  Refine(result);
  while(! WithinTolerance(tolerance))
    result.Refine(result);
}

//************************************************************************ 
// 
// Check the tolerance P1, P2, P3 
// The distance square between  P2 to line P1P3 is smaller than tolerance 
//============================================================================ 
bool SJCCubicBSplined::
WithinTolerance(const double tolerance)
//============================================================================ 
{ 
  // V12's projection point on V13
  SJCVector3d projectionPoint;
 
  // Point for the three points 
  SJCVector3d p1, p2, p3; 

  // The std::vector between p1 P2 and p1 p3, and p2 projection 
  SJCVector3d V12, V13, V2P; 
  
  double toleranceSquare = tolerance * tolerance; 
  
  // Total number to calculate
  int num = m_bLoop ? m_VInstances.size() : m_VInstances.size() - 2;
  
  // Go through all point in the curve 
  int numInstances = m_VInstances.size(); 
  for ( int i = 0 ; i < num ; i++ ) { 
    // Get the  control point in the curve
    p1 = m_VInstances[i % numInstances]; 
    p2 = m_VInstances[(i + 1) % numInstances]; 
    p3 = m_VInstances[(i + 2) % numInstances]; 
    
    // Calculate the std::vector 
    V12 = p2 - p1; 
    V13 = p3 - p1; 
    
    // Compute the V13 length square 
    double lengthSquare = V13 * V13; 
    if (lengthSquare == 0.0) 
      continue; 
    
    // V12 on V13's projection will be
    double ratio =  (V12 * V13) / lengthSquare; 
 
    projectionPoint = p1 + V13 * ratio; 
    
    V2P = projectionPoint - p2; 
    if( (V2P * V2P) > toleranceSquare) 
      return false; 
  } 
  
  return true;
}
 

//***************************************************************************
//
// * Using the frame position to find the least square fit curve 
// * numCPS: total number of control points
// * coeff:
// * framePos:
//=========================================================================== 
void SJCCubicBSplined::
LeastSquareFit( std::vector<SJCVector3d>& framePos) 
//===========================================================================  
{ 
  double         coeffi, coeffj; 
  double         **coefficient; 
  std::vector<double> x; 
  std::vector<double> y;
  std::vector<double> z; 
  uint           numCPs = framePos.size() /m_ucControlPointsDensity;
  if(numCPs < (uint)m_ucNumMinControlPoints)
    numCPs = (uint)m_ucNumMinControlPoints;

  numCPs = 12;


  double increment = (double)(numCPs - 3) / (double)framePos.size();

  x.resize(numCPs);
  y.resize(numCPs); 
  z.resize(numCPs); 

  // Allocate the N by N coefficient array
  coefficient	 = new double*[numCPs]; 
  for(uint i =0; i < numCPs; i++) 
    coefficient[i] = new double[numCPs]; 

  // Initialize the array
  for(uint i =0; i < numCPs; i++) 
    for(uint j = 0; j < numCPs; j++) 
      coefficient[i][j] = 0; 
  
  for(uint i =0; i < numCPs; i++){ 
    x[i] = 0; 
    y[i] = 0;
    z[i] = 0; 
    
    for(uint k = 0; k < framePos.size(); k++){ 
      double coeff =  increment * (double)k;
      coeffi = BasicFunction(i, coeff); 

      x[i] += coeffi * framePos[k][0];
      y[i] += coeffi * framePos[k][1];
      z[i] += coeffi * framePos[k][2]; 
      
      for(uint j = 0; j < numCPs; j++){ 
	coeffj = BasicFunction(j, coeff); 
	coefficient[i][j] += coeffi * coeffj; 
      }// end of j 
    }// end of k 
  }// end of i 


  std::vector<double> diagonal; 
  std::vector<double> cx;
  std::vector<double> cy; 
  std::vector<double> cz; 
  diagonal.resize(numCPs);
  cx.resize(numCPs);
  cy.resize(numCPs);
  cz.resize(numCPs);
  for(uint i = 0; i < numCPs; i++){
    diagonal[i] = cx[i] = cy[i] = cz[i] = 0;
  }
 
  Cholesky(numCPs, coefficient, diagonal); 
 
  Solver(numCPs, coefficient, diagonal, x, cx)	; 
  Solver(numCPs, coefficient, diagonal, y, cy)	;
 
  if(m_b3D)
    Solver(numCPs, coefficient, diagonal, z, cz)	; 

 
  m_VControls.clear(); 
  m_VControls.resize(numCPs);
  m_VInstances.clear();
  m_VInstances.resize(numCPs);
 
  for(uint i = 0; i < numCPs; i++){ 
    m_VControls[i][0] = cx[i]; 
    m_VControls[i][1] = cy[i]; 
    if(m_b3D)
      m_VControls[i][2] = cz[i];
    else
      m_VControls[i][2] = 0;
    m_VInstances[i] = m_VControls[i];
  } 

   
  for(uint i =0; i < numCPs; i++) 
    delete [] coefficient[i]; 
  delete [] coefficient; 
} 


//***************************************************************************
//
// * Using the frame position to find the least square fit curve 
// * numCPS: total number of control points
// * coeff:
// * framePos:
//=========================================================================== 
void SJCCubicBSplined::
LeastSquareFit( std::vector<SJCVector3d>& framePos, uint numCPs) 
//===========================================================================  
{ 
  double         coeffi, coeffj; 
  double         **coefficient; 
  std::vector<double> x; 
  std::vector<double> y;
  std::vector<double> z; 


  double increment = (double)(numCPs - 3) / (double)framePos.size();

  x.resize(numCPs);
  y.resize(numCPs); 
  z.resize(numCPs); 

  // Allocate the N by N coefficient array
  coefficient	 = new double*[numCPs]; 
  for(uint i =0; i < numCPs; i++) 
    coefficient[i] = new double[numCPs]; 

  // Initialize the array
  for(uint i =0; i < numCPs; i++) 
    for(uint j = 0; j < numCPs; j++) 
      coefficient[i][j] = 0; 
  
  for(uint i =0; i < numCPs; i++){ 
    x[i] = 0; 
    y[i] = 0;
    z[i] = 0; 
    
    for(uint k = 0; k < framePos.size(); k++){ 
      double coeff =  increment * (double)k;
      coeffi = BasicFunction(i, coeff); 

      x[i] += coeffi * framePos[k][0];
      y[i] += coeffi * framePos[k][1];
      z[i] += coeffi * framePos[k][2]; 
      
      for(uint j = 0; j < numCPs; j++){ 
	coeffj = BasicFunction(j, coeff); 
	coefficient[i][j] += coeffi * coeffj; 
      }// end of j 
    }// end of k 
  }// end of i 


  std::vector<double> diagonal; 
  std::vector<double> cx;
  std::vector<double> cy; 
  std::vector<double> cz; 
  diagonal.resize(numCPs);
  cx.resize(numCPs);
  cy.resize(numCPs);
  cz.resize(numCPs);
  for(uint i = 0; i < numCPs; i++){
    diagonal[i] = cx[i] = cy[i] = cz[i] = 0;
  }
 
  Cholesky(numCPs, coefficient, diagonal); 
 
  Solver(numCPs, coefficient, diagonal, x, cx)	; 
  Solver(numCPs, coefficient, diagonal, y, cy)	;
 
  if(m_b3D)
    Solver(numCPs, coefficient, diagonal, z, cz)	; 

 
  m_VControls.clear(); 
  m_VControls.resize(numCPs);
  m_VInstances.clear();
  m_VInstances.resize(numCPs);
 
  for(uint i = 0; i < numCPs; i++){ 
    m_VControls[i][0] = cx[i]; 
    m_VControls[i][1] = cy[i]; 
    if(m_b3D)
      m_VControls[i][2] = cz[i];
    else
      m_VControls[i][2] = 0;
    m_VInstances[i] = m_VControls[i];
  } 

   
  for(uint i =0; i < numCPs; i++) 
    delete [] coefficient[i]; 
  delete [] coefficient; 
} 
 
//***************************************************************************
//
// * Using the frame position to find the least square fit curve 
// * numCPS: total number of control points
// * framePos: the desired position for them to pass
//=========================================================================== 
void SJCCubicBSplined::
LeastSquareFitWithLimitAtStartAndEnd(std::vector<SJCVector3d>& framePos,
				     uint numCPs) 
//===========================================================================  
{ 
  // The increment for each step
  double increment     = (double)(numCPs - 3) / (double)framePos.size();

  // Calcluate the interval where last control points is in effect
  double last_interval = (double)(numCPs - 4);

  // Compute the least square fit number
  uint  num_fitCPs    = numCPs - 2;
  uint  num_fitFrames = framePos.size() - 2;
  uint  last_frame    = framePos.size() - 1;


  // Create the least square metric
  std::vector<std::vector<double> > A(num_fitFrames);
  std::vector<double>          b_x(num_fitFrames);
  std::vector<double>          b_y(num_fitFrames);
  std::vector<double>          b_z(num_fitFrames);
 
  // Create the m by n matrix, where m is the number of fit frames, n is the
  // number of control points
  for(uint i = 0; i < num_fitFrames; i++){
    A[i].resize(num_fitCPs);
  }

  // Compute the cofficent A * C = b
  for(uint i = 0; i < num_fitFrames; i++) {

    // Get the coefficient for this frames
    double coeff =  increment * (double)(i + 1); // +1, we start from frame 1

    if(coeff < 1.0){ // for those affected by first control points
                     // We need to redistributed its coefficient to others

      // Compute the contribution of coefficient 0
      double coeff0 = BasicFunction(0, coeff);
      b_x[i] = framePos[i + 1][0] - framePos[0][0] * coeff0 * 6.0;
      b_y[i] = framePos[i + 1][1] - framePos[0][1] * coeff0 * 6.0;
      b_z[i] = framePos[i + 1][2] - framePos[0][2] * coeff0 * 6.0;

      // Compute the matrix coefficient
      for(uint j = 0; j < num_fitCPs; j++){
	if( j == 0) // Need to do redistributed to first, and second turn
	  A[i][j] = BasicFunction(j + 1, coeff) - 4.f * coeff0;
	else if (j == 1)
	  A[i][j] = BasicFunction(j + 1, coeff) - coeff0;
	else 
	  A[i][j] = BasicFunction(j + 1, coeff);
      } // end of for
    }// end of if

    if (coeff > last_interval){ // for those affected by last control
                                // We need to redistributed its 
                                // coefficient to others

      // Compute the coefficient
      double coeff_last = BasicFunction(numCPs - 1, coeff);
 
      // Redistributed
      b_x[i] = framePos[i + 1][0] - framePos[last_frame][0] * coeff_last * 6.0;
      b_y[i] = framePos[i + 1][1] - framePos[last_frame][1] * coeff_last * 6.0;
      b_z[i] = framePos[i + 1][2] - framePos[last_frame][2] * coeff_last * 6.0;

      for(uint j = 0; j < num_fitCPs; j++){
	if( j == num_fitCPs - 1) // need to redistributed for the last two term
	  A[i][j] = BasicFunction(j + 1, coeff) - 4.f * coeff_last;
	else if (j == num_fitCPs - 2)
	  A[i][j] = BasicFunction(j + 1, coeff) - coeff_last;
	else 
	  A[i][j] = BasicFunction(j + 1, coeff);
      } // end of for
    } // end of else
    else { // inbetween nothing affected
      b_x[i] = framePos[i + 1][0];
      b_y[i] = framePos[i + 1][1];
      b_z[i] = framePos[i + 1][2];

      for(uint j = 0; j < num_fitCPs; j++){
	A[i][j] = BasicFunction(j + 1, coeff);
      } // end of for 
    } // end of else
  }
  
  double         **coefficient; 

 std::vector<double> x(num_fitCPs); 
 std::vector<double> y(num_fitCPs);
 std::vector<double> z(num_fitCPs);
   
 // Allocate the N by N coefficient array
 coefficient	 = new double*[num_fitCPs]; 
 for(uint i = 0; i < num_fitCPs; i++) {
   x[i] = 0; 
   y[i] = 0;
   z[i] = 0; 

   coefficient[i] = new double[num_fitCPs]; 
   for(uint j = 0; j < num_fitCPs; j++) 
      coefficient[i][j] = 0; 
 }// end of for i
 
  
  // coeff = A^T * A
  // A^T * A * C = A^T * b
  for( uint i = 0; i < num_fitCPs; i++) {
    for( uint j = 0; j < num_fitFrames; j++) {
     
      // Compute the output  A^T(i, j) = A(j, i)
      x[i] += A[j][i] * b_x[j];
      y[i] += A[j][i] * b_y[j];
      z[i] += A[j][i] * b_z[j];
      
      // Compute A^T . A
      for(uint k = 0; k < num_fitCPs; k++){ 
	coefficient[i][k] += A[j][i] * A[j][k]; 
      }// end of for k
    } // end of for j
  } // end of for i


  //*********************************************************************
  // Solve it
  //*********************************************************************
  std::vector<double> diagonal; 
  std::vector<double> cx;
  std::vector<double> cy; 
  std::vector<double> cz; 

  diagonal.resize(num_fitCPs);

  cx.resize(num_fitCPs);
  cy.resize(num_fitCPs);
  cz.resize(num_fitCPs);

  for(uint i = 0; i < num_fitCPs; i++){
    diagonal[i] = cx[i] = cy[i] = cz[i] = 0;
  }
  
  // Cholesky decomposition
  Cholesky(num_fitCPs, coefficient, diagonal); 
 

  // Solve x, y, z
  Solver(num_fitCPs, coefficient, diagonal, x, cx)	; 
  Solver(num_fitCPs, coefficient, diagonal, y, cy)	;
  if(m_b3D)
    Solver(num_fitCPs, coefficient, diagonal, z, cz)	; 

  
  // Clear the control and instances 
  m_VControls.clear(); 
  m_VControls.resize(numCPs);
  m_VInstances.clear();
  m_VInstances.resize(numCPs);

 
  // For the first points
  m_VControls[0][0] = 6.f * framePos[0][0] - 4.f * cx[0] - cx[1];
  m_VControls[0][1] = 6.f * framePos[0][1] - 4.f * cy[0] - cy[1];
  if(m_b3D)
    m_VControls[0][2] = 6.f * framePos[0][2] - 4.f * cz[0] - cz[1];
  else
    m_VControls[0][2] = 0.f;
  m_VInstances[0] = m_VControls[0];


  // Copy All fit points 
  for(uint i = 0; i < num_fitCPs; i++){ 
    m_VControls[i + 1][0] = cx[i]; 
    m_VControls[i + 1][1] = cy[i]; 
    if(m_b3D)
      m_VControls[i + 1][2] = cz[i];
    else
      m_VControls[i + 1][2] = 0;
    m_VInstances[i + 1] = m_VControls[i + 1];
  } 

  // For the last points
  m_VControls[numCPs - 1][0] = 
    6.f * framePos[last_frame][0] - 4.f * cx[num_fitCPs - 1] - 
    cx[num_fitCPs - 2];
  m_VControls[numCPs - 1][1] = 
    6.f * framePos[last_frame][1] - 4.f * cy[num_fitCPs - 1] - 
    cy[num_fitCPs - 2];
  if(m_b3D)
    m_VControls[numCPs - 1][2] = 
      6.f * framePos[last_frame][2] - 4.f * cz[num_fitCPs - 1] - 
      cz[num_fitCPs - 2];
  else
    m_VControls[numCPs - 1][2] = 0.f;

  m_VInstances[numCPs - 1] = m_VControls[numCPs - 1];
   
  for(uint i = 0; i < num_fitCPs; i++) 
    delete [] coefficient[i]; 
  delete [] coefficient; 
 
  




} 
 
//***************************************************************************
//
// To get the basic coefficient for non-loop
//===========================================================================
double SJCCubicBSplined::
BasicFunction(int pointIndex, double t)
//===========================================================================  
{ 
  int index; 
  
  double u; 
  double uSquare; 
  double uCube; 
  double basis; 
  
  index = (uint) floor(t); 
  
  if ((index > (int)m_VControls.size() - 4) && (!m_bLoop))  
    std::cout<<"something wrong in CubicBSpline::basicFunction"<< std::endl; 
  
  u       = t - index; 
  uSquare = u * u; 
  uCube   = u * uSquare; 
  
  //given an index, find it's A value 
  if(index == pointIndex) 
    basis = -uCube + 3.0 * uSquare - 3.0 * u + 1.0; 
  else if(index == pointIndex - 1) 
    basis = 3.0 * uCube - 6.0 * uSquare + 4.0; 
  else if(index == pointIndex - 2) 
    basis = -3.0 * uCube + 3.0 * uSquare + 3.0 * u + 1.0; 
  else if(index == pointIndex - 3) 
    basis = uCube; 
  else 
    basis = 0.0; 
  
  basis = basis / 6.0; 
  
  return basis; 
} 

//***************************************************************************
//
// Use Cholesky to solve function
//===========================================================================  
void SJCCubicBSplined::
Cholesky(int num, double **decomp, std::vector<double>& diagonal) 
//===========================================================================  
{ 
 
  double sum; 
  int i, j, k; 
  for(i = 0; i < num; i++){ 
    for(j = i; j < num; j++){ 
      for(sum = decomp[i][j], k = i - 1; k >= 0; k--) 
	sum -= decomp[i][k] * decomp[j][k]; 
      if(i == j){ 
	if(sum <= 0.0){ 
	  std::cout<<"sum is less than 0 in cholesky"<< std::endl; 
	  return; 
	} 
	diagonal[i] = sqrt(sum); 
      } 
      else 
	decomp[j][i] = sum / diagonal[i]; 
    } 
  } 
} 
 
//*************************************************************************** 
//
// Solve the constraint question
//=========================================================================== 
void SJCCubicBSplined::
Solver(int num, double **decomp, std::vector<double>& diagonal, 
       std::vector<double>& b, std::vector<double>& x) 
//=========================================================================== 
{

  double sum;
  int i, k;

 
  for(i = 0; i < num; i++){
    for( sum = b[i], k = i - 1; k >= 0; k--)
      sum -= decomp[i][k] * x[k];

     x[i] = sum / diagonal[i];
 
  }// end of for i
  for(i = num - 1; i >= 0; i--){
    for(sum = x[i], k = i + 1; k < num; k++)
      sum -= decomp[k][i] * x[k];
    x[i] = sum / diagonal[i];
  }// end of for i
} 

//****************************************************************************
//
// * Given an arbitrary point ("A"), returns the nearest point ("P") on
//   this path.  Also returns, via output arguments, the path tangent at
//   P and a measure of how far A is outside the Pathway's "tube".  Note
//   that a negative distance indicates A is inside the Pathway.
//=========================================================================== 
SJCVector3d SJCCubicBSplined::
MapPointToPath (const SJCVector3d& point, SJCVector3d& tangent,
		double& distance_to_center)
//=========================================================================== 
{
  double max_coeff = m_VControls.size() - 3;
  distance_to_center = SJC_INFINITE;
  SJCVector3d nearP;
  double     nearC;
  // Go throught all sample point to find the nearest position
  for(double coeff = 0.f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    SJCVector3d temp;
    EvaluatePoint(coeff, temp);
    SJCVector3d difference = temp - point;
    double dist = difference.length();
    if(dist < distance_to_center){ // smaller distance found
      nearP = temp;
      nearC = coeff;
      distance_to_center = dist;
    } // end of if
  } // end of for
  EvaluateDerivative(nearC, tangent);
  return nearP;
}

//****************************************************************************
//
// * Given a distance along the path, convert it to a point on the path
//=========================================================================== 
SJCVector3d SJCCubicBSplined::
MapPathDistanceToPoint (double pathDistance)
//=========================================================================== 
{
  double max_coeff = m_VControls.size() - 3;
  double accumulate = 0.f;

  SJCVector3d prev_p;
  SJCVector3d current_p;
  EvaluatePoint(0.f, prev_p);
  current_p = prev_p;
  
  // Go throught all sample point to find the nearest position
  for(double coeff = 0.1f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    EvaluatePoint(coeff, current_p);
    SJCVector3d difference = current_p - prev_p;
    double dist = difference.length();
    accumulate += dist;
    if(accumulate >= pathDistance){
      return current_p;
    }
    prev_p = current_p;
  }// end of for
  return current_p;
}

//****************************************************************************
//
// * Given a distance along the path, convert it to a point on the path
//=========================================================================== 
SJCVector3d SJCCubicBSplined::
MapPathDistanceToDirection (double pathDistance)
//=========================================================================== 
{
  double max_coeff = m_VControls.size() - 3;
  double accumulate = 0.f;

  SJCVector3d prev_p;
  SJCVector3d current_p;
  EvaluatePoint(0.f, prev_p);
  current_p = prev_p;
  
  // Go throught all sample point to find the nearest position
  for(double coeff = 0.1f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    EvaluatePoint(coeff, current_p);
    SJCVector3d difference = current_p - prev_p;
    double dist = difference.length();
    accumulate += dist;

    if(accumulate >= pathDistance){
      if(dist > 0.00001)
	return difference.normal();
    }
    prev_p = current_p;
  }// end of for
  return EndDirection();
}
//****************************************************************************
//
// * Given a distance along the path, convert it to a point on the path
//=========================================================================== 
void SJCCubicBSplined::
MapPathDistanceToPositionDirection(double pathDistance, SJCVector3d& pos,
				   SJCVector3d& dir)
//=========================================================================== 
{
  double max_coeff = m_VControls.size() - 3;
  double accumulate = 0.f;

  SJCVector3d prev_p;
  SJCVector3d current_p;
  EvaluatePoint(0.f, prev_p);
  current_p = prev_p;
  
  // Go throught all sample point to find the nearest position
  for(double coeff = 0.1f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    EvaluatePoint(coeff, current_p);
    SJCVector3d difference = current_p - prev_p;
    double dist = difference.length();
    accumulate += dist;
    if(accumulate >= pathDistance){
      pos = current_p;
      dir = difference.normal();
      return;
    }
    prev_p = current_p;
  }// end of for
  pos = current_p;
  dir = EndDirection();
  return;
}

//****************************************************************************
//
// * Given an arbitrary point, convert it to a distance_square along the path.
//=========================================================================== 
double SJCCubicBSplined::
MapPointToPathDistance (const SJCVector3d& point, SJCVector3d& pathPoint,
			SJCVector3d& tangent, double& distance_to_center)
//=========================================================================== 
{
  double max_coeff = m_VControls.size() - 3;
  distance_to_center  = SJC_INFINITE;
  SJCVector3d nearP;
  double    nearC;

  // Go throught all sample point to find the nearest position
  for(double coeff = 0.f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    SJCVector3d temp;
    EvaluatePoint(coeff, temp);
    SJCVector3d difference = temp - point;
    double dist = difference.length();
    if(dist < distance_to_center){ // smaller distance found
      nearP = temp;
      nearC = coeff;
      distance_to_center = dist;
    } // end of if
  } // end of for

  pathPoint  = nearP;
  EvaluateDerivative(nearC, tangent);

  double accumulate = 0.f;

  SJCVector3d prev_p;
  SJCVector3d current_p;
  EvaluatePoint(0.f, prev_p);

  // Go throught all sample point to find the nearest position
  for(double coeff = 0.1f; coeff <= nearC; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    EvaluatePoint(coeff, current_p);
    SJCVector3d difference = current_p - prev_p;
    double dist = difference.length();
    accumulate += dist;
    prev_p = current_p;
    }// end of for
  return accumulate;
}


//****************************************************************************
//
// * Given an arbitrary point ("A"), returns the nearest point ("P") on
//   this path between distance start and end.  
//=========================================================================== 
SJCVector3d SJCCubicBSplined::
MapPointToPathSection (const SJCVector3d& point, const double path_start,
		       const double path_end, const double step, 
		       double& best_path_length)
//=========================================================================== 
{
  SJCVector3d curr(0.f, 0.f, 0.f);
  double     min = 1e20;

  for(double path_dist = path_start; path_dist <= path_end; path_dist += step){
    SJCVector3d path_point = MapPathDistanceToPoint(path_dist);
    double dist_sqrt = (path_point - point).lengthSquare();
    if(dist_sqrt < min){
      min = dist_sqrt;
      curr = path_point;
      best_path_length = path_dist;
    } // end of if
  }// end of for


  return curr;
}

//************************************************************************ 
// 
// Create a path with length and projected to an arc with angle
// + angle means right turn 
// - angle means left turn
// * start at (0, 0), and center is on the y axis
//========================================================================== 
void SJCCubicBSplined::
Create2DPath(const double angle, const double length)
//========================================================================== 
{
  Clear();
  m_bLoop = false;
  m_b3D   = false;
  const uint point_size = 60;

  std::vector<SJCVector3d> pathPoints(point_size);
  if(angle == 0.f){
    double increment = length / (double)point_size;
    for(uint i = 0; i < point_size; i++){
      pathPoints[i].set((double)i * increment, 0.f, 0.f);
    }
  }
  else if (angle > 0.f){
    double angle_radian = fabs(angle) * SJC_DEG_TO_RAD;
    double radius = length / angle_radian;
    double angle_inc = angle_radian / 60.f;
    for(uint i = 0; i < point_size; i++){
      double a = M_PI_2 - (double)i * angle_inc;
      double x = radius * cos(a);
      double y = radius * sin(a) - radius;
      pathPoints[i].set(x, y, 0.f);
    }
  }
  else {
    double angle_radian = fabs(angle) * SJC_DEG_TO_RAD;
    double radius = length / angle_radian;
    double angle_inc = angle_radian / 60.f;

    for(uint i = 0; i < point_size; i++){
      double a = - M_PI_2 + (double)i * angle_inc;
      double x = radius * cos(a);
      double y = radius * sin(a) + radius;

      pathPoints[i].set(x, y, 0.f);
    }// end of for
  }// end of else
  LeastSquareFit(pathPoints);
  RefineTolerance(*this, 0.1f);
  CalBound();
}


//************************************************************************ 
// 
// Create a path with length and projected to an arc with angle
// + angle means right turn 
// - angle means left turn
// * start at (0, 0), and center is on the y axis
//========================================================================== 
void SJCCubicBSplined::
Create3DPath(const double theta, const double phi, const double length)
//========================================================================== 
{
  Clear();

  m_bLoop = false;
  m_b3D   = true;

  const uint point_size = 60;

  std::vector<SJCVector3d> pathPoints(point_size);
  if(theta == 0.f){
    double increment = length / (double)point_size;
    for(uint i = 0; i < point_size; i++){
      pathPoints[i].set((double)i * increment, 0.f, 0.f);
    }
  }
  else {
    double theta_radian = fabs(theta) * SJC_DEG_TO_RAD;
    double radius       = length / theta_radian;
    double theta_inc    = theta_radian / 60.f;
    double cos_phi      = cos( phi * SJC_DEG_TO_RAD);
    double sin_phi      = sin( phi * SJC_DEG_TO_RAD);

    for(uint i = 0; i < point_size; i++){
      double a = M_PI_2 - (double)i * theta_inc;
      double x = radius * cos(a);
      
      double yz = radius * sin(a) - radius;
      double y  = yz * cos_phi;
      double z  = yz * sin_phi;
      pathPoints[i].set(x, y, z);
    }
  }

  LeastSquareFit(pathPoints);
  RefineTolerance(*this, 0.1f);
  CalBound();
}


//************************************************************************** 
//
// * transform from xyz to zxy configuration
//=========================================================================== 
void SJCCubicBSplined::
XYZToZXY(void)
//=========================================================================== 
{
  for(uint i = 0; i < m_VControls.size(); i++){
    SJCVector3d xyz = m_VControls[i];
    m_VControls[i].set(xyz.y(), xyz.z(), xyz.x());
  }

  for(uint i = 0; i < m_VInstances.size(); i++){
    SJCVector3d xyz = m_VInstances[i];
    m_VInstances[i].set(xyz.y(), xyz.z(), xyz.x());
  }
  
  SJCVector3d bound  = m_vMinBounding; 
  m_vMinBounding.set(bound.y(), bound.z(), bound.x());

  bound = m_vMaxBounding;
  m_vMaxBounding.set(bound.y(), bound.z(), bound.x());

}

//*****************************************************************************
//
// * output operator
//============================================================================
std::ostream& operator<<(std::ostream &o, const SJCCubicBSplined& g)
//============================================================================
{
  if(g.m_bLoop)
    o << "LoopFlag true\n";
  else
    o << "LoopFlag false\n";
 
  if(g.m_b3D)
    o << "IN3DFlag true\n";
  else
    o << "IN3DFlag false\n";

  o << "NumControls " << g.m_VControls.size() << " { ";
  for(uint i = 0; i < g.m_VControls.size(); i++){
    o << g.m_VControls[i] << " ";
  }
  o << " } ";

  return o;
  
}// end of output operator

//*****************************************************************************
//
// * Create the exactly fit b-spline
//=============================================================================
void SJCCubicBSplined::
ExactFitSplineToData(const std::vector<SJCVector3d>& data) 
//=============================================================================
{
  // Make the number of control points equal to the amount of data + 2, so that
  // we can have an exact fit to the data
  int npts  = m_VControls.size() - 2;
  int dsize = data.size();
  if (npts > dsize) {
    m_VControls.resize(dsize + 2);
  } 
  else if (npts < dsize) {
    for (int i = npts; i < dsize; i++) {
      AppendControl(SJCVector3d(static_cast<double>(i), 0, 0));
    } // end of for
  }
  FitSplineToData(data);
}

//****************************************************************************
//
// * Fit the spline to the data
//============================================================================
void  SJCCubicBSplined::
FitSplineToData(const std::vector<SJCVector3d>& data) 
//============================================================================
{
  // Set up the control points
  int npts = m_VControls.size() - 2;
  int i, j;
  int dsize = data.size();

  // Error message
  if (data.size() == 0)
    SJCError("Whoa!  Moron!  You didn't give me any data");

  // Only one data in the spline
  if (data.size() == 1) {
    m_VControls.resize(0);
    for (int i = 0; i < 4; i++) {
      AppendControl(data[0]);
    }
    return;
  }
  
  if (dsize < npts) {
    if (dsize >= 2) {
      SJCWarning("Amount of data too small for number of control points; "
		 "deleting some control points");
      m_VControls.resize(dsize + 2);
      npts = dsize;
    }
    else {
      SJCWarning("Fewer than 2 data points; aborting");
      return;
    }
  } // end of data
  
  std::vector<SJCVector3d> points;
  if (dsize == npts) {
    points = data;
  } 
  else {
    int incr = dsize / npts;
    for (i = 0; i < dsize; i += incr) {
      points.push_back(data[i]);
    }
  }
  
  // We already can choose values for the end control points
  // (we're underconstrained, and all that)
  // ---- first ----
  SetControl(0, points[0] + (points[0] - points[1]));
  points[0] = points[0] - GetControl(0) / 6.f;
  
  // ---- last ----
  SetControl(m_VControls.size() - 1,
	     points[points.size() - 1] +
	     (points[points.size() - 1] - points[points.size() - 2]));
  points[points.size() - 1] = (points[points.size()-1] - 
			       GetControl(m_VControls.size()-1) /
			       6.f);
   
  // Initialize arrays for the solver
  double **a = new double * [npts];
  double **b = new double * [npts];
  for (i = 0; i < npts; i++) {
    a[i] = new double [npts]; 
    b[i] = new double [3];
  }

  // Build arrays for the solver
  for (i = 0; i < npts; i++) {
    // Build a...
    for (j = 0; j < npts; j++) {
      if (j == i-1 || j == i+1) {
	a[i][j] = 1.f / 6.f;
      }
      else if (j == i)          {
	a[i][j] = 4.f / 6.f;
      }
      else  {
	a[i][j] = 0;
      }
     }
    // Build b...
    b[i][0] = points[i].x(); 
    b[i][1] = points[i].y(); 
    b[i][2] = points[i].z();
  }
   
  // Solve
  if (gaussj(a, npts, b, 3)) {
    // Set control points
    for (i = 0; i < npts; i++) {
      m_VControls[i+1] = SJCVector3d(b[i][0], b[i][1], b[i][2]);
    } // end of for 
  } // end of if
  else {
    SJCWarning("Solver failed to find a solution; not fitting BSpline");
  } // end of else 
  
  // delete the arrays used in the solver
  for (i = 0; i < npts; i++) {
    delete [] a[i];
    delete [] b[i];
  }

  delete [] a;
  delete [] b;
}

#define SWAP(a,b) {temp=(a); (a) = (b); (b) = temp;}
//****************************************************************************
//
// * THE SOLVER from Numerical Recipies in C
//============================================================================
bool SJCCubicBSplined::
gaussj(double **a, int n, double **b, int m) 
//============================================================================
{
  int *indxc, *indxr, *ipiv = NULL;
  indxc = new int[n];
  indxr = new int[n];
  ipiv  = new int[n];
  int i, icol = 0, irow = 0, j, k, l, ll;
  double big, dum, pivinv, temp;
  
  for (i = 0; i < n; i++) {
    indxc[i] = indxr[i] = i; 
    ipiv[i] = 0;
  }
  
  for (i = 0; i < n; i++) {
    big = 0.f;
    for (j = 0; j < n; j++) {
      if (ipiv[j] != 1) {
	for (k = 0; k < n; k++) {
	  if (ipiv[k] == 0) {
	    if (fabs(a[j][k]) >= big) {
	      big = fabsf(a[j][k]);
	      irow = j;
	      icol = k;
	    }
	  } else if (ipiv[k] > 1) {
	    SJCWarning("Singular Matrix-1");
	    return false;
	  } // end of else if
	} // end of for k
      } // end of if ipiv
    } // end of for j
    ++(ipiv[icol]);
    if (irow != icol) {
      for (l = 0; l < n; l++) {SWAP(a[irow][l],a[icol][l]);}
      for (l = 0; l < m; l++) {SWAP(b[irow][l],b[icol][l]);}
    }
    indxr[i]=irow;
    indxc[i]=icol;
    if (a[icol][icol] == 0) {
      SJCWarning("Singular Matrix-2");
      return false;
    }
    pivinv=1.0f/a[icol][icol];
    a[icol][icol] = 1.f;

    for (l = 0; l < n; l++) {
      a[icol][l] *= pivinv;
    }

    for (l = 0; l < m; l++) {
      b[icol][l] *= pivinv;
    }
    for (ll = 0; ll < n; ll++) {
      if (ll != icol) {
	dum = a[ll][icol];
	a[ll][icol] = 0.0f;
	for (l = 0; l < n; l++) {
	  a[ll][l] -= a[icol][l] * dum;
	}
	for (l = 0; l < m; l++) {
	  b[ll][l] -= b[icol][l] * dum;
	}
      } // end of for if
    }  // end of for ll
  } // end of for i
  
  for (l = n-1; l >= 0; l--) {
    if (indxr[l] != indxc[l]) {
      for (k = 0; k < n; k++) {SWAP(a[k][indxr[l]],a[k][indxc[l]]);}
    }
  }
  delete [] indxc;
  delete [] indxr;
  delete [] ipiv;
  return true;
}


//************************************************************************** 
//
// Constructor 
//=========================================================================== 
SJCCubicBSplinef::
SJCCubicBSplinef(const bool loop_var, const bool b3d, int size, 
		SJCVector3f *points)
//=========================================================================== 
{
  // Set up the loop flag 
  m_bLoop = loop_var; 

  // Set up 3D information
  m_b3D = b3d;
  
  // Read the data in 
  m_VControls.resize(size); 
  for(int i = 0; i < size; i++){
    m_VControls[i]  = points[i]; 
    m_VInstances[i] = points[i]; 
  }
} 

//**************************************************************************
// 
// Constructor 
//========================================================================== 
SJCCubicBSplinef::
SJCCubicBSplinef(const bool loop_var, const bool b3d,
	      std::vector<SJCVector3f>& points) 
//========================================================================== 
{ 
  // Set up the loop flag 
  m_bLoop = loop_var; 

  // Set up the 3d informatin
  m_b3D = b3d;

  // Get the size of the std::vector 
  int size = points.size(); 
  
  // Read the data in 
  m_VControls.resize(size); 
  for(int i = 0; i < size; i++){ 
    m_VControls[i]  = points[i]; 
    m_VInstances[i] = points[i]; 
  }
} 
 
 
//************************************************************************* 
//
// Operator =
//========================================================================== 
SJCCubicBSplinef & SJCCubicBSplinef::
operator =(const SJCCubicBSplinef & src)
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
    m_b3D   = src.m_b3D;
    m_vMinBounding = src.m_vMinBounding;
    m_vMaxBounding = src.m_vMaxBounding;
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
    
  }// end of else 

  return (*this);
} 

//************************************************************************ 
// 
// Get the control point according to index 
//========================================================================== 
void SJCCubicBSplinef::
CalBound(void)
//========================================================================== 
{ 
  m_vMinBounding.set(SJC_INFINITE, SJC_INFINITE, SJC_INFINITE);
  m_vMaxBounding.set(-SJC_INFINITE, -SJC_INFINITE, -SJC_INFINITE);
  if(!m_VControls.size())
    return;

  for(uint i = 0; i < m_VControls.size(); i++){
    if(m_VControls[i].x() < m_vMinBounding.x())
       m_vMinBounding.x( m_VControls[i].x());

    if(m_VControls[i].y() < m_vMinBounding.y())
       m_vMinBounding.y( m_VControls[i].y());

    if(m_VControls[i].z() < m_vMinBounding.z())
       m_vMinBounding.z( m_VControls[i].z());

    if(m_VControls[i].x() > m_vMaxBounding.x())
      m_vMaxBounding.x( m_VControls[i].x());

    if(m_VControls[i].y() > m_vMaxBounding.y())
      m_vMaxBounding.y( m_VControls[i].y());

    if(m_VControls[i].z() > m_vMaxBounding.z())
      m_vMaxBounding.z( m_VControls[i].z());
  }
} 

//************************************************************************ 
// 
// Get the start point
//========================================================================== 
SJCVector3f SJCCubicBSplinef::
StartPoint(void)
//========================================================================== 
{
  float start_coeff = 0.f;
  SJCVector3f pos;
  EvaluatePoint(start_coeff, pos);
  return pos;
}

//************************************************************************ 
// 
// Get the start direction
//========================================================================== 
SJCVector3f SJCCubicBSplinef::
StartDirection(void)
//========================================================================== 
{
  float start_coeff = 0.f;
  SJCVector3f deriv;
  EvaluateDerivative(start_coeff, deriv);
  deriv.normalize();
  return deriv;
}

//************************************************************************ 
// 
// Get the position of the end point 
//========================================================================== 
SJCVector3f SJCCubicBSplinef::
EndPoint(void)
//========================================================================== 
{
  float end_coeff = (float)(m_VControls.size() - 3) - 0.00001;
  SJCVector3f pos;

  EvaluatePoint(end_coeff, pos);
  return pos;
}

//************************************************************************ 
// 
// Get the end direction of the path 
//========================================================================== 
SJCVector3f SJCCubicBSplinef::
EndDirection(void)
//========================================================================== 
{
  float end_coeff = (float)(m_VControls.size() - 3) - 0.01;
  SJCVector3f deriv;
  EvaluateDerivative(end_coeff, deriv);
  deriv.normalize();
  return deriv;
}

//************************************************************************ 
// 
// Get the control point according to index 
//========================================================================== 
void SJCCubicBSplinef::
GetControl(const unsigned short index, SJCVector3f& point)
//========================================================================== 
{ 
  // The index out of range
  if (index >= m_VControls.size())
    SJCError("SJCCubicBSplinef::ControlPoint - Index out of range");
  
  // Return the point
  point = m_VControls[index];
}

//************************************************************************ 
// 
// Get the control point according to index 
//========================================================================== 
SJCVector3f SJCCubicBSplinef::
GetControl(const unsigned short index)
//========================================================================== 
{ 
  // The index out of range
  if (index >= m_VControls.size())
    SJCError("SJCCubicBSplinef::ControlPoint - Index out of range");
  
  // Return the point
  return m_VControls[index];
}

//************************************************************************ 
// 
// Get the instance point according to index 
//========================================================================== 
void SJCCubicBSplinef::
GetInstance(const unsigned short index, SJCVector3f& point) 
//========================================================================== 
{ 
  // The index out of range 
  if (index >= m_VInstances.size()) 
    SJCError("SJCCubicBSplinef::InstancePoint - Index out of range"); 
  
  // Return the point 
  point = m_VInstances[index]; 
} 

//*********************************************************************** 
// 
// Get num samples from the path
//============================================================================ 
void SJCCubicBSplinef::
GetSamples(uint numSamples, std::vector<SJCVector3f>& samples)
//============================================================================ 
{ 
  // Calculate the increment
  float increment = (float)(m_VControls.size() - 3) / 
                     (float)numSamples;
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
SJCVector3f SJCCubicBSplinef::
EvaluatePoint(const float t, SJCVector3f& point)
//============================================================================ 
{ 
  // For the index in the curse
  int index; 
  int localIndex; 
  
  // For local parametric value
  float u;
  float uSquare;
  float uCube; 
  
  // Basic function
  float basis[4]; 

  //pick the previous control point 
  index = (int) floor(t);
  
  // Because we need for point to evaluate P0(n-4), P1(n-3), P2(n-2), P3(n-1)
  if ((index > (int)m_VControls.size() - 4) && (!m_bLoop)) 
    SJCError("SJCCubicBSplinef::EvaluatePoint - Parameter out of range");
  
  // Calculate the parametric value
  u	  = t - index; 
  uSquare = u * u;
  uCube	  = u * uSquare;
  
  //evaluate basic function
  basis[0] =       -uCube + 3.0 * uSquare - 3.0 * u + 1.0;
  basis[1] =  3.0 * uCube - 6.0 * uSquare + 4.0;
  basis[2] = -3.0 * uCube + 3.0 * uSquare + 3.0 * u + 1.0;
  basis[3] = uCube;

  //sum up the control points * the basis function for each dimension
  // i =  control points, j = dimensino
  point.set(0, 0, 0); 
  

  for(int i = 0; i < 4; i++){ 
    // For loop calculation
    localIndex = (i + index) % m_VControls.size();

    point += m_VControls[localIndex] * basis[i];
  } 
  
  // For the average coefficient
  point /= 6.0;

  return point;
}

//************************************************************************ 
// 
// Evalute the curve derivative
//=========================================================================== 
SJCVector3f SJCCubicBSplinef::
EvaluateDerivative(const float t, SJCVector3f& deriv)
//============================================================================ 
{
  // For the index in the curse 
  int	 index; 
  int	 localIndex; 
  
  // For local parametric value 
  float u; 
  float uSquare; 
  float uCube; 
  // Basic function 
  float basis[4]; 
  
  //pick the previous control point 
  index = (int) floor(t);
  
  // Because we need for point to evaluate P0(n-4), P1(n-3), P2(n-2), P3(n-1) 
  if ((index > (int)m_VControls.size() - 4) && (!m_bLoop))  
    SJCError("SJCCubicBSplinef::EvaluatePoint - Parameter out of range"); 
  
  // Calculate the parametric value 
  u	  = t - index; 
  uSquare = u * u; 
  uCube	  = u * uSquare; 
  
  //eval the derivs of the basis functions
  basis[0] = -3.0 * uSquare +  6.0 * u - 3.0;
  basis[1] =  9.0 * uSquare - 12.0 * u;
  basis[2] = -9.0 * uSquare +  6.0 * u + 3.0;
  basis[3] =  3.0 * uSquare;
 
  //sum up the control points * the basis function for each dimension 
  // i =  control points, j = dimensino 
  deriv.set(0, 0, 0); 
  
  for(int i = 0; i < 4; i++){ 
    // For loop calculation 
    localIndex = (i + index) % m_VControls.size(); 
    deriv += m_VControls[localIndex] * basis[i]; 
  } 
  // For the average coefficient 
  deriv /= 6.0; 

  return deriv;
} 

//**************************************************************** 
// 
// Calculate the length of the B spline line 
//================================================================ 
float SJCCubicBSplinef::
Length() 
//================================================================ 
{ 
  float l = 0; 
 
  for(uint i = 0; i < m_VInstances.size() - 1; i++){ 
    SJCVector3f P1 = m_VInstances[i]; 
    SJCVector3f P2 = m_VInstances[i + 1]; 
    SJCVector3f V12 = P2 - P1; 
    l += V12.length(); 
  } 
  return l; 
} 
//************************************************************************ 
// 
// * Clear all the point in the path
//========================================================================== 
void SJCCubicBSplinef::
Clear(void)
//========================================================================== 
{
  m_VControls.clear();
  m_VInstances.clear();
}

//**************************************************************************** 
// 
// Set the control points
//============================================================================ 
void SJCCubicBSplinef::
SetControls(std::vector<SJCVector3f>& points)
//============================================================================ 
{ 
  m_VControls.resize(points.size());
  m_VInstances.resize(points.size());

  for(uint i = 0; i < points.size(); i++){
    m_VControls[i] = points[i];
    m_VInstances[i] = points[i];
  }
 
} 

//**************************************************************************** 
// 
// Set the control point according to index 
//============================================================================ 
void SJCCubicBSplinef::
SetControl(const unsigned short index, const SJCVector3f& point)
//============================================================================ 
{ 
  // Out of bound
  if (index >= m_VControls.size())
    SJCError("SJCCubicBSplinef::SetControl - Index out of range");
  
  m_VControls[index] = point; 
} 

//*********************************************************************** 
// 
// Set x, y, or z = certain value 
//============================================================================ 
void SJCCubicBSplinef::
SetControlComponent(int direction, float value) 
//============================================================================ 
{ 
  if(direction < 0 || direction > 0)
    SJCError("The indexing is wrong\n");

  for(uint i = 0; i < m_VControls.size(); i++) 
    m_VControls[i][direction] = value; 
} 

//************************************************************************ 
// 
// Append the control point at the end
//============================================================================ 
void SJCCubicBSplinef::
AppendControl(const SJCVector3f& point)
//============================================================================ 
{
  m_VControls.push_back(point);
}

//************************************************************************ 
// 
// Insert a control point into the pos. Throws an exception  
// if the pos is beyond the end. 
//============================================================================ 
void SJCCubicBSplinef::
InsertControl(const unsigned short index,  const SJCVector3f& point) 
//========================================================================== 
{ 
  // Out of range 
  if(index > m_VControls.size()) 
    SJCError("SJCCubicBSplinef::ControlPoint - Index out of range"); 
  
  std::vector<SJCVector3f>::iterator insertP = m_VControls.begin() + index; 
  m_VControls.insert(insertP, point); 
} 

//************************************************************************ 
// 
// Delete the control point according to index 
//============================================================================ 
void SJCCubicBSplinef::
DeleteControl(const unsigned short index)
//============================================================================ 
{ 
  // Out of bound
  if (index >= m_VControls.size())
    SJCError("SJCCubicBSplinef::DeleteControl - Index out of range");
 
  std::vector<SJCVector3f>::iterator deleteP = m_VControls.begin() + index;
  m_VControls.erase(deleteP);
} 

//****************************************************************************
//
// Move the entire path to certain distance 
//=========================================================================== 
void SJCCubicBSplinef::
MovePath(SJCVector3f& howfar)
//=========================================================================== 
{
  for (uint i = 0; i < m_VControls.size(); i++){
    m_VControls[i] = m_VControls[i] + howfar;
  }
  for (uint i = 0; i < m_VInstances.size(); i++){
    m_VInstances[i] = m_VInstances[i] + howfar;
  }
}

//****************************************************************************
//
// * Transform the entire path 
//=========================================================================== 
void SJCCubicBSplinef::
Transform(SJCQuaternionf& r, SJCVector3f& t)
//=========================================================================== 
{
  for (uint i = 0; i < m_VControls.size(); i++){
    m_VControls[i] = r * m_VControls[i] + t;
  }
  for (uint i = 0; i < m_VInstances.size(); i++){
    m_VInstances[i] = r * m_VInstances[i] +t;
  }
}



//************************************************************************ 
// 
// Evalute the curve velocity, Problem whether the refinement will reserve 
// The first and end points' position 
//=========================================================================== 
void SJCCubicBSplinef::
Refine(SJCCubicBSplinef &result)
//============================================================================ 
{ 
  // Loop variable 
  int i, j; 
  std::vector<SJCVector3f> old; 
  
  // Get the size of original control elements 
  int numInstances = result.m_VInstances.size(); 
  
  // Resize the old one 
  old.resize(numInstances); 

  // Copy the information in 
  for(i = 0; i < numInstances; i++) 
    old[i] = result.m_VInstances[i]; 

  // Calc new drawing points count
  if (m_bLoop)
    result.m_VInstances.resize(numInstances * 2);
  else
    result.m_VInstances.resize(numInstances * 2 - 3);
  
  int newNumInstances = result.m_VInstances.size(); 
  
  // i is index at new array 
  // j is index at old array
  for(i = 0, j = 0; i < newNumInstances; i+=2, j++){
    
    // These are the indices of the points to average
    SJCVector3f p0 = old[j % numInstances];
    SJCVector3f p1 = old[(j + 1) % numInstances];
    SJCVector3f p2 = old[(j + 2) % numInstances];
    
    result.m_VInstances[i] = 0.5 * (p0 + p1);
    if(i + 1 < newNumInstances)
      result.m_VInstances[i+1] = 0.125 * (p0 + p1 * 6.0 + p2);
  } // Duplicate the result
}

//*************************************************************************** 
// 
// Refine the std::vector until it reach the tolerance 
//============================================================================ 
void SJCCubicBSplinef::
RefineTolerance(SJCCubicBSplinef &result, const float tolerance)
//============================================================================ 
{
  Refine(result);
  while(! WithinTolerance(tolerance))
    result.Refine(result);
}

//************************************************************************ 
// 
// Check the tolerance P1, P2, P3 
// The distance square between  P2 to line P1P3 is smaller than tolerance 
//============================================================================ 
bool SJCCubicBSplinef::
WithinTolerance(const float tolerance)
//============================================================================ 
{ 
  // V12's projection point on V13
  SJCVector3f projectionPoint;
 
  // Point for the three points 
  SJCVector3f p1, p2, p3; 

  // The std::vector between p1 P2 and p1 p3, and p2 projection 
  SJCVector3f V12, V13, V2P; 
  
  float toleranceSquare = tolerance * tolerance; 
  
  // Total number to calculate
  int num = m_bLoop ? m_VInstances.size() : m_VInstances.size() - 2;
  
  // Go through all point in the curve 
  int numInstances = m_VInstances.size(); 
  for ( int i = 0 ; i < num ; i++ ) { 
    // Get the  control point in the curve
    p1 = m_VInstances[i % numInstances]; 
    p2 = m_VInstances[(i + 1) % numInstances]; 
    p3 = m_VInstances[(i + 2) % numInstances]; 
    
    // Calculate the std::vector 
    V12 = p2 - p1; 
    V13 = p3 - p1; 
    
    // Compute the V13 length square 
    float lengthSquare = V13 * V13; 
    if (lengthSquare == 0.0) 
      continue; 
    
    // V12 on V13's projection will be
    float ratio =  (V12 * V13) / lengthSquare; 
 
    projectionPoint = p1 + V13 * ratio; 
    
    V2P = projectionPoint - p2; 
    if( (V2P * V2P) > toleranceSquare) 
      return false; 
  } 
  
  return true;
}
 

//***************************************************************************
//
// * Using the frame position to find the least square fit curve 
// * numCPS: total number of control points
// * coeff:
// * framePos:
//=========================================================================== 
void SJCCubicBSplinef::
LeastSquareFit( std::vector<SJCVector3f>& framePos) 
//===========================================================================  
{ 
  float         coeffi, coeffj; 
  float         **coefficient; 
  std::vector<float> x; 
  std::vector<float> y;
  std::vector<float> z; 
  uint           numCPs = framePos.size() /m_ucControlPointsDensity;
  if(numCPs < (uint)m_ucNumMinControlPoints)
    numCPs = (uint)m_ucNumMinControlPoints;

  numCPs = 12;


  float increment = (float)(numCPs - 3) / (float)framePos.size();

  x.resize(numCPs);
  y.resize(numCPs); 
  z.resize(numCPs); 

  // Allocate the N by N coefficient array
  coefficient	 = new float*[numCPs]; 
  for(uint i =0; i < numCPs; i++) 
    coefficient[i] = new float[numCPs]; 

  // Initialize the array
  for(uint i =0; i < numCPs; i++) 
    for(uint j = 0; j < numCPs; j++) 
      coefficient[i][j] = 0; 
  
  for(uint i =0; i < numCPs; i++){ 
    x[i] = 0; 
    y[i] = 0;
    z[i] = 0; 
    
    for(uint k = 0; k < framePos.size(); k++){ 
      float coeff =  increment * (float)k;
      coeffi = BasicFunction(i, coeff); 

      x[i] += coeffi * framePos[k][0];
      y[i] += coeffi * framePos[k][1];
      z[i] += coeffi * framePos[k][2]; 
      
      for(uint j = 0; j < numCPs; j++){ 
	coeffj = BasicFunction(j, coeff); 
	coefficient[i][j] += coeffi * coeffj; 
      }// end of j 
    }// end of k 
  }// end of i 


  std::vector<float> diagonal; 
  std::vector<float> cx;
  std::vector<float> cy; 
  std::vector<float> cz; 
  diagonal.resize(numCPs);
  cx.resize(numCPs);
  cy.resize(numCPs);
  cz.resize(numCPs);
  for(uint i = 0; i < numCPs; i++){
    diagonal[i] = cx[i] = cy[i] = cz[i] = 0;
  }
 
  Cholesky(numCPs, coefficient, diagonal); 
 
  Solver(numCPs, coefficient, diagonal, x, cx)	; 
  Solver(numCPs, coefficient, diagonal, y, cy)	;
 
  if(m_b3D)
    Solver(numCPs, coefficient, diagonal, z, cz)	; 

 
  m_VControls.clear(); 
  m_VControls.resize(numCPs);
  m_VInstances.clear();
  m_VInstances.resize(numCPs);
 
  for(uint i = 0; i < numCPs; i++){ 
    m_VControls[i][0] = cx[i]; 
    m_VControls[i][1] = cy[i]; 
    if(m_b3D)
      m_VControls[i][2] = cz[i];
    else
      m_VControls[i][2] = 0;
    m_VInstances[i] = m_VControls[i];
  } 

   
  for(uint i =0; i < numCPs; i++) 
    delete [] coefficient[i]; 
  delete [] coefficient; 
} 


//***************************************************************************
//
// * Using the frame position to find the least square fit curve 
// * numCPS: total number of control points
// * coeff:
// * framePos:
//=========================================================================== 
void SJCCubicBSplinef::
LeastSquareFit( std::vector<SJCVector3f>& framePos, uint numCPs) 
//===========================================================================  
{ 
  float         coeffi, coeffj; 
  float         **coefficient; 
  std::vector<float> x; 
  std::vector<float> y;
  std::vector<float> z; 


  float increment = (float)(numCPs - 3) / (float)framePos.size();

  x.resize(numCPs);
  y.resize(numCPs); 
  z.resize(numCPs); 

  // Allocate the N by N coefficient array
  coefficient	 = new float*[numCPs]; 
  for(uint i =0; i < numCPs; i++) 
    coefficient[i] = new float[numCPs]; 

  // Initialize the array
  for(uint i =0; i < numCPs; i++) 
    for(uint j = 0; j < numCPs; j++) 
      coefficient[i][j] = 0; 
  
  for(uint i =0; i < numCPs; i++){ 
    x[i] = 0; 
    y[i] = 0;
    z[i] = 0; 
    
    for(uint k = 0; k < framePos.size(); k++){ 
      float coeff =  increment * (float)k;
      coeffi = BasicFunction(i, coeff); 

      x[i] += coeffi * framePos[k][0];
      y[i] += coeffi * framePos[k][1];
      z[i] += coeffi * framePos[k][2]; 
      
      for(uint j = 0; j < numCPs; j++){ 
	coeffj = BasicFunction(j, coeff); 
	coefficient[i][j] += coeffi * coeffj; 
      }// end of j 
    }// end of k 
  }// end of i 


  std::vector<float> diagonal; 
  std::vector<float> cx;
  std::vector<float> cy; 
  std::vector<float> cz; 
  diagonal.resize(numCPs);
  cx.resize(numCPs);
  cy.resize(numCPs);
  cz.resize(numCPs);
  for(uint i = 0; i < numCPs; i++){
    diagonal[i] = cx[i] = cy[i] = cz[i] = 0;
  }
 
  Cholesky(numCPs, coefficient, diagonal); 
 
  Solver(numCPs, coefficient, diagonal, x, cx)	; 
  Solver(numCPs, coefficient, diagonal, y, cy)	;
 
  if(m_b3D)
    Solver(numCPs, coefficient, diagonal, z, cz)	; 

 
  m_VControls.clear(); 
  m_VControls.resize(numCPs);
  m_VInstances.clear();
  m_VInstances.resize(numCPs);
 
  for(uint i = 0; i < numCPs; i++){ 
    m_VControls[i][0] = cx[i]; 
    m_VControls[i][1] = cy[i]; 
    if(m_b3D)
      m_VControls[i][2] = cz[i];
    else
      m_VControls[i][2] = 0;
    m_VInstances[i] = m_VControls[i];
  } 

   
  for(uint i =0; i < numCPs; i++) 
    delete [] coefficient[i]; 
  delete [] coefficient; 
} 
 
//***************************************************************************
//
// * Using the frame position to find the least square fit curve 
// * numCPS: total number of control points
// * framePos: the desired position for them to pass
//=========================================================================== 
void SJCCubicBSplinef::
LeastSquareFitWithLimitAtStartAndEnd(std::vector<SJCVector3f>& framePos,
				     uint numCPs) 
//===========================================================================  
{ 
  // The increment for each step
  float increment     = (float)(numCPs - 3) / (float)framePos.size();

  // Calcluate the interval where last control points is in effect
  float last_interval = (float)(numCPs - 4);

  // Compute the least square fit number
  uint  num_fitCPs    = numCPs - 2;
  uint  num_fitFrames = framePos.size() - 2;
  uint  last_frame    = framePos.size() - 1;


  // Create the least square metric
  std::vector<std::vector<float> > A(num_fitFrames);
  std::vector<float>          b_x(num_fitFrames);
  std::vector<float>          b_y(num_fitFrames);
  std::vector<float>          b_z(num_fitFrames);
 
  // Create the m by n matrix, where m is the number of fit frames, n is the
  // number of control points
  for(uint i = 0; i < num_fitFrames; i++){
    A[i].resize(num_fitCPs);
  }

  // Compute the cofficent A * C = b
  for(uint i = 0; i < num_fitFrames; i++) {

    // Get the coefficient for this frames
    float coeff =  increment * (float)(i + 1); // +1, we start from frame 1

    if(coeff < 1.0){ // for those affected by first control points
                     // We need to redistributed its coefficient to others

      // Compute the contribution of coefficient 0
      float coeff0 = BasicFunction(0, coeff);
      b_x[i] = framePos[i + 1][0] - framePos[0][0] * coeff0 * 6.0;
      b_y[i] = framePos[i + 1][1] - framePos[0][1] * coeff0 * 6.0;
      b_z[i] = framePos[i + 1][2] - framePos[0][2] * coeff0 * 6.0;

      // Compute the matrix coefficient
      for(uint j = 0; j < num_fitCPs; j++){
	if( j == 0) // Need to do redistributed to first, and second turn
	  A[i][j] = BasicFunction(j + 1, coeff) - 4.f * coeff0;
	else if (j == 1)
	  A[i][j] = BasicFunction(j + 1, coeff) - coeff0;
	else 
	  A[i][j] = BasicFunction(j + 1, coeff);
      } // end of for
    }// end of if

    if (coeff > last_interval){ // for those affected by last control
                                // We need to redistributed its 
                                // coefficient to others

      // Compute the coefficient
      float coeff_last = BasicFunction(numCPs - 1, coeff);
 
      // Redistributed
      b_x[i] = framePos[i + 1][0] - framePos[last_frame][0] * coeff_last * 6.0;
      b_y[i] = framePos[i + 1][1] - framePos[last_frame][1] * coeff_last * 6.0;
      b_z[i] = framePos[i + 1][2] - framePos[last_frame][2] * coeff_last * 6.0;

      for(uint j = 0; j < num_fitCPs; j++){
	if( j == num_fitCPs - 1) // need to redistributed for the last two term
	  A[i][j] = BasicFunction(j + 1, coeff) - 4.f * coeff_last;
	else if (j == num_fitCPs - 2)
	  A[i][j] = BasicFunction(j + 1, coeff) - coeff_last;
	else 
	  A[i][j] = BasicFunction(j + 1, coeff);
      } // end of for
    } // end of else
    else { // inbetween nothing affected
      b_x[i] = framePos[i + 1][0];
      b_y[i] = framePos[i + 1][1];
      b_z[i] = framePos[i + 1][2];

      for(uint j = 0; j < num_fitCPs; j++){
	A[i][j] = BasicFunction(j + 1, coeff);
      } // end of for 
    } // end of else
  }
  
  float         **coefficient; 

 std::vector<float> x(num_fitCPs); 
 std::vector<float> y(num_fitCPs);
 std::vector<float> z(num_fitCPs);
   
 // Allocate the N by N coefficient array
 coefficient	 = new float*[num_fitCPs]; 
 for(uint i = 0; i < num_fitCPs; i++) {
   x[i] = 0; 
   y[i] = 0;
   z[i] = 0; 

   coefficient[i] = new float[num_fitCPs]; 
   for(uint j = 0; j < num_fitCPs; j++) 
      coefficient[i][j] = 0; 
 }// end of for i
 
  
  // coeff = A^T * A
  // A^T * A * C = A^T * b
  for( uint i = 0; i < num_fitCPs; i++) {
    for( uint j = 0; j < num_fitFrames; j++) {
     
      // Compute the output  A^T(i, j) = A(j, i)
      x[i] += A[j][i] * b_x[j];
      y[i] += A[j][i] * b_y[j];
      z[i] += A[j][i] * b_z[j];
      
      // Compute A^T . A
      for(uint k = 0; k < num_fitCPs; k++){ 
	coefficient[i][k] += A[j][i] * A[j][k]; 
      }// end of for k
    } // end of for j
  } // end of for i


  //*********************************************************************
  // Solve it
  //*********************************************************************
  std::vector<float> diagonal; 
  std::vector<float> cx;
  std::vector<float> cy; 
  std::vector<float> cz; 

  diagonal.resize(num_fitCPs);

  cx.resize(num_fitCPs);
  cy.resize(num_fitCPs);
  cz.resize(num_fitCPs);

  for(uint i = 0; i < num_fitCPs; i++){
    diagonal[i] = cx[i] = cy[i] = cz[i] = 0;
  }
  
  // Cholesky decomposition
  Cholesky(num_fitCPs, coefficient, diagonal); 
 

  // Solve x, y, z
  Solver(num_fitCPs, coefficient, diagonal, x, cx)	; 
  Solver(num_fitCPs, coefficient, diagonal, y, cy)	;
  if(m_b3D)
    Solver(num_fitCPs, coefficient, diagonal, z, cz)	; 

  
  // Clear the control and instances 
  m_VControls.clear(); 
  m_VControls.resize(numCPs);
  m_VInstances.clear();
  m_VInstances.resize(numCPs);

 
  // For the first points
  m_VControls[0][0] = 6.f * framePos[0][0] - 4.f * cx[0] - cx[1];
  m_VControls[0][1] = 6.f * framePos[0][1] - 4.f * cy[0] - cy[1];
  if(m_b3D)
    m_VControls[0][2] = 6.f * framePos[0][2] - 4.f * cz[0] - cz[1];
  else
    m_VControls[0][2] = 0.f;
  m_VInstances[0] = m_VControls[0];


  // Copy All fit points 
  for(uint i = 0; i < num_fitCPs; i++){ 
    m_VControls[i + 1][0] = cx[i]; 
    m_VControls[i + 1][1] = cy[i]; 
    if(m_b3D)
      m_VControls[i + 1][2] = cz[i];
    else
      m_VControls[i + 1][2] = 0;
    m_VInstances[i + 1] = m_VControls[i + 1];
  } 

  // For the last points
  m_VControls[numCPs - 1][0] = 
    6.f * framePos[last_frame][0] - 4.f * cx[num_fitCPs - 1] - 
    cx[num_fitCPs - 2];
  m_VControls[numCPs - 1][1] = 
    6.f * framePos[last_frame][1] - 4.f * cy[num_fitCPs - 1] - 
    cy[num_fitCPs - 2];
  if(m_b3D)
    m_VControls[numCPs - 1][2] = 
      6.f * framePos[last_frame][2] - 4.f * cz[num_fitCPs - 1] - 
      cz[num_fitCPs - 2];
  else
    m_VControls[numCPs - 1][2] = 0.f;

  m_VInstances[numCPs - 1] = m_VControls[numCPs - 1];
   
  for(uint i = 0; i < num_fitCPs; i++) 
    delete [] coefficient[i]; 
  delete [] coefficient; 
 
  




} 
 
//***************************************************************************
//
// To get the basic coefficient for non-loop
//===========================================================================
float SJCCubicBSplinef::
BasicFunction(int pointIndex, float t)
//===========================================================================  
{ 
  int index; 
  
  float u; 
  float uSquare; 
  float uCube; 
  float basis; 
  
  index = (uint) floor(t); 
  
  if ((index > (int)m_VControls.size() - 4) && (!m_bLoop))  
    std::cout<<"something wrong in CubicBSpline::basicFunction"<< std::endl; 
  
  u       = t - index; 
  uSquare = u * u; 
  uCube   = u * uSquare; 
  
  //given an index, find it's A value 
  if(index == pointIndex) 
    basis = -uCube + 3.0 * uSquare - 3.0 * u + 1.0; 
  else if(index == pointIndex - 1) 
    basis = 3.0 * uCube - 6.0 * uSquare + 4.0; 
  else if(index == pointIndex - 2) 
    basis = -3.0 * uCube + 3.0 * uSquare + 3.0 * u + 1.0; 
  else if(index == pointIndex - 3) 
    basis = uCube; 
  else 
    basis = 0.0; 
  
  basis = basis / 6.0; 
  
  return basis; 
} 

//***************************************************************************
//
// Use Cholesky to solve function
//===========================================================================  
void SJCCubicBSplinef::
Cholesky(int num, float **decomp, std::vector<float>& diagonal) 
//===========================================================================  
{ 
 
  float sum; 
  int i, j, k; 
  for(i = 0; i < num; i++){ 
    for(j = i; j < num; j++){ 
      for(sum = decomp[i][j], k = i - 1; k >= 0; k--) 
	sum -= decomp[i][k] * decomp[j][k]; 
      if(i == j){ 
	if(sum <= 0.0){ 
	  std::cout<<"sum is less than 0 in cholesky"<< std::endl; 
	  return; 
	} 
	diagonal[i] = sqrt(sum); 
      } 
      else 
	decomp[j][i] = sum / diagonal[i]; 
    } 
  } 
} 
 
//*************************************************************************** 
//
// Solve the constraint question
//=========================================================================== 
void SJCCubicBSplinef::
Solver(int num, float **decomp, std::vector<float>& diagonal, 
       std::vector<float>& b, std::vector<float>& x) 
//=========================================================================== 
{

  float sum;
  int i, k;

 
  for(i = 0; i < num; i++){
    for( sum = b[i], k = i - 1; k >= 0; k--)
      sum -= decomp[i][k] * x[k];

     x[i] = sum / diagonal[i];
 
  }// end of for i
  for(i = num - 1; i >= 0; i--){
    for(sum = x[i], k = i + 1; k < num; k++)
      sum -= decomp[k][i] * x[k];
    x[i] = sum / diagonal[i];
  }// end of for i
} 

//****************************************************************************
//
// * Given an arbitrary point ("A"), returns the nearest point ("P") on
//   this path.  Also returns, via output arguments, the path tangent at
//   P and a measure of how far A is outside the Pathway's "tube".  Note
//   that a negative distance indicates A is inside the Pathway.
//=========================================================================== 
SJCVector3f SJCCubicBSplinef::
MapPointToPath (const SJCVector3f& point, SJCVector3f& tangent,
		float& distance_to_center)
//=========================================================================== 
{
  float max_coeff = m_VControls.size() - 3;
  distance_to_center = SJC_INFINITE;
  SJCVector3f nearP;
  float     nearC;
  // Go throught all sample point to find the nearest position
  for(float coeff = 0.f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    SJCVector3f temp;
    EvaluatePoint(coeff, temp);
    SJCVector3f difference = temp - point;
    float dist = difference.length();
    if(dist < distance_to_center){ // smaller distance found
      nearP = temp;
      nearC = coeff;
      distance_to_center = dist;
    } // end of if
  } // end of for
  EvaluateDerivative(nearC, tangent);
  return nearP;
}

//****************************************************************************
//
// * Given a distance along the path, convert it to a point on the path
//=========================================================================== 
SJCVector3f SJCCubicBSplinef::
MapPathDistanceToPoint (float pathDistance)
//=========================================================================== 
{
  float max_coeff = m_VControls.size() - 3;
  float accumulate = 0.f;

  SJCVector3f prev_p;
  SJCVector3f current_p;
  EvaluatePoint(0.f, prev_p);
  current_p = prev_p;
  
  // Go throught all sample point to find the nearest position
  for(float coeff = 0.1f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    EvaluatePoint(coeff, current_p);
    SJCVector3f difference = current_p - prev_p;
    float dist = difference.length();
    accumulate += dist;
    if(accumulate >= pathDistance){
      return current_p;
    }
    prev_p = current_p;
  }// end of for
  return current_p;
}

//****************************************************************************
//
// * Given a distance along the path, convert it to a point on the path
//=========================================================================== 
SJCVector3f SJCCubicBSplinef::
MapPathDistanceToDirection (float pathDistance)
//=========================================================================== 
{
  float max_coeff = m_VControls.size() - 3;
  float accumulate = 0.f;

  SJCVector3f prev_p;
  SJCVector3f current_p;
  EvaluatePoint(0.f, prev_p);
  current_p = prev_p;
  
  // Go throught all sample point to find the nearest position
  for(float coeff = 0.1f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    EvaluatePoint(coeff, current_p);
    SJCVector3f difference = current_p - prev_p;
    float dist = difference.length();
    accumulate += dist;

    if(accumulate >= pathDistance){
      if(dist > 0.00001)
	return difference.normal();
    }
    prev_p = current_p;
  }// end of for
  return EndDirection();
}
//****************************************************************************
//
// * Given a distance along the path, convert it to a point on the path
//=========================================================================== 
void SJCCubicBSplinef::
MapPathDistanceToPositionDirection(float pathDistance, SJCVector3f& pos,
				   SJCVector3f& dir)
//=========================================================================== 
{
  float max_coeff = m_VControls.size() - 3;
  float accumulate = 0.f;

  SJCVector3f prev_p;
  SJCVector3f current_p;
  EvaluatePoint(0.f, prev_p);
  current_p = prev_p;
  
  // Go throught all sample point to find the nearest position
  for(float coeff = 0.1f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    EvaluatePoint(coeff, current_p);
    SJCVector3f difference = current_p - prev_p;
    float dist = difference.length();
    accumulate += dist;
    if(accumulate >= pathDistance){
      pos = current_p;
      dir = difference.normal();
      return;
    }
    prev_p = current_p;
  }// end of for
  pos = current_p;
  dir = EndDirection();
  return;
}

//****************************************************************************
//
// * Given an arbitrary point, convert it to a distance_square along the path.
//=========================================================================== 
float SJCCubicBSplinef::
MapPointToPathDistance (const SJCVector3f& point, SJCVector3f& pathPoint,
			SJCVector3f& tangent, float& distance_to_center)
//=========================================================================== 
{
  float max_coeff = m_VControls.size() - 3;
  distance_to_center  = SJC_INFINITE;
  SJCVector3f nearP;
  float    nearC;

  // Go throught all sample point to find the nearest position
  for(float coeff = 0.f; coeff < max_coeff; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    SJCVector3f temp;
    EvaluatePoint(coeff, temp);
    SJCVector3f difference = temp - point;
    float dist = difference.length();
    if(dist < distance_to_center){ // smaller distance found
      nearP = temp;
      nearC = coeff;
      distance_to_center = dist;
    } // end of if
  } // end of for

  pathPoint  = nearP;
  EvaluateDerivative(nearC, tangent);

  float accumulate = 0.f;

  SJCVector3f prev_p;
  SJCVector3f current_p;
  EvaluatePoint(0.f, prev_p);

  // Go throught all sample point to find the nearest position
  for(float coeff = 0.1f; coeff <= nearC; coeff +=m_dcEvaluateIncrement){
    // Calcualte the point and the distance
    EvaluatePoint(coeff, current_p);
    SJCVector3f difference = current_p - prev_p;
    float dist = difference.length();
    accumulate += dist;
    prev_p = current_p;
    }// end of for
  return accumulate;
}


//****************************************************************************
//
// * Given an arbitrary point ("A"), returns the nearest point ("P") on
//   this path between distance start and end.  
//=========================================================================== 
SJCVector3f SJCCubicBSplinef::
MapPointToPathSection (const SJCVector3f& point, const float path_start,
		       const float path_end, const float step, 
		       float& best_path_length)
//=========================================================================== 
{
  SJCVector3f curr(0.f, 0.f, 0.f);
  float     min = 1e20;

  for(float path_dist = path_start; path_dist <= path_end; path_dist += step){
    SJCVector3f path_point = MapPathDistanceToPoint(path_dist);
    float dist_sqrt = (path_point - point).lengthSquare();
    if(dist_sqrt < min){
      min = dist_sqrt;
      curr = path_point;
      best_path_length = path_dist;
    } // end of if
  }// end of for


  return curr;
}

//************************************************************************ 
// 
// Create a path with length and projected to an arc with angle
// + angle means right turn 
// - angle means left turn
// * start at (0, 0), and center is on the y axis
//========================================================================== 
void SJCCubicBSplinef::
Create2DPath(const float angle, const float length)
//========================================================================== 
{
  Clear();
  m_bLoop = false;
  m_b3D   = false;
  const uint point_size = 60;

  std::vector<SJCVector3f> pathPoints(point_size);
  if(angle == 0.f){
    float increment = length / (float)point_size;
    for(uint i = 0; i < point_size; i++){
      pathPoints[i].set((float)i * increment, 0.f, 0.f);
    }
  }
  else if (angle > 0.f){
    float angle_radian = fabs(angle) * SJC_DEG_TO_RAD;
    float radius = length / angle_radian;
    float angle_inc = angle_radian / 60.f;
    for(uint i = 0; i < point_size; i++){
      float a = M_PI_2 - (float)i * angle_inc;
      float x = radius * cos(a);
      float y = radius * sin(a) - radius;
      pathPoints[i].set(x, y, 0.f);
    }
  }
  else {
    float angle_radian = fabs(angle) * SJC_DEG_TO_RAD;
    float radius = length / angle_radian;
    float angle_inc = angle_radian / 60.f;

    for(uint i = 0; i < point_size; i++){
      float a = - M_PI_2 + (float)i * angle_inc;
      float x = radius * cos(a);
      float y = radius * sin(a) + radius;

      pathPoints[i].set(x, y, 0.f);
    }// end of for
  }// end of else
  LeastSquareFit(pathPoints);
  RefineTolerance(*this, 0.1f);
  CalBound();
}


//************************************************************************ 
// 
// Create a path with length and projected to an arc with angle
// + angle means right turn 
// - angle means left turn
// * start at (0, 0), and center is on the y axis
//========================================================================== 
void SJCCubicBSplinef::
Create3DPath(const float theta, const float phi, const float length)
//========================================================================== 
{
  Clear();

  m_bLoop = false;
  m_b3D   = true;

  const uint point_size = 60;

  std::vector<SJCVector3f> pathPoints(point_size);
  if(theta == 0.f){
    float increment = length / (float)point_size;
    for(uint i = 0; i < point_size; i++){
      pathPoints[i].set((float)i * increment, 0.f, 0.f);
    }
  }
  else {
    float theta_radian = fabs(theta) * SJC_DEG_TO_RAD;
    float radius       = length / theta_radian;
    float theta_inc    = theta_radian / 60.f;
    float cos_phi      = cos( phi * SJC_DEG_TO_RAD);
    float sin_phi      = sin( phi * SJC_DEG_TO_RAD);

    for(uint i = 0; i < point_size; i++){
      float a = M_PI_2 - (float)i * theta_inc;
      float x = radius * cos(a);
      
      float yz = radius * sin(a) - radius;
      float y  = yz * cos_phi;
      float z  = yz * sin_phi;
      pathPoints[i].set(x, y, z);
    }
  }

  LeastSquareFit(pathPoints);
  RefineTolerance(*this, 0.1f);
  CalBound();
}


//************************************************************************** 
//
// * transform from xyz to zxy configuration
//=========================================================================== 
void SJCCubicBSplinef::
XYZToZXY(void)
//=========================================================================== 
{
  for(uint i = 0; i < m_VControls.size(); i++){
    SJCVector3f xyz = m_VControls[i];
    m_VControls[i].set(xyz.y(), xyz.z(), xyz.x());
  }

  for(uint i = 0; i < m_VInstances.size(); i++){
    SJCVector3f xyz = m_VInstances[i];
    m_VInstances[i].set(xyz.y(), xyz.z(), xyz.x());
  }
  
  SJCVector3f bound  = m_vMinBounding; 
  m_vMinBounding.set(bound.y(), bound.z(), bound.x());

  bound = m_vMaxBounding;
  m_vMaxBounding.set(bound.y(), bound.z(), bound.x());

}

//*****************************************************************************
//
// * output operator
//============================================================================
std::ostream& operator<<(std::ostream &o, const SJCCubicBSplinef& g)
//============================================================================
{
  if(g.m_bLoop)
    o << "LoopFlag true\n";
  else
    o << "LoopFlag false\n";
 
  if(g.m_b3D)
    o << "IN3DFlag true\n";
  else
    o << "IN3DFlag false\n";

  o << "NumControls " << g.m_VControls.size() << " { ";
  for(uint i = 0; i < g.m_VControls.size(); i++){
    o << g.m_VControls[i] << " ";
  }
  o << " } ";

  return o;
  
}// end of output operator


//*****************************************************************************
//
// * Create the exactly fit b-spline
//=============================================================================
void SJCCubicBSplinef::
ExactFitSplineToData(const std::vector<SJCVector3f>& data) 
//=============================================================================
{
  // Make the number of control points equal to the amount of data + 2, so that
  // we can have an exact fit to the data
  int npts  = m_VControls.size() - 2;
  int dsize = data.size();
  if (npts > dsize) {
    m_VControls.resize(dsize + 2);
  } 
  else if (npts < dsize) {
    for (int i = npts; i < dsize; i++) {
      AppendControl(SJCVector3f(static_cast<float>(i), 0, 0));
    } // end of for
  }
  FitSplineToData(data);
}

//****************************************************************************
//
// * Fit the spline to the data
//============================================================================
void  SJCCubicBSplinef::
FitSplineToData(const std::vector<SJCVector3f>& data) 
//============================================================================
{
  // Set up the control points
  int npts = m_VControls.size() - 2;
  int i, j;
  int dsize = data.size();

  // Error message
  if (data.size() == 0)
    SJCError("Whoa!  Moron!  You didn't give me any data");

  // Only one data in the spline
  if (data.size() == 1) {
    m_VControls.resize(0);
    for (int i = 0; i < 4; i++) {
      AppendControl(data[0]);
    }
    return;
  }
  
  if (dsize < npts) {
    if (dsize >= 2) {
      SJCWarning("Amount of data too small for number of control points; "
		 "deleting some control points");
      m_VControls.resize(dsize + 2);
      npts = dsize;
    }
    else {
      SJCWarning("Fewer than 2 data points; aborting");
      return;
    }
  } // end of data
  
  std::vector<SJCVector3f> points;
  if (dsize == npts) {
    points = data;
  } 
  else {
    int incr = dsize / npts;
    for (i = 0; i < dsize; i += incr) {
      points.push_back(data[i]);
    }
  }
  
  // We already can choose values for the end control points
  // (we're underconstrained, and all that)
  // ---- first ----
  SetControl(0, points[0] + (points[0] - points[1]));
  points[0] = points[0] - GetControl(0) / 6.f;
  
  // ---- last ----
  SetControl(m_VControls.size() - 1,
	     points[points.size() - 1] +
	     (points[points.size() - 1] - points[points.size() - 2]));
  points[points.size() - 1] = (points[points.size()-1] - 
			       GetControl(m_VControls.size()-1) /
			       6.f);
   
  // Initialize arrays for the solver
  float **a = new float * [npts];
  float **b = new float * [npts];
  for (i = 0; i < npts; i++) {
    a[i] = new float [npts]; 
    b[i] = new float [3];
  }

  // Build arrays for the solver
  for (i = 0; i < npts; i++) {
    // Build a...
    for (j = 0; j < npts; j++) {
      if (j == i-1 || j == i+1) {
	a[i][j] = 1.f / 6.f;
      }
      else if (j == i)          {
	a[i][j] = 4.f / 6.f;
      }
      else  {
	a[i][j] = 0;
      }
     }
    // Build b...
    b[i][0] = points[i].x(); 
    b[i][1] = points[i].y(); 
    b[i][2] = points[i].z();
  }
   
  // Solve
  if (gaussj(a, npts, b, 3)) {
    // Set control points
    for (i = 0; i < npts; i++) {
      m_VControls[i+1] = SJCVector3f(b[i][0], b[i][1], b[i][2]);
    } // end of for 
  } // end of if
  else {
    SJCWarning("Solver failed to find a solution; not fitting BSpline");
  } // end of else 
  
  // delete the arrays used in the solver
  for (i = 0; i < npts; i++) {
    delete [] a[i];
    delete [] b[i];
  }

  delete [] a;
  delete [] b;
}

#define SWAP(a,b) {temp=(a); (a) = (b); (b) = temp;}
//****************************************************************************
//
// * THE SOLVER from Numerical Recipies in C
//============================================================================
bool SJCCubicBSplinef::
gaussj(float **a, int n, float **b, int m) 
//============================================================================
{
  int *indxc, *indxr, *ipiv = NULL;
  indxc = new int[n];
  indxr = new int[n];
  ipiv  = new int[n];
  int i, icol = 0, irow = 0, j, k, l, ll;
  float big, dum, pivinv, temp;
  
  for (i = 0; i < n; i++) {
    indxc[i] = indxr[i] = i; 
    ipiv[i] = 0;
  }
  
  for (i = 0; i < n; i++) {
    big = 0.f;
    for (j = 0; j < n; j++) {
      if (ipiv[j] != 1) {
	for (k = 0; k < n; k++) {
	  if (ipiv[k] == 0) {
	    if (fabs(a[j][k]) >= big) {
	      big = fabsf(a[j][k]);
	      irow = j;
	      icol = k;
	    }
	  } else if (ipiv[k] > 1) {
	    SJCWarning("Singular Matrix-1");
	    return false;
	  } // end of else if
	} // end of for k
      } // end of if ipiv
    } // end of for j
    ++(ipiv[icol]);
    if (irow != icol) {
      for (l = 0; l < n; l++) {SWAP(a[irow][l],a[icol][l]);}
      for (l = 0; l < m; l++) {SWAP(b[irow][l],b[icol][l]);}
    }
    indxr[i]=irow;
    indxc[i]=icol;
    if (a[icol][icol] == 0) {
      SJCWarning("Singular Matrix-2");
      return false;
    }
    pivinv=1.0f/a[icol][icol];
    a[icol][icol] = 1.f;

    for (l = 0; l < n; l++) {
      a[icol][l] *= pivinv;
    }

    for (l = 0; l < m; l++) {
      b[icol][l] *= pivinv;
    }
    for (ll = 0; ll < n; ll++) {
      if (ll != icol) {
	dum = a[ll][icol];
	a[ll][icol] = 0.0f;
	for (l = 0; l < n; l++) {
	  a[ll][l] -= a[icol][l] * dum;
	}
	for (l = 0; l < m; l++) {
	  b[ll][l] -= b[icol][l] * dum;
	}
      } // end of for if
    }  // end of for ll
  } // end of for i
  
  for (l = n-1; l >= 0; l--) {
    if (indxr[l] != indxc[l]) {
      for (k = 0; k < n; k++) {SWAP(a[k][indxr[l]],a[k][indxc[l]]);}
    }
  }
  delete [] indxc;
  delete [] indxr;
  delete [] ipiv;
  return true;
}

