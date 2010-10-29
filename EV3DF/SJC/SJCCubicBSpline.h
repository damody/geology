/************************************************************************
   Main File :     Main.cpp

 
   File:           CubicBSpline.h

    
   Author:         Yu-Chi Lai, yu-chi@cs.wisc.edu

                
   Comment:        BSplineCurve which is used for creating the path

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
                  36. MapPointToPathDistance: Given an arbitrary point, 
                      convert it to a distance along the path.
                  37. MapPointToPathSection: Given an arbitrary point ("A"), 
                      returns the nearest point ("P") on this path in the 
                      section with path length between start and end
                  38. Create2DPath: create a 2D circular path
                  39. Create3DPath: create a 3D circular path
                  40. ConstantY: set up the constant value of Y
                  41. XYZToZXY: XYZ to ZXY
                  42. 

    
   Compiler:       g++

 
   Platform:       Linux
*************************************************************************/    

#ifndef _SJC_CUBICBSPLINE_H_
#define _SJC_CUBICBSPLINE_H_


// The global definition information
#include "SJC.h"
#include "SJCConstants.h"

// General C library
#include <stdio.h>
#include <math.h> 
#include <stdlib.h>

// General C++ library
#include <iostream> 
#include <vector> 
#include <string>

// My own library
#include "SJCVector3.h" 
#include "SJCQuaternion.h"

#include "SJCErrorHandling.h"

class SJCDLL SJCCubicBSplined
{
 protected:
  // The minimum number of control points for least square fitting algorithm
  static const uint   m_ucNumMinControlPoints;  
  // The average number of vertices we should put a control point
  static const uint   m_ucControlPointsDensity;
  // The increment evaluation in control coefficient
  static const double  m_dcEvaluateIncrement;

 protected: 
  // The name of the path
  std::string m_sLabel; 

  // Whether the spline line is a loop
  bool	m_bLoop;           
  // Apply the fitting algorithm to 2D or 3D
  bool  m_b3D;             

  // The control point to generate the entire spline 
  std::vector<SJCVector3d> m_VControls;
  // The point used to draw the spline line 
  std::vector<SJCVector3d> m_VInstances;	

  // The bounding minimum point
  SJCVector3d m_vMinBounding; 
  // The bounding maximum point
  SJCVector3d m_vMaxBounding;
  
  
 public:
	 
  //********************************************************
  // Constructors and destructor
  //******************************************************** 
  SJCCubicBSplined(const bool loop_var = true, const bool b3d = false)
    { m_bLoop = loop_var; m_b3D = b3d;} 
  
  // Initilize spline with size and loop vars and control points
  SJCCubicBSplined(const bool loop_var, const bool b3d, int size,
		SJCVector3d *control_points);
  
  // Initilize spline with dim and loop vars and control points 
  SJCCubicBSplined(const bool loop_var, const bool b3d, 
		std::vector<SJCVector3d>& control_points); 
		
  // Destructor
  virtual ~SJCCubicBSplined(void){ }

  //*************************************************************
  // Operator
  //************************************************************
  // Copy Operator
  SJCCubicBSplined & operator=(const SJCCubicBSplined &);

  //**************************************************************
  // Access function
  //**************************************************************
  // Set and access the label
  void Label(std::string& label) { m_sLabel = label; }
  std::string Label(void){ return m_sLabel; }

  void Set(bool bLoop, bool b3d){ m_bLoop = bLoop; m_b3D = b3d;}

  // Calculate the bound of the path
  void CalBound(void);
  // The minimum bound of the curve 
  SJCVector3d MinBound(void) { return m_vMinBounding; }
  // The maximum bound of the curve
  SJCVector3d MaxBound(void) { return m_vMaxBounding; }

  // Get the start point and end point derivative
  SJCVector3d StartPoint(void);
  SJCVector3d StartDirection(void);
  
  // Get the end point and end derivative
  SJCVector3d EndPoint(void);
  SJCVector3d EndDirection(void);

  // Get the control point numbers 
  unsigned short NumControls(void) {return m_VControls.size();} 

  // Get the number of instance points 
  unsigned short NumInstances(void) {return m_VInstances.size();} 

  // Get the control point in index 
  void GetControl(const unsigned short, SJCVector3d&); 
  SJCVector3d GetControl(const unsigned short); 

  // Get the control point out 
  const std::vector<SJCVector3d>& GetControls(void) { return m_VControls; }  
 
  // Get the control point in index 
  void GetInstance(const unsigned short, SJCVector3d&); 

  // Get the drawing point out 
  const std::vector<SJCVector3d>& GetInstances(void){ return m_VInstances; } 
  
  // Get sample points from the path
  void GetSamples(uint numSamples, std::vector<SJCVector3d>& samples);

  //Evaluate curve at a parameter and return result in SJCVector3d 
  //throws exception if parameters out of range, unless looping. 
  SJCVector3d EvaluatePoint(const double, SJCVector3d&); 
  
  //Evaluate derivitive at a given parameter and return result in SJCVector3d 
  //throws exception if parameters out of range, unless looping. 
  SJCVector3d EvaluateDerivative(const double, SJCVector3d&); 

  // The total length of the path 
  double Length(); 
   
  //****************************************************************** 
  // Manipulate the control points of the curve 
  //****************************************************************** 
  // Clear the point in the path
  void   Clear(void);

  // Set control point 
  void SetControl(const unsigned short, const SJCVector3d&); 
  void SetControls(std::vector<SJCVector3d>& controls);
  
  // Set the x, y, or z to certain value 
  void SetControlComponent(int direciton, double value); 
  
  // Add a control point to the end of the spline 
  void AppendControl(const SJCVector3d&); 
 
  // Insert a control point. Throws an exception if the pos is beyond the end. 
  void InsertControl(const unsigned short, const SJCVector3d&); 
  
  // Remove a control point, throws exception if invalid pos 
  void DeleteControl(const unsigned short); 

  //****************************************************************** 
  // Manupulate the entire curve: rigid transform 
  //****************************************************************** 
  // Move the entire path
  void   MovePath(SJCVector3d&);

  // Transform the entire path
  void   Transform(SJCQuaterniond& r, SJCVector3d& t);
    
  //******************************************************************* 
  // Refine curve 
  //******************************************************************* 
  //refine curve 
  void Refine(SJCCubicBSplined &result); 

  // Check whether the new formed spline within certain threshold 
  void RefineTolerance(SJCCubicBSplined &result, const double tolerance); 

  // Check whether the control point is withing tolerance 
  bool WithinTolerance(const double); 


  //******************************************************************* 
  // Find the fitting curve 
  //******************************************************************* 
  // Use the least square error algorithm to find the fitting curve
  void LeastSquareFit(std::vector<SJCVector3d>&);

  // Use the least square error algorithm to find the fitting curve
  void LeastSquareFit(std::vector<SJCVector3d>&, uint num_controls);

  // Function to calculate the basic function of B-Spline
  double BasicFunction(int, double);

  // Find the coefficient of
  double FindCoeff(int, double t);

  // Decompose the coefficient array into UV
  void   Cholesky(int, double **, std::vector<double>&);

  // Solve the leave square
  void   Solver(int, double **, std::vector<double>&, std::vector<double>&, 
	        std::vector<double>&);


  // Use the least square error algorithm to find the fitting curve
  void   LeastSquareFitWithLimitAtStartAndEnd(std::vector<SJCVector3d>&,
					      uint num_controls);

  //******************************************************************* 
  // Mapping from real world to the path
  //******************************************************************* 
  // Given an arbitrary point ("A"), returns the nearest point ("P") on
  // this path.  Also returns, via output arguments, the path tangent at
  // P and a measure of how far A is outside the Pathway's "tube".  Note
  // that a negative distance indicates A is inside the Pathway.
  SJCVector3d MapPointToPath (const SJCVector3d& point,
			    SJCVector3d& tangent,
			    double& distance_to_center);
  
  // Given a distance along the path, convert it to a point on the path
  SJCVector3d MapPathDistanceToPoint (double pathDistance);

  // Given a distance along the path, convert it to a direction on the path
  SJCVector3d MapPathDistanceToDirection (double pathDistance);

  // Given a distance along the path, convert it to a pos and dir on the path
  void MapPathDistanceToPositionDirection (double pathDistance, 
					   SJCVector3d& pos,
					   SJCVector3d& dir);
  
  // Given an arbitrary point, convert it to a distance along the path.
  double MapPointToPathDistance (const SJCVector3d& point, 
				 SJCVector3d& pathPoint,
				 SJCVector3d& tangent, 
				 double& distance_to_center);

  // * Given an arbitrary point ("A"), returns the nearest point ("P") on
  //   this path in the section with path length between start and end
  SJCVector3d MapPointToPathSection(const SJCVector3d& point, 
				    const double path_start, 
				    const double path_end,
				    const double step, 
				    double& best_path_length);

  //***********************************************************************
  // Misc function
  //***********************************************************************
  // Create a path
  void   Create2DPath(const double angle, const double length);

  void   Create3DPath(const double theta, const double phi, 
		      const double length);

  void   ConstantY(const double constant_y){
    for(uint i = 0; i < m_VControls.size(); i++){
      m_VControls[i].y(constant_y);
    }
    for(uint i = 0; i < m_VInstances.size(); i++){
      m_VInstances[i].y(constant_y);
    }
  }

  // XYZ to ZXY
  void   XYZToZXY(void);

  // For arc length
  SJCVector3d PointAtArcLength(const double arc_length);

  //************************************************************************
  // From Eric method
  //************************************************************************
  void      FitSplineToData(const std::vector<SJCVector3d>& data);
  void      ExactFitSplineToData(const std::vector<SJCVector3d>& data);

 private:
  bool   gaussj(double **a, int n, double **b, int m);


 public: 


  // Output operator
  friend std::ostream& operator<<(std::ostream &o, const SJCCubicBSplined& g);



};

class SJCDLL SJCCubicBSplinef
{
 protected:
  // The minimum number of control points for least square fitting algorithm
  static const uint   m_ucNumMinControlPoints;  
  // The average number of vertices we should put a control point
  static const uint   m_ucControlPointsDensity;
  // The increment evaluation in control coefficient
  static const float  m_dcEvaluateIncrement;

 protected: 
  // The name of the path
	 std::string m_sLabel; 
  // Whether the spline line is a loop
  bool	m_bLoop;           
  // Apply the fitting algorithm to 2D or 3D
  bool  m_b3D;             

  // The control point to generate the entire spline 
  std::vector<SJCVector3f> m_VControls;
  // The point used to draw the spline line 
  std::vector<SJCVector3f> m_VInstances;	

  // The bounding minimum point
  SJCVector3f m_vMinBounding; 
  // The bounding maximum point
  SJCVector3f m_vMaxBounding;
  
  
 public:
	 
  //********************************************************
  // Constructors and destructor
  //******************************************************** 
  SJCCubicBSplinef(const bool loop_var = true, const bool b3d = false)
    { m_bLoop = loop_var; m_b3D = b3d;} 
  
  // Initilize spline with size and loop vars and control points
  SJCCubicBSplinef(const bool loop_var, const bool b3d, int size,
		SJCVector3f *control_points);
  
  // Initilize spline with dim and loop vars and control points 
  SJCCubicBSplinef(const bool loop_var, const bool b3d, 
		std::vector<SJCVector3f>& control_points); 
		
  // Destructor
  ~SJCCubicBSplinef(void){ }

  //*************************************************************
  // Operator
  //************************************************************
  // Copy Operator
  SJCCubicBSplinef & operator=(const SJCCubicBSplinef &);

  //**************************************************************
  // Access function
  //**************************************************************
  // Set and access the label
  void Label(std::string& label) { m_sLabel = label; }
  std::string Label(void){ return m_sLabel; }

  void Set(bool bLoop, bool b3d){ m_bLoop = bLoop; m_b3D = b3d;}

  // Calculate the bound of the path
  void CalBound(void);
  // The minimum bound of the curve 
  SJCVector3f MinBound(void) { return m_vMinBounding; }
  // The maximum bound of the curve
  SJCVector3f MaxBound(void) { return m_vMaxBounding; }

  // Get the start point and end point derivative
  SJCVector3f StartPoint(void);
  SJCVector3f StartDirection(void);
  
  // Get the end point and end derivative
  SJCVector3f EndPoint(void);
  SJCVector3f EndDirection(void);

  // Get the control point numbers 
  unsigned short NumControls(void) {return m_VControls.size();} 

  // Get the number of instance points 
  unsigned short NumInstances(void) {return m_VInstances.size();} 

  // Get the control point in index 
  void GetControl(const unsigned short, SJCVector3f&); 
  SJCVector3f GetControl(const unsigned short); 

  // Get the control point out 
  const std::vector<SJCVector3f>& GetControls(void) { return m_VControls; }  
 
  // Get the control point in index 
  void GetInstance(const unsigned short, SJCVector3f&); 

  // Get the drawing point out 
  const std::vector<SJCVector3f>& GetInstances(void){ return m_VInstances; } 
  
  // Get sample points from the path
  void GetSamples(uint numSamples, std::vector<SJCVector3f>& samples);

  //Evaluate curve at a parameter and return result in SJCVector3f 
  //throws exception if parameters out of range, unless looping. 
  SJCVector3f EvaluatePoint(const float, SJCVector3f&); 
  
  //Evaluate derivitive at a given parameter and return result in SJCVector3f 
  //throws exception if parameters out of range, unless looping. 
  SJCVector3f EvaluateDerivative(const float, SJCVector3f&); 

  // The total length of the path 
  float Length(); 
   
  //****************************************************************** 
  // Manipulate the control points of the curve 
  //****************************************************************** 
  // Clear the point in the path
  void   Clear(void);

  // Set control point 
  void SetControl(const unsigned short, const SJCVector3f&); 
  void SetControls(std::vector<SJCVector3f>& controls);
  
  // Set the x, y, or z to certain value 
  void SetControlComponent(int direciton, float value); 
  
  // Add a control point to the end of the spline 
  void AppendControl(const SJCVector3f&); 
 
  // Insert a control point. Throws an exception if the pos is beyond the end. 
  void InsertControl(const unsigned short, const SJCVector3f&); 
  
  // Remove a control point, throws exception if invalid pos 
  void DeleteControl(const unsigned short); 

  //****************************************************************** 
  // Manupulate the entire curve: rigid transform 
  //****************************************************************** 
  // Move the entire path
  void   MovePath(SJCVector3f&);

  // Transform the entire path
  void   Transform(SJCQuaternionf& r, SJCVector3f& t);

    
  //******************************************************************* 
  // Refine curve 
  //******************************************************************* 
  //refine curve 
  void Refine(SJCCubicBSplinef &result); 

  // Check whether the new formed spline within certain threshold 
  void RefineTolerance(SJCCubicBSplinef &result, const float tolerance); 

  // Check whether the control point is withing tolerance 
  bool WithinTolerance(const float); 


  //******************************************************************* 
  // Find the fitting curve 
  //******************************************************************* 
  // Use the least square error algorithm to find the fitting curve
  void LeastSquareFit(std::vector<SJCVector3f>&);

  // Use the least square error algorithm to find the fitting curve
  void LeastSquareFit(std::vector<SJCVector3f>&, uint num_controls);

  // Function to calculate the basic function of B-Spline
  float BasicFunction(int, float);

  // Find the coefficient of
  float FindCoeff(int, float t);

  // Decompose the coefficient array into UV
  void   Cholesky(int, float **, std::vector<float>&);

  // Solve the leave square
  void   Solver(int, float **, std::vector<float>&, std::vector<float>&, 
	        std::vector<float>&);


  //  bool   gaussj(float **a, int n, float **b, int m);
  //  void   FitSplineToData(const std::vector<SJCVector3f>& data, 
  //			 const uint num_controls) ;

  // Use the least square error algorithm to find the fitting curve
  void   LeastSquareFitWithLimitAtStartAndEnd(std::vector<SJCVector3f>&,
					      uint num_controls);

  //******************************************************************* 
  // Mapping from real world to the path
  //******************************************************************* 
  // Given an arbitrary point ("A"), returns the nearest point ("P") on
  // this path.  Also returns, via output arguments, the path tangent at
  // P and a measure of how far A is outside the Pathway's "tube".  Note
  // that a negative distance indicates A is inside the Pathway.
  SJCVector3f MapPointToPath (const SJCVector3f& point,
			    SJCVector3f& tangent,
			    float& distance_to_center);
  
  // Given a distance along the path, convert it to a point on the path
  SJCVector3f MapPathDistanceToPoint (float pathDistance);

  // Given a distance along the path, convert it to a direction on the path
  SJCVector3f MapPathDistanceToDirection (float pathDistance);

  // Given a distance along the path, convert it to a pos and dir on the path
  void MapPathDistanceToPositionDirection (float pathDistance, 
					   SJCVector3f& pos,
					   SJCVector3f& dir);
  
  // Given an arbitrary point, convert it to a distance along the path.
  float MapPointToPathDistance (const SJCVector3f& point, 
				 SJCVector3f& pathPoint,
				 SJCVector3f& tangent, 
				 float& distance_to_center);

  // * Given an arbitrary point ("A"), returns the nearest point ("P") on
  //   this path in the section with path length between start and end
  SJCVector3f MapPointToPathSection(const SJCVector3f& point, 
				    const float path_start, 
				    const float path_end,
				    const float step, 
				    float& best_path_length);

  //***********************************************************************
  // Misc function
  //***********************************************************************
  // Create a path
  void   Create2DPath(const float angle, const float length);

  void   Create3DPath(const float theta, const float phi, 
		      const float length);

  void   ConstantY(const float constant_y){
    for(uint i = 0; i < m_VControls.size(); i++){
      m_VControls[i].y(constant_y);
    }
    for(uint i = 0; i < m_VInstances.size(); i++){
      m_VInstances[i].y(constant_y);
    }
  }

  // XYZ to ZXY
  void   XYZToZXY(void);

  //************************************************************************
  // From Eric method
  //************************************************************************
  void      FitSplineToData(const std::vector<SJCVector3f>& data);
  void      ExactFitSplineToData(const std::vector<SJCVector3f>& data);

 private:
  bool   gaussj(float **a, int n, float **b, int m);

 public: 


  // Output operator
  friend std::ostream& operator<<(std::ostream &o, const SJCCubicBSplinef& g);



};

#endif
