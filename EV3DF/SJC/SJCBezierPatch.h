/************************************************************************
     Main File:

     File:        SJCBezierPatch.h

     Author:     
                  Steven Chenney, schenney@cs.wisc.edu
     Modifier:     
                  Yu-Chi Lai, yu-chi@cs.wisc.edu

     Comment:     Using the bezier patch algorithm
                  16 control points form the bezier patch

                  *   *   *   *
                  *   *   *   *
                  *   *   *   * 
                  *   *   *   *
                  Refinement is to refine s direction and then t direction

     Constructors:
                  1. 0 : the default contructor
                  2. 2 : Initialize with the dimension and the control points
                  3. 3 : Initialize with the dimension and u, v and control
                         points
                   
     Functions: 
                 1. = : Assign operator which copy the parameter of random
                 2. D : Query the dimension.
                 3. C : Query the (m, n)th control point
                 4. Set_Control: Change the (m, n)th control point 
                 5. Evaluate_Point: Evaluate the curve at a parameter value and
                    copy the result into the given array.
                 6. Evaluate_Dx_Ds, Evaluate_Dx_Dt: Evaluate the derivative at 
                    a parameter value and copy the result into the given array.
                 7. Refine, Refine_S, Refine_T
                    Refine the curve one level, putting the result into the 
                    given curves.
                 8. Within_Tol_S, Within_Tol_T: The s, t direction must within 
                    the error limit  
************************************************************************/
 
#ifndef _SJCBEZIERPATCH_H_
#define _SJCBEZIERPATCH_H_

#include <SJC/SJC.h>
#include <stdio.h>
#include <utility>

class SJCBezierPatch {

 public:
  // Default constructor
  SJCBezierPatch(void);
  
  // Initializes with the given dimension and control points.
  SJCBezierPatch(const unsigned short, float*const [4][4]);
  
  // Constructor that takes 16 points on the patch. 
  SJCBezierPatch(const unsigned short,
		 float u[16], float v[16], float *p[16]);
  
  // Destructor.
  ~SJCBezierPatch(void);
  
  // Copy operator.
  SJCBezierPatch& operator=(const SJCBezierPatch&);
  
  // Query the dimension. 
  unsigned short	D(void) { return d; };
  
  // Query a control point, putting the value into the given array.
  // Throws an exception if the index is out of range. 
  void		C(const unsigned short, const unsigned short, float*);
  
  // Change a control point at the given position.
  // Will throw an exception if the position is out of range. 
  void    Set_Control(const float*,
		      const unsigned short, const unsigned short);
  
  // Evaluate the curve at a parameter value and copy the result into
  // the given array. Throws an exception if the parameter is out of
  // range, unless told to wrap. 
  void    Evaluate_Point(const float, const float, float*);
  
  // Evaluate the derivative at a parameter value and copy the result into
  // the given array. Throws an exception if the parameter is out of
  // range, unless told to wrap. 
  void    Evaluate_Dx_Ds(const float, const float, float*);
  void    Evaluate_Dx_Dt(const float, const float, float*);
  
  // Refine the curve one level, putting the result into the given curves.
  void    Refine(SJCBezierPatch&, SJCBezierPatch&,
		 SJCBezierPatch&, SJCBezierPatch&);
  void    Refine_S(SJCBezierPatch&, SJCBezierPatch&);
  void    Refine_T(SJCBezierPatch&, SJCBezierPatch&);
  
  // The s, t direction must within the error limit  
  bool    Within_Tol_S(const float);
  bool    Within_Tol_T(const float);
  
 private:
  // Copy the control points
  void    Copy_Controls(float*const[4][4]);
  // Delete the control points
  void    Delete_Controls(void);
  // Evaluate the basis function
  void    Evaluate_Basis(const float u, float vals[4]);
 private:
  unsigned short  d;		  // The dimension of the patch. 
  float   	  *c_pts[4][4];   // The control points. 
};


#endif

