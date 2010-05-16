/************************************************************************
     File:        SJCJointTransform.h

     Author:
                  Rachel Heck and Lucas Kovar

     Modifier:
                  Yu-Chi Lai, yu-chi@cs.wisc.edu

     Comment:     
                  A Joint rigid transform matrix class

                  A rigid transform is a linear transformation consisting of a 
                  3D rotation followed by a 3D translation.  
                  This class provides a representation for a rigid transform 
                  along with many  methods for creating rigid transforms, 
                  methods for doing basic arithmetic on rigid transforms, 
                  and several helper functions for performing simple 
                  operations on multiple rigid transforms.    

     class CJointTransform


      Constructor:     
                  4 : transform, flag to reverse the order, key informations

     Interfaces:
                  1. WorldBound: The bound in world space

     Funcitons:
                  1. ObjectBound : THe bound in object space
                  2. WorldBound: The bound in world space
                  3. CanIntersect : Whether the shape can be intersected
                  4. Refine : Refine the object if it is not intersectable

 ************************************************************************/

#ifndef __SJC_JOINT_TRANSFORM_H__
#define __SJC_JOINT_TRANSFORM_H__

#include "SJC.h"

#include "SJCQuaternion.h"
#include "SJCVector3.h"

class SJCDLL SJCJointTransform
{
 public:
  
  // Create an Identity SJCJointTransform 
  SJCJointTransform(void);

		
  // Create a new SJCJointTransform that is an exact copy of an 
  // existing SJCJointTransform
  SJCJointTransform( const SJCJointTransform& otherTransform );

		
  //
  // The rotation can be specified as 4 doubles, w,x,y,z, as a 
  // SJCQuaterniond, or as an array whose first four elements are 
  // (w,x,y,z)
  //
  // The translation can be specified as 3 doubles, x,y,z, as a 
  // SJCVector3d, or as an array whose first three elements are (x,y,z)
  
  // Create a new SJCJointTransform with the specified rotation and 
  // translation.
  SJCJointTransform( const SJCQuaterniond& rotation, 
		     const SJCVector3d& translation );

  // Create a new SJCJointTransform with the specified rotation and 
  // translation.
  // @param rotation A rotation specified as an array whose first four
  //                   elements are the coordinates of a quaternion (w,x,y,z)
  //  @param translation A translation specified as an array whose first three
  //                   elements are the coordinates of a translation (x,y,z)
  SJCJointTransform( const double* rotation, const double* translation );

  
  // Create a new SJCJointTransform with the specified rotation (using 
  // quaternion coordinates w, x_rotation, y_rotation, and z_rotation) and
  // translation (using coordinates x_translation, y_translation, and 
  // z_translation).
  SJCJointTransform( double w, double x_rotation, double y_rotation, 
		     double z_rotation, double x_translation, 
		     double y_translation, double z_translation );	

  //***********************************************************************
  //
  // Creating a rigid transform
  // --------------------------------------------------------------
  // A rigid transform can be converted to a new rigid transform 
  // by specifying new values for the rotation and translation.
  //
  //***********************************************************************
  
  //  Convert this rigid transform into the identity 
  void Identity(void);

  
  // Convert this SJCJointTransform so that it is an exact copy of an 
  // existing SJCJointTransform
  void Copy( const SJCJointTransform& otherTransform );
		
  
  // Convert this SJCJointTransform so that it is an exact copy of an 
  // existing SJCJointTransform
  SJCJointTransform& operator=( const SJCJointTransform& otherTransform );

  // Converts this SJCJointTransform so that it has the specified 
  // rotation and translation.
  void Set( const SJCQuaterniond& rotation, const SJCVector3d& translation );

  // Converts this SJCJointTransform so that it has the specified 
  // rotation and translation.
  // @param rotation A rotation specified as an array whose first four 
  //        elements are the coordinates of a quaternion (w,x,y,z)
  // @param translation A translation specified as an array whose first three 
  //        elements are the coordinates of a translation (x,y,z)
  void Set( const double* rotation, const double* translation );
		
  // Converts this SJCJointTransform so that it has the specified 
  // rotation (using quaternion coordinates
  // w, x_rotation, y_rotation, and z_rotation) and translation 
  // (using coordinates x_translation, y_translation, and z_translation).
  void Set( double w, double x_rotation, double y_rotation, double z_rotation,
	    double x_translation, double y_translation, 
	    double z_translation );	

  // Converts this SJCJointTransform into one corresponding
  // to a rotation of [rotation] centered about the point [rotateCenter].
  void SetToRotationAboutPoint( const SJCQuaterniond &rotation, 
				const SJCVector3d &rotateCenter );

  // Converts this SJCJointTransform into one corresponding
  // to a rotation of [rotation] centered about the point [rotateCenter].  
  // @param rotation A rotation specified as an array whose first four 
  //        elements are the coordinates of a quaternion (w,x,y,z)
  // @param rotateCenter The rotation center specified as an array whose 
  //        first three elements are the point's coordinates (x,y,z)
  void SetToRotationAboutPoint( const double *rotation, 
				const double *rotateCenter );

  // Returns a SJCJointTransform that corresponds
  // to a rotation of [rotation] centered about the point [rotateCenter].
  static SJCJointTransform 
    CreateRotationAboutPoint( const SJCQuaterniond &rotation, 
			      const SJCVector3d &rotateCenter );

  // Returns a SJCJointTransform that corresponds
  // to a rotation of [rotation] centered about the point [rotateCenter].  
  // @param rotation A rotation specified as an array whose first four
  //        elements are the coordinates of a quaternion (w,x,y,z)
  // @param rotateCenter The rotation center specified as an array whose first 
  //        three elements are  the point's coordinates (x,y,z)
  static SJCJointTransform 
    CreateRotationAboutPoint( const double *rotation, 
			      const double *rotateCenter );

		
  // Multiplies this SJCJointTransform by [otherTransform] 
  // in place.  If the second argument is true, 
  // [this]*[otherTransform] is computed; otherwise 
  // [otherTransform]*[this] is computed. 
  void Multiply( const SJCJointTransform& otherTransform, bool rightMultiply );

  // Calculates [this]*[otherTransform] in place. 
  SJCJointTransform& operator*=( const SJCJointTransform& otherTransform );

  // Returns a rigid transform that is equal to [this]*[otherTransform]. 
  SJCJointTransform operator*( const SJCJointTransform& otherTransform ) const;

  // Rotates this SJCJointTransform by [rotation] in place. 
  void Rotate( const SJCQuaterniond& rotation );
		
  //  Translates this SJCJointTransform by [translation]  in place. 
  void Translate( const SJCVector3d& translation );

  // Converts this SJCJointTransform into its inverse 
  void Inverse(void);

  // Returns the inverse of this SJCJointTransform 
  SJCJointTransform CopyAndInvert(void) const;

  // Apply a SJCJointTransform to a 3D point (specified by [x, y, z]), 
  // replacing that point with the result.
  void TransformInPlace( double& x, double& y, double& z ) const;

  // Apply a SJCJointTransform to a 3D point (specified by the array 
  // [x, y, z]), replacing that point with the result.
  void TransformInPlace( double* point ) const;
  
  // Apply a SJCJointTransform to a 3D point, replacing that 
  // point with the result.
  void TransformInPlace( SJCVector3d& point ) const;

  // Returns the result of applying this transform to [point]. 
  SJCVector3d operator*( const SJCVector3d& point ) const;

  // Modifies the array of length 3 so that it contains the result of ([x,y,z])
  // of applying this transform to [point]. 
  void Transform( const double* point, double* result ) const;

  // Returns the result of applying this transform to [point]. 
  SJCVector3d Transform( const SJCVector3d& point ) const;

  // Returns the current value of the x component of the 3D translation	
  double GetX() const;

  // Returns the current value of the y component of the 3D translation 
  double GetY() const;

  // Returns the current value of the z component of the 3D translation	
  double GetZ() const;

  // Returns a SJCQuaterniond representing the current value of the 3D 
  // rotation 
  const SJCQuaterniond& GetRotation(void) const;

  // Returns a SJCVector3d object representing all 3 components of the current
  // 3D translation.
  const SJCVector3d& GetTranslation(void) const;

  // Sets the current value of the x component of the 3D translation to [x] 
  void SetX( double x );

  // Sets the current value of the y component of the 3D translation to [y]
  void SetY( double y );

  // Sets the current value of the z component of the 3D translation to [z] 
  void SetZ( double z );

  // Sets the current value of the 3D rotation to [rotation] 
  void SetRotation( const SJCQuaterniond& rotation );

  // Sets the current value of the 3D translation to [translation] 
  void SetTranslation( const SJCVector3d& translation );

		
  // Modifies the given sequence array [transforms] of numTrans rigid 
  // transforms so that element i is on the same side of the 
  // 4-sphere as element i-1. That is, this eliminates numerical 
  // discontinuities that stem from antipode equivalence.
  static void SelectAntipodesForContinuity( SJCJointTransform* transforms, 
					    int numTrans );

  // Modifies the given std::vector of rigid 
  // transforms so that element i is on the same side of the 
  // 4-sphere as element i-1. That is, this eliminates numerical 
  // discontinuities that stem from antipode equivalence.
  static void 
    SelectAntipodesForContinuity( std::vector<SJCJointTransform>& transforms );

  // Computes the average rigid transform of the given array of 
  // numTrans transforms given the weights for each rigid transform in 
  // [weights] and returns the result.  The input 
  // sequence is first run through selectAntipodesForContinuity to 
  // eliminate spurious numerical differences.
  static SJCJointTransform Average( SJCJointTransform* transforms, 
				    int numTrans, const double* weights );

  // Computes the average rigid transform of the given std::vector of 
  // transforms given the weights for each rigid transform in [weights] 
  // and returns the result.  The input 
  // sequence is first run through selectAntipodesForContinuity to 
  // eliminate spurious numerical differences.
  static SJCJointTransform Average( std::vector<SJCJointTransform>& transforms,
				    const std::vector<double>& weights );

  // Performs linear interpolation on t1 and t2.  If u = 0, this 
  // returns t1; if u = 1 this returns t2.
  static SJCJointTransform Interpolate( const SJCJointTransform& t1, 
					const SJCJointTransform& t2, 
					double u );

  // Returns a string representation of this transform, which can be
  // useful for debugging.  This representation is of the form "Quat: 
  // [quaternion coordinates] Vec: [translation coordinates]".
  std::string ToString(void) const;

  friend std::ostream& operator<<(std::ostream&o, const SJCJointTransform &v) {
    o << "Qaut: " << v.m_qRotation << " Vec: " << v.m_vTranslation << "\n";
    return o;
  }

 private:
  SJCQuaterniond m_qRotation; // 3D rotation
  
  SJCVector3d    m_vTranslation; // 3D translation
};

#endif
