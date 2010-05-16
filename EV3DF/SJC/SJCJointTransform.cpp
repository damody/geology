#include "SJCJointTransform.h"

/********** SJCJointTransform Constructor(s)/Deconstructor(s) **********/
// Also see inlined methods in header

/********** SJCJointTransform Inlined Methods **********/

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform::
SJCJointTransform(void)
//============================================================================
{
  m_qRotation.identity();
  m_vTranslation.set(0., 0., 0.);
}
//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform::
SJCJointTransform( const SJCQuaterniond& rotation,
		   const SJCVector3d& translation )
//============================================================================
{
  Set( rotation, translation );
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform::
SJCJointTransform( double w, double x_rotation, double y_rotation, 
		   double z_rotation, double x_translation, 
		   double y_translation, double z_translation )
//============================================================================
{
  Set( w, x_rotation, y_rotation, z_rotation, x_translation, y_translation,
       z_translation);
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform::
SJCJointTransform( const double* rotation, const double* translation )
//============================================================================
{
  Set( rotation, translation );
}
//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform::
SJCJointTransform( const SJCJointTransform& otherTransform )
//============================================================================
{
  Copy( otherTransform );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Identity(void)
//============================================================================
{
  m_qRotation.identity();
  m_vTranslation.set(0., 0., 0.);
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Set( const SJCQuaterniond& rotation, const SJCVector3d& translation )
//============================================================================
{
  m_qRotation    = rotation;
  m_vTranslation = translation;
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Set( double w, double x_rotation, double y_rotation, 
     double z_rotation, double x_translation, double y_translation,
     double z_translation )
//============================================================================
{
  m_qRotation.set( w, x_rotation, y_rotation, z_rotation );
  m_vTranslation.set( x_translation, y_translation, z_translation );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Set( const double* rotation, const double* translation )
//============================================================================
{
  m_qRotation.set( rotation[0], rotation[1], rotation[2], rotation[3] );
  m_vTranslation.set( translation[0], translation[1], translation[2] );
}

//****************************************************************************
//
// * 
//============================================================================
double SJCJointTransform::
GetX(void) const
//============================================================================
{
  return( m_vTranslation.x() );
}

//****************************************************************************
//
// * 
//============================================================================
double SJCJointTransform::
GetY(void) const
//============================================================================
{
  return( m_vTranslation.y() );
}

//****************************************************************************
//
// * 
//============================================================================
double SJCJointTransform::
GetZ(void) const
//============================================================================
{
  return( m_vTranslation.z() );
}

//****************************************************************************
//
// * 
//============================================================================
const SJCVector3d& SJCJointTransform::
GetTranslation(void) const
//============================================================================
{
  return( m_vTranslation );
}

//****************************************************************************
//
// * 
//============================================================================
const SJCQuaterniond& SJCJointTransform::
GetRotation(void) const
//============================================================================
{
  return( m_qRotation );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
SetX(double x)
//============================================================================
{
  m_vTranslation.x( x );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
SetY(double y)
//============================================================================
{
  m_vTranslation.y( y );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
SetZ(double z)
//============================================================================
{
  m_vTranslation.z( z );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
SetTranslation( const SJCVector3d& translation )
//============================================================================
{
  m_vTranslation.set( translation.x(), translation.y(), translation.z() );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
SetRotation( const SJCQuaterniond& rotation )
//============================================================================
{
  m_qRotation = rotation;
}



//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Copy( const SJCJointTransform& otherTransform )
//============================================================================
{
  Set( otherTransform.m_qRotation,otherTransform.m_vTranslation );
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform& SJCJointTransform::
operator=( const SJCJointTransform& otherTransform )
//============================================================================
{
  if( this != &otherTransform )	{
    Copy( otherTransform );
  }
  return( *this );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
SetToRotationAboutPoint( const SJCQuaterniond &rotation,
			 const SJCVector3d &rotateCenter ) 
//============================================================================
{
  m_qRotation    = rotation;
  m_vTranslation = rotateCenter - rotation * rotateCenter;
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
SetToRotationAboutPoint( const double *rotation, const double *rotateCenter ) 
//============================================================================
{
  SetToRotationAboutPoint(SJCQuaterniond( rotation[0], rotation[1], 
					  rotation[2], rotation[3] ), 
			  SJCVector3d( rotateCenter[0], rotateCenter[1], 
				       rotateCenter[2] ) );
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform SJCJointTransform::
CreateRotationAboutPoint( const SJCQuaterniond &rotation,
			  const SJCVector3d &rotateCenter ) 
//============================================================================
{
  SJCJointTransform result;
  result.SetToRotationAboutPoint( rotation, rotateCenter );

  return( result );
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform SJCJointTransform::
CreateRotationAboutPoint( const double *rotation, const double *rotateCenter ) 
//============================================================================
{
  SJCJointTransform result;
  result.SetToRotationAboutPoint( rotation, rotateCenter );

  return( result );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Multiply( const SJCJointTransform& otherTransform, bool rightMultiply )
//============================================================================
{
  SJCJointTransform a = *this;
  SJCJointTransform b = otherTransform;

  if( !rightMultiply ) {
    a = b;
    b = *this;
  }

  m_qRotation    = a.m_qRotation * b.m_qRotation;
  m_vTranslation = a.m_vTranslation + a.m_qRotation * b.m_vTranslation;
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Rotate( const SJCQuaterniond& rotation )
//============================================================================
{
  Multiply( SJCJointTransform(rotation, SJCVector3d(0., 0., 0.)), false );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Translate( const SJCVector3d& translation )
//============================================================================
{
  m_vTranslation += translation;
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform& SJCJointTransform::
operator*=( const SJCJointTransform& otherTransform )
//============================================================================
{
  Multiply( otherTransform, true );
  return( *this );
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform SJCJointTransform::
operator*( const SJCJointTransform& otherTransform ) const
//============================================================================
{
  SJCJointTransform result = SJCJointTransform( *this );
  result.Multiply( otherTransform, true );
  return( result );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
TransformInPlace( double& x, double& y, double& z ) const
//============================================================================
{
  SJCVector3d pt = SJCVector3d( x, y, z );
  pt = m_qRotation * pt; 
  
  x = pt.x() + GetX();
  y = pt.y() + GetY();
  z = pt.z() + GetZ();
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
TransformInPlace( double* point ) const
//============================================================================
{
  TransformInPlace( point[0], point[1], point[2] );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
TransformInPlace( SJCVector3d& point ) const
//============================================================================
{
  TransformInPlace( point[0], point[1], point[2] );
}

//****************************************************************************
//
// * 
//============================================================================
SJCVector3d SJCJointTransform::
operator*( const SJCVector3d& point ) const
//============================================================================
{
  SJCVector3d rotated = m_qRotation * point;

  rotated.set(rotated.x() + GetX(), 
	      rotated.y() + GetY(), 
	      rotated.z() + GetZ());

  return( rotated );
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Transform( const double* point, double* result ) const
//============================================================================
{
  result[0] = point[0];
  result[1] = point[1];
  result[2] = point[2];
  
  TransformInPlace( result );
}

//****************************************************************************
//
// * 
//============================================================================
SJCVector3d SJCJointTransform::
Transform( const SJCVector3d& point ) const
//============================================================================
{
  SJCVector3d result( point );

  TransformInPlace( result );
  
  return( result );
}


//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Inverse(void)
//============================================================================
{
  SJCQuaterniond invertquat = m_qRotation.inverse();
  SJCVector3d inverttrans   = -1.0 * (invertquat * m_vTranslation);

  SetRotation( invertquat );
  SetTranslation( inverttrans );
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform SJCJointTransform::
CopyAndInvert(void) const
//============================================================================
{
  SJCJointTransform copy( *this );
  copy.Inverse();
  return copy ;
}

/*
//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
Filter(const double* signal, int numPts, const double* kernel, 
       int kernelWidth, double* result)
//============================================================================
{
  int lowLim = -(kernelWidth - 1)/2;
  int highLim = kernelWidth - 1 + lowLim;
  int signalIdx;
  
  for(int i = 0; i < numPts; i++)	{
    result[i] = 0;
    for(int j = i + lowLim; j <= i + highLim; j++)		{
      if(j < 0)	{
	signalIdx = 0;
      } 
      else if(j > numPts - 1)			{
	signalIdx = numPts - 1;
      }
      else	{
	signalIdx = j;
      }
      
      result[i] += kernel[j - (i + lowLim)]*signal[signalIdx];
    }
  }
}
*/

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
SelectAntipodesForContinuity( SJCJointTransform* transforms, int numTrans )
//============================================================================
{
  for( int i = 1; i < numTrans; i++ )	{
    if(transforms[i].m_qRotation.dotProduct(transforms[i-1].m_qRotation)<0){
      transforms[i].m_qRotation *= -1;
    }
  }
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
SelectAntipodesForContinuity( std::vector<SJCJointTransform>& transforms )
{
  SelectAntipodesForContinuity( &(transforms[0]), (int) transforms.size() );
}

/*

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
FilterTransforms( SJCJointTransform* transforms, int numTrans, 
		  const double *kernel, int kernelWidth, 
		  SJCJointTransform* result )
//============================================================================
{
  // Filter the Rotation
  
  selectAntipodesForContinuity( transforms, numTrans );

  SJCQuaterniond* quats = new SJCQuaterniond[numTrans];
  SJCQuaterniond* filteredquats = new SJCQuaterniond[numTrans];
  for(int i = 0; i < numTrans; i++)	{
    quats[i] = transforms[i].m_qRotation;
  }
  SJCQuaterniond::FilterRotations(quats,numTrans,kernel,kernelWidth,
				  filteredquats);
  for(int i = 0; i < numTrans; i++)	{
    result[i].SetRotation(filteredquats[i]);
  }
  delete[] quats;
  delete[] filteredquats;
	
  // Filter the Translation
  double* unfiltered = new double[numTrans];
  double* filtered = new double[numTrans];
  
  // Filter in x
  for(int i = 0; i < numTrans; i++)	{
    unfiltered[i] = transforms[i].getX();
  }
  filter(unfiltered, numTrans, kernel, kernelWidth, filtered);
  for(int i = 0; i < numTrans; i++)    {
    result[i].setX(filtered[i]);
  }
  
  // Filter in y
  for(int i = 0; i < numTrans; i++)	{
    unfiltered[i] = transforms[i].getY();
  }
  filter(unfiltered, numTrans, kernel, kernelWidth, filtered);
  for(int i = 0; i < numTrans; i++)	{
    result[i].setY(filtered[i]);
  }
  
  // Filter in z
  for(int i = 0; i < numTrans; i++)    {
    unfiltered[i] = transforms[i].getZ();
  }
  filter(unfiltered, numTrans, kernel, kernelWidth, filtered);
  for(int i = 0; i < numTrans; i++)    {
      result[i].setZ(filtered[i]);
    }
  
  delete [] filtered;
  delete [] unfiltered;
}

//****************************************************************************
//
// * 
//============================================================================
void SJCJointTransform::
FilterTransforms(std::vector<SJCJointTransform>& transforms, 
		 const std::vector<double>& kernel, 
		 std::vector<SJCJointTransform>& result)
//============================================================================
{
  FilterTransforms(&(transforms[0]), (int) transforms.size(), &(kernel[0]), 
		   (int) kernel.size(), &(result[0]));
}
*/

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform SJCJointTransform::
Average( SJCJointTransform* transforms, int numTrans, const double* weights )
//============================================================================
{
  SJCJointTransform result;
  result.m_qRotation.set(0., 0., 0., 0.);

  // Get the transform for putting other tranfroms in the reference frame of 
  // the first

  SJCJointTransform firstTransform = SJCJointTransform( transforms[0] );
  SJCJointTransform firstTransformInverse = transforms[0].CopyAndInvert();
		
  for( int i = 0; i < numTrans; i++ )	{
    transforms[i].Multiply( firstTransformInverse, false );
  }

  SelectAntipodesForContinuity( transforms, numTrans );

  for( int i = 0; i < numTrans; i++ )	{
    result.SetRotation( result.GetRotation() + 
			transforms[i].GetRotation() * weights[i] );
    result.SetTranslation( result.GetTranslation() + 
			   transforms[i].GetTranslation() * weights[i]);
  }

  for(int i = 0; i < numTrans; i++) {
    transforms[i].Multiply( firstTransform, false );
  }
  
  result.Multiply( firstTransform, false );

  return( result );
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform SJCJointTransform::
Average( std::vector<SJCJointTransform>& transforms, 
	 const std::vector<double>& weights )
//============================================================================
{
  return( Average(&(transforms[0]), (int) transforms.size(),
		  &(weights[0])) );
}

//****************************************************************************
//
// * 
//============================================================================
SJCJointTransform SJCJointTransform::
Interpolate( const SJCJointTransform& t1, const SJCJointTransform& t2, 
	     double u )
//============================================================================
{
  SJCJointTransform result;

  // Compute the slerp of two rotation
  SJCQuaterniond re;
  SJCQuaterniond::slerp(t1.GetRotation(), t2.GetRotation(), u, re) ;
  result.SetRotation(re);


  // Compute the interpolation
  result.SetX( (1-u)*t1.m_vTranslation.x() + u*t2.m_vTranslation.x() );
  result.SetY( (1-u)*t1.m_vTranslation.y() + u*t2.m_vTranslation.y() );
  result.SetZ( (1-u)*t1.m_vTranslation.z() + u*t2.m_vTranslation.z() );

  return( result );
}

//****************************************************************************
//
// * 
//============================================================================
std::string SJCJointTransform::
ToString(void) const
//============================================================================
{
  std::ostringstream stringStream;
  stringStream << "Quat: " << m_qRotation << " Vec: " 
	       << m_vTranslation;
  return( stringStream.str() );
}
