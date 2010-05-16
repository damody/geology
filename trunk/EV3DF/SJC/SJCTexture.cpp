/************************************************************************
     File:         SJCTexture.cpp 

     Author:     
                  Eric McDaniel, chate@cs.wisc.edu

     Written:     Summer 2002

     Comment:     Definitions of methods of the CTexture class.   
                  See header for documentation.
     
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include <SJCTexture.h>

//*****************************************************************************
//
// * Constructor.
//=============================================================================
SJCTexture::
SJCTexture(uint uTextureId, const std::string& sName) 
         : m_uTextureId(uTextureId), m_sName(sName)
//=============================================================================
{}// SJCTexture


//*****************************************************************************
//
// * Destructor.
//=============================================================================
SJCTexture::~SJCTexture(void)
{}// ~SJCTexture

//*****************************************************************************
//
// * Get the texture id of this texture object.
//=============================================================================
uint SJCTexture::
GetTextureId(void) const
//=============================================================================
{
   return m_uTextureId;
}// GetTextureId


//*****************************************************************************
//
// * Get the name of this texture object.
//=============================================================================
std::string SJCTexture::
GetName(void) const
//=============================================================================
{
  return m_sName;
}// GetName
