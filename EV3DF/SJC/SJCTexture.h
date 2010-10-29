/************************************************************************
     File:        SJCTexture.h

     Author:     
                  Eric McDaniel, chate@cs.wisc.edu

     Written:     Summer 2002

     Comment:     Data structure for the .CSM file

     Functions:   
                  1. Constructor and destructor
                  2. GetTextureId: get the texture id in OpenGL
                  3. GetName:  Get the name of this texture object.
 
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#ifndef _SJCTEXTURE_H
#define _SJCTEXTURE_H

#include <string>

class SJCTexture
{
  // interface
 public:
  // Constructor.
  SJCTexture(uint uTextureId, const std::string& sName);

  // Destructor.
  ~SJCTexture(void);
  
  // Get the texture id of this texture object.
  uint GetTextureId(void) const;
  
  // Get the name of this texture object.
  std::string GetName(void) const;
  
 // implementation
 private:
  uint           m_uTextureId;   // The texture ID in open gl
  std::string    m_sName;        // The file name of the texture
};// SJCTexture

#endif // _CTEXTURE
