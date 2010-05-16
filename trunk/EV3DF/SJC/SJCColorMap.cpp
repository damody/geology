/************************************************************************
     Main File:


     File:        SJCColorMap.cpp


     Author:      
                  Steven Chenney, schenney@cs.wisc.edu
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
 
     Comment:     Set up the color information
	
     Contructor:
		  0 paras: identity 
     Functions: 
                  1. Register(uint, SJCVector3f): register a new color
                  2. Register(uint, Real*): register a new color
                  3. SetGLColor: set up OpenGL color
                  4. Size: the size of the current table
	
   	
     Compiler:    g++


     Platform:    Linux
*************************************************************************/

#include "SJCColorMap.h"


float SJCColorMap::m_cBasic_color[][3] = { 
  {1.0f, 1.0f, 1.0f},   // White
  {0.5f, 0.0f, 0.0f},   // Dark red
  {0.0f, 0.5f, 0.0f},   // Dark green
  {0.0f, 0.0f, 0.5f},   // Dark blue
  {0.5f, 0.25f, 0.0f},  // 
  {0.0f, 0.25f, 0.5f},  // 
  {0.25f, 0.0f, 0.5f},  //
  {0.25f, 0.5f, 0.0f},   // 
  {0.5f, 0.0f, 0.25f},   // 
  {0.0f, 0.5f, 0.25f},   // 
 
  {1.0f, 0.5f, 0.0f},   // Orange
  {0.0f, 0.5f, 1.0f},   // Sky blue
  {0.5f, 0.0f, 1.0f},   // Purple
  {0.5f, 1.0f, 0.0f},   // Grass green
  {1.0f, 0.0f, 0.5f},   // Peach red
  {0.0f, 1.0f, 0.5f},   // Sky green
  {1.0f, 1.0f, 0.5f},   // Light yellow
  {1.0f, 0.5f, 1.0f},   // Light pink
  {0.5f, 1.0f, 1.0f},   // Light cyan
  {1.0f, 0.5f, 0.5f},   // Skin
  {0.5f, 1.0f, 0.5f},   // Light green
  {0.5f, 0.5f, 1.0f},   // light blue
  {0.5f, 0.5f, 0.5f},   // gray
  {1.0f, 0.0f, 0.0f},   // Red
  {0.0f, 1.0f, 0.0f},   // Green
  {0.0f, 0.0f, 1.0f},   // Blue
  {1.0f, 1.0f, 0.0f},   // Yellow
  {0.0f, 1.0f, 1.0f},   // Cyan
  {1.0f, 0.0f, 1.0f},   // Pink
  {0.0f, 0.0f, 0.0f}    // black
};

SJCColorMap::ColorMap SJCColorMap::m_ColorTable;

//*************************************************************************
//
// Constructor 
//=========================================================================
SJCColorMap::
SJCColorMap(void)
//=========================================================================
{
  for(uint i = 0; i < NUM_COLORS; i++){
    SJCVector3f color(m_cBasic_color[i]);
    Register(i, color);
  }
}

//*************************************************************************
//
// Register a color
//=========================================================================
bool SJCColorMap::
Register(uint index, SJCVector3f& color)
//=========================================================================
{
  if( m_ColorTable.find(index) == m_ColorTable.end()){
    m_ColorTable.insert(ColorMapValue(index, color));
    return true;
  }
  else{
    SJCWarning("The register color index already exist");
    return false;
  }
}

//*************************************************************************
//
// Register a color
//=========================================================================
bool SJCColorMap::
Register(uint index, float* color)
//=========================================================================
{
  SJCVector3f insert_c(color);
  if( m_ColorTable.find(index) == m_ColorTable.end()){
    m_ColorTable.insert(ColorMapValue(index, insert_c));
    return true;
  }
  else{
    SJCWarning("The register color index already exist");
    return false;
  }
}

//*************************************************************************
//
// Set the glcolor
//=========================================================================
void SJCColorMap::
SetGLColor(uint index)
//=========================================================================
{
  ColorMap::iterator itr = m_ColorTable.find(index);
  if( itr == m_ColorTable.end())
    return;

  SJCVector3f& color = itr->second;
  glColor3f(color.x(), color.y(), color.z());
  
}
