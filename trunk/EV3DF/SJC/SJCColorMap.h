/************************************************************************
     Main File:

     File:        SJCColorMap.h

     Author:      
                  Steven Chenney, schenney@cs.wisc.edu
                  Yu-Chi Lai, yu-chi@cs.wisc.edu
 
     Comment:     Set up the color information
	
     Contructor:
		  0 paras: identity 
     Functions: 
                  1. Register(uint, SimVector): register a new color
                  2. Register(uint, Real*): register a new color
                  3. SetGLColor: set up OpenGL color
                  4. Size: the size of the current table
   
     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#ifndef SJCCOLORMAP_H_
#define SJCCOLORMAP_H_

// Define the global variables
#include <SJC/SJC.h>

// C libary
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>

// C++ library
#include <map>

// Fltk library
#include <Fl/gl.h>


#include <SJCVector3.h>

using namespace std;

class SJCColorMap {
 public:
  // define the color
  enum EColorName{
    SJC_WHITE,
    SJC_DARK_RED,
    SJC_DARK_GREEN,
    SJC_DARK_BLUE,
    SJC_ORANGE,
    SJC_SKY_BLUE,
    SJC_PURPLE,
    SJC_GRASS_GREEN,
    SJC_PEACH_RED,
    SJC_SKY_GREEN,
    SJC_LIGHT_YELLOW,
    SJC_LIGHT_PINK,
    SJC_LIGHT_CYAN,
    SJC_SKIN,
    SJC_LIGHT_GREEN,
    SJC_LIGHT_BLUE,
    SJC_GRAY,
    SJC_RED,
    SJC_GREEN,
    SJC_BLUE,
    SJC_YELLOW,
    SJC_CYAN,
    SJC_PINK,
    SJC_BLACK,
    NUM_COLORS
  }; // end of Enum color

  // Define the map
  typedef map<uint, SJCVector3f>   ColorMap; 
  typedef pair<uint, SJCVector3f>  ColorMapValue;

 private:
  static   float    m_cBasic_color[][3]; // global color 
  static   ColorMap m_ColorTable;             // Color map

 public:
  // Constructor and destructor
  SJCColorMap(void);
  ~SJCColorMap(void){}

  // Register the color
  bool Register(uint i, SJCVector3f& color);
  bool Register(uint i, float* color);

  // Set up OpenGL color
  void SetGLColor(uint i);

  // Size of the table
  uint Size(void) { return m_ColorTable.size(); }
};


#endif

