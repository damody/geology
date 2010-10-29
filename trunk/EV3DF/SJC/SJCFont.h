/************************************************************************
     Main File:


     File:        SJCFont.h


     Author:      
                  Steven Chenney, schenney@cs.wisc.edu
                  Yu-Chi Lai, yu-chi@cs.wisc.edu

 
     Comment:     Set up the font for printing character in opengl window
	
     Functions:
                  1. Constructor and destructor
                  2. PrintString: set up the message and print it on screen

     Compiler:    g++


     Platform:    Linux
*************************************************************************/

#ifndef SJCFONT_H_
#define SJCFONT_H_

#include <SJC/SJC.h>

#include <stdio.h>
#include <string.h>
#include <map>

#include <Fl/gl.h>


class SJCFont {
 private:
  static const  GLubyte m_uSpace[];        // The space ' '
  static const  GLubyte m_uLetters [][13]; // The letter 'A' to 'Z', '0' to '9'
  static GLuint m_uFontOffset;             //
  static bool   m_bConstruct;              // The flag to indicate whether
                                           // We have construct the window
  static uint   m_uCount;
 private:
  void MakeRasterFont(void);               // make the raster fonts

 public:
  // Constructor and destructor
  SJCFont(void);
  ~SJCFont(void);

  // Print out the string
  void PrintString(uint x, uint y, uint z, const char* s);

};


#endif

