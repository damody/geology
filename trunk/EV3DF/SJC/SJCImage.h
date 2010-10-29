/************************************************************************
     Main File:

     File:        SJCImage.h

     Author:      
                  Eric McDaniel, chat@cs.wisc.edu

     Written:     2002 summer
 
     Comment:     Image operation

     Functions: 
                  1. Constructor and destructor
                  2. LoadImage: Read in the image and according to the 
                     extesion to call proper loading fun
                  3. ResizeImage: Resize the image
                  4. GetPixel: Get the color of the pixel at x, y
                  The targa image reading is implement by eric which
                  is not anything about the libtarga.
     Compiler:    g++

     Platform:    Linux
*************************************************************************/
#ifndef _SJCIMAGE_H
#define _SJCIMAGE_H

#include <SJC/SJC.h>


#include <SJCColor.h>

class SJCImage
{
 public:
  enum SJCEAlignment   {
    INVALID_ALIGNMENT   = 0,
    BYTE_ALIGNED        = 1,
    WORD_ALIGNED        = 2,
    DOUBLE_WORD_ALIGNED = 4
  };// SJCEAlignment

 public:
  uint           width;            // width and height of the image
  uint           height;          
  uint           format;           // the format of the image

  uint           bytesPerPixel;    // how many byte per pixels
  SJCEAlignment  alignment;        // The alignment time
  ubyte         *aImageData;       // Data

 public:
  // Constructor and desctructor
  SJCImage(void);
  ~SJCImage(void);

  // Read in the image and according to the extesion to call proper loading fun
  bool LoadImage(const std::string& sFileName);

  // Resize the image
  bool ResizeImage(void);

  // Get the color of the pixel at x, y
  SJCColor GetPixel(uint x, uint y) const;
  
 private:
  // Load in the bmp file
  bool LoadBMP(const std::string& sFileName);

  // Load in the jpeg file
  bool LoadJPEG(const std::string& sFileName);

  // Load in the targa file
  bool LoadTARGA(const std::string& sFileName);

  // Read one byte from the stream
  byte ReadByte(std::istream& in) const;

  // Read one word from the stream which 2byte
  uint16 ReadWord(std::istream& in) const;

  // Read in 4 bytes from the stream
  uint32 ReadDoubleWord(std::istream& in) const;
};// SJCImage

#endif // _CIMAGE_
