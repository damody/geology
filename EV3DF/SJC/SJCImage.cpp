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

                  ** Change the original format byte to ubyte.

     Compiler:    g++

     Platform:    Linux
*************************************************************************/

#include <SJCImage.h>
#include <fstream>

#include <math.h>

#include <FL/gl.h>
#include <GL/glu.h>

// had to hack header to get it to work in windows
extern "C"
{
#ifdef _WINDOWS
  #include "jpeglib.h" 
#else
  #include <jpeglib.h>
#endif
}// extern "C"

//****************************************************************************
//
// * Constructor.
//============================================================================
SJCImage::SJCImage(void) 
       : width(0), height(), format(0), bytesPerPixel(0), 
	 alignment(INVALID_ALIGNMENT), aImageData(NULL)
//============================================================================
{}// SJCImage


//****************************************************************************
//
// * Destructor.
//============================================================================
SJCImage::
~SJCImage()
//============================================================================
{
  delete [] aImageData;
}// ~SJCImage


//****************************************************************************
//
// * Parse the file name for the file extension and call the appropriate 
//   image loader.
//============================================================================
bool SJCImage::
LoadImage(const std::string& sFileName)
//============================================================================
{
  // extract file extension from file name
  uint dotPosition = sFileName.rfind('.');
  if (dotPosition == std::string::npos)  {
    SJCWarning(sFileName + " could not be parsed for file extension.");
    return false;
  }// if

  char aExtension[10];
  sFileName.copy(aExtension, 
		 sFileName.size() - dotPosition - 1, 
		 dotPosition + 1);
  aExtension[sFileName.size() - dotPosition - 1] = '\0';
  std::string sExtension(aExtension);

  for (uint i = 0; i < sExtension.size(); ++i)
    sExtension[i] = toupper(sExtension[i]);

  // call appropriate image loader
  if (!sExtension.compare("BMP"))
    return LoadBMP(sFileName);
  else if (!sExtension.compare("JPG") || !sExtension.compare("JPEG"))
    return LoadJPEG(sFileName);
  else if (!sExtension.compare("TGA") || !sExtension.compare("TARGA"))
    return LoadTARGA(sFileName);
  else  {
    SJCWarning((sFileName + " is not a supprted image type.").c_str());
    return false;
  }// else
  return true;
}// LoadImage

//****************************************************************************
//
// * Resize the image to have dimensions that are powers of 2 and are 
//   greater than 64.  Width and height do not have to be equal.
//============================================================================
bool SJCImage::
ResizeImage(void)
//============================================================================
{
  // local variables
  GLubyte*	aScaledBuffer; // buffer for image data after it is scaled
  GLuint	scaledWidth;   // width to scale image to
  GLuint	scaledHeight;  // height to scale image to

  // the texture must have dimensions of 2^n by 2^m, so 
  // it may need to be scaled
  scaledWidth = (GLuint) pow(2, ceil(log(width)/log(2)));
  scaledHeight = (GLuint) pow(2, ceil(log(height)/log(2)));
  
  // texture must be at least 64 x 64
  scaledWidth = SJCMax(scaledWidth, static_cast<unsigned int>(64));
  scaledHeight = SJCMax(scaledHeight, static_cast<unsigned int>(64));

  // scale image if neccessary
  if (width != scaledWidth || height != scaledHeight)  {
    if (((aScaledBuffer = 
	  new GLubyte[scaledWidth * scaledHeight * bytesPerPixel]) == NULL) || 
	gluScaleImage(format, width, height, GL_UNSIGNED_BYTE, aImageData, 
		      scaledWidth, scaledHeight, GL_UNSIGNED_BYTE, 
		      aScaledBuffer)) {	
      delete[] aScaledBuffer;
      return false;
    }// if

    delete aImageData;
    aImageData = aScaledBuffer;
    width = scaledWidth;
    height = scaledHeight;
  }// if
   	
  return true;
}// ResizeImage

//****************************************************************************
//
// * Get the color of the given pixel.
//============================================================================
SJCColor SJCImage::
GetPixel(uint x, uint y) const
//============================================================================
{
  uint bytesPerRow = 0;
  if (alignment == BYTE_ALIGNED)
    bytesPerRow = width * bytesPerPixel;
  else if (alignment == WORD_ALIGNED)  {
    bytesPerRow = (width * bytesPerPixel % 2) ? 
      width * bytesPerPixel + 1 : width * bytesPerPixel;
  }// else if
  else if (alignment == DOUBLE_WORD_ALIGNED)  {
    bytesPerRow = (width * bytesPerPixel % 4) ? 
      (width * bytesPerPixel / 4 + 1) * 4 : width * bytesPerPixel;
  }// else if
  else
    SJCError("Image has an invalid alignment.");

  ubyte* pPixel = aImageData + (y * bytesPerRow + x * bytesPerPixel);
  if (format == GL_RGB)
    return SJCColor(*pPixel, *(pPixel + 1), *(pPixel + 2));
  else if (format == GL_BGRA_EXT)
    return SJCColor(*(pPixel + 2), *(pPixel + 1), *pPixel, *(pPixel + 3));
  else  {
    SJCError("Image has an invalid pixel format.");
    return SJCColor();
  }// else
}// GetPixel

//****************************************************************************
//
// * Load a bmp image file.  Only uncompressed true color images are 
//   supported.
//============================================================================
bool SJCImage::
LoadBMP(const std::string& sFileName)
//============================================================================
{
  // attempt to open file for reading
  std::ifstream file(sFileName.c_str(), std::ios::in | std::ios::binary);
  if (!file.is_open())  {
    SJCWarning("Unable to open file " + sFileName + ".");
    return false;
  }// if
  
  // read header
  if (ReadByte(file) != 'B' || ReadByte(file) != 'M')  {
    SJCWarning(sFileName + " is not of a supported BMP format.");
    return false;
  }// if

  file.ignore(8);
  unsigned int uImageDataOffset = ReadDoubleWord(file);
  file.ignore(4);
  width = ReadDoubleWord(file);
  height = ReadDoubleWord(file);
  file.ignore(2);
  bytesPerPixel = ReadWord(file) / 8;
  format = GL_RGB;
  if (ReadDoubleWord(file))  {
    SJCWarning((sFileName + " is not true color.  Only true color BMPs are supported.").c_str());
    return false;
  }// if
  file.seekg(uImageDataOffset);

  if (bytesPerPixel == 3)  {
    alignment = DOUBLE_WORD_ALIGNED;
    int temp = width * 3;
    int bytesPerRow = (temp % 4) ? (temp / 4 + 1) * 4 : temp;
    aImageData = new ubyte[height * bytesPerRow];
    
    for (int row = height - 1; row >= 0; --row)
      file.read(reinterpret_cast<char*>(&aImageData[row * bytesPerRow]), 
		bytesPerRow);
  }// if
  else  {  // 4 bytes per pixel (hi byte not used)
    alignment = BYTE_ALIGNED;
    int bytesPerRow = width * 3;
    for (uint row = height - 1; row >= 0; --row)
      for (int col = 0; row < width; ++col)   {
	file.ignore(1);
	file.read(reinterpret_cast<char*>
		  (&aImageData[row * bytesPerRow + col * 3]), 3);
      }// for
  }// else

  return true;
}// LoadBMP

//****************************************************************************
//
// * Load a jpeg image file.  Only true color jpegs are supported.
//============================================================================
bool SJCImage::
LoadJPEG(const std::string& sFileName)
//============================================================================
{
  // attemp to open the file
  FILE *pFile = fopen(sFileName.c_str(), "rb");
  if (!pFile)  {
    SJCWarning(("Unable to open file " + sFileName + ".").c_str());
    return false;
  }// if

   // create jped decompressor and read the header
  jpeg_decompress_struct  decompressor;
  jpeg_error_mgr errorManager;
  jpeg_create_decompress(&decompressor);
  decompressor.err = jpeg_std_error(&errorManager);
  jpeg_stdio_src(&decompressor, pFile);
  jpeg_read_header(&decompressor, TRUE);

  // is the image true color?
  if (decompressor.num_components != 3)  {
    SJCWarning((sFileName + " is not true color.  Only true color JPEGs are supprted.").c_str());
    jpeg_destroy_decompress(&decompressor);
    return false;
  }// if
  jpeg_start_decompress(&decompressor);
  
  // create the image object and set the image properties from the header
  width = decompressor.output_width;
  height = decompressor.output_height;
  format = GL_RGB;
  bytesPerPixel = decompressor.out_color_components;
  alignment = BYTE_ALIGNED;
  int rowLength = width * bytesPerPixel;
  aImageData = new ubyte[height * rowLength];

  // get the image data
  ubyte* pRow;
  for (uint i = 0; i < height; ++i)  {
    pRow = &aImageData[i * rowLength];
    jpeg_read_scanlines(&decompressor, &pRow, 1);
  }// for
  
  // clean up
  jpeg_finish_decompress(&decompressor);
  jpeg_destroy_decompress(&decompressor);
  
  return true;
}// LoadJPEG


//****************************************************************************
//
// * Load a targa image file.  Only true color targas are supported.
//============================================================================
bool SJCImage::
LoadTARGA(const std::string& sFileName)
//============================================================================
{
  // attempt to open file for reading
  std::ifstream file(sFileName.c_str(), std::ios::in | std::ios::binary);
  if (!file.is_open())  {
    SJCWarning(("Unable to open file " + sFileName + ".").c_str());
    return false;
  }// if

  // read the targa header
  byte imageDescriptionLength = ReadByte(file);
  if (ReadByte(file))  {
    SJCWarning((sFileName + 
	     "is palleted.  Palleted TARGAs are not supported.").c_str());
    file.close();
    return false;
  }// if

  int imageType = ReadByte(file);
  if (imageType != 2 && imageType != 10)  {
    SJCWarning((sFileName + "is not true color. Only true color TARGAs are supported.").c_str());
    file.close();
    return false;
  }// if
  bool bImageIsRLE = (imageType == 10);
  
  file.ignore(9);
  width = ReadWord(file);
  height = ReadWord(file);
  bytesPerPixel = ReadByte(file) / 8;
  if (bytesPerPixel == 3)
    bytesPerPixel = GL_BGR_EXT;
  else if (bytesPerPixel == 4)
    format = GL_BGRA_EXT;
  else  {
    SJCWarning((sFileName + "is not true color.  Only true color TARGAs are supported.").c_str());
    file.close();
    return false;
  }// else
  
  bool bFlipped = (ReadByte(file) & 32) == 1;
  file.ignore(imageDescriptionLength);
  alignment = BYTE_ALIGNED;
  
  // allocate space for image data
  long imageSize = width * height * bytesPerPixel;
  aImageData = new ubyte[imageSize];

  // read in the image data
  if (bImageIsRLE)  {
    bool    bRLEPacket;
    ubyte   header;
    ubyte   aBuffer[4];
    ubyte  *pCurrentPos;
    int     numPixelsInPacket;
    int     pixelsRead = 0;
    int     totalPixels = width * height;
      
    while (pixelsRead != totalPixels)   {
      SJCAssert(pixelsRead < totalPixels, 
	     "Error reading targa file.  RLE has failed.");

      // read a packet header
      file.read(reinterpret_cast<char*>(&header), 1);
      bRLEPacket = (header && 128) == 1;
      numPixelsInPacket = (header && 127) + 1;
      
      if (bRLEPacket)  {
	// get the pixel value
	pCurrentPos = aImageData;
	file.read(reinterpret_cast<char*>(aBuffer), bytesPerPixel);
               
	// copy it into the image data repeatedly
	for (int i = 0; i < numPixelsInPacket; ++i) {
	  memcpy(pCurrentPos, aBuffer, bytesPerPixel);
	  pCurrentPos += bytesPerPixel;
	}// for
      }// if
      else // this packet is not rle encoded so just read in the data
	file.read(reinterpret_cast<char*>
		  (aImageData + (pixelsRead * bytesPerPixel)), 
		  numPixelsInPacket);       

         
      pixelsRead += numPixelsInPacket;
    }// while
  }// if
  else // no rle encoding so just read the image data
    file.read(reinterpret_cast<char*>(aImageData), imageSize);          

  file.close();

  // flip the image data if needed
  if (bFlipped)  {
    int rowsToCopy = height / 2;
    ubyte *pStartOfLastRow = aImageData + (height - 1) * width;
    ubyte *aSwapBuffer = new ubyte[width];
            
    for (int i = 0; i <= rowsToCopy; ++i)  {
      memcpy(aSwapBuffer, aImageData + (width * i), width);
      memcpy(aImageData + (width * i), pStartOfLastRow - (width * i), width);
      memcpy(aImageData + (width * i), aSwapBuffer, width);
    }// for
    delete[] aSwapBuffer;
  }// if
  
  return true;
}// LoadTarga

//****************************************************************************
//
// * Read a byte (8-bit value) from the given stream.
//============================================================================
byte SJCImage::ReadByte(std::istream& in) const
//============================================================================
{
  char temp;
  in.read(&temp, 1); 
  return temp;
}// ReadByte


//****************************************************************************
//
// * Read a word (16-bit value) from the given stream.
//============================================================================
uint16 SJCImage::ReadWord(std::istream& in) const
//============================================================================
{
   uint16 temp;
   in.read(reinterpret_cast<char*>(&temp), 2);
   return temp;
}// ReadWord

//****************************************************************************
//
// * Read a double word (32-bit value) from the given stream.
//============================================================================
uint32 SJCImage::
ReadDoubleWord(std::istream& in) const
//============================================================================
{
  uint32 temp;
  in.read(reinterpret_cast<char*>(&temp), 4);
  return temp;
}// ReadDoubleWord
