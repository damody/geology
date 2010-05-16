/************************************************************************
     Main File:

     File:        SJCTargaImage.h

     Author:
                  Steven Chenney, schenney@cs.wisc.edu

     Modifier
                  Yu-Chi Lai, yu-chi@cs.wisc.edu

     Comment:

    Truecolor images supported:

    bits            breakdown   components
    --------------------------------------
    32              8-8-8-8     RGBA
    24              8-8-8       RGB
    16              5-6-5       RGB
    15              5-5-5-1     RGB (ignore extra bit)


    Paletted images supported:
    
    index size      palette entry   breakdown   components
    ------------------------------------------------------
    8               <any of above>  <same as above> ..
    16              <any of above>  <same as above> ..
    24              <any of above>  <same as above> ..

    Image data will start in the low-left corner of the image.

    All data inside is stored in RGBA

    Contructor:
                 0 paras: default
                 2 Real: w, h and allocate the memory
                 2 real, array: w, h and data
                 ifstream& f, input from stream
                 
                 TargaImage: copy the image

    Functions:  
                 1. GetRGB(array, bool) : Convert the image to RGB format, 
                             by dividing out 
                             the alpha. If flip is true, the image is flipped 
                             top to bottom 
                 2. GetRGBA(array, bool): Get the RGBA format
                 3. GetA(array, bool): Get the A
                 4. Width: get the width of the image
                 5. Height: get the height of the image
                 8. Grayscale(ubyte* data_out): Transform to gray scale;
                 9. Comp_Over(SJCTargaImage*): Composite the current image 
                    over the given image.
                10. Comp_In(SJCTargaImage*): Composite this image "in" the 
                    given image
                11. Comp_Out(SJCTargaImage*): Composite this image "out" the 
                    given image.
                12. Comp_Atop(SJCTargaImage*): Composite current image "atop" 
                    given image.
                13. Comp_Xor(SJCTargaImage*): Composite this image with given 
                    image using exclusive or (XOR).
                14. Difference(SJCTargaImage*): Calculate the difference 
                    bewteen this imag and the given one.  Image dimensions 
                    must be equal.;
                15. Filter_Enhance: Perform a 5x5 enhancement filter to this 
                    image
                16.  Filter_Box: Perform 5x5 box filter on this image
                17.  Filter_Bartlett: Perform 5x5 Bartlett filter on this 
                                      image 
                18.  Filter_Gaussian Perform 5x5 Gaussian filter on this image
                19.  Filter_Gaussian_N(uint N): Perform NxN Gaussian filter on 
                                     this image
                20.  Filter_Edge: Perform 5x5 edge detect (high pass) filter 
                                      on this image
                21. Filter_Enhance: Perform a 5x5 enhancement filter to this 
                                       image
                22.  Resize(float scale): Resize operation;

                23.  Rotate(float): Rotate the image clockwise by the given 
                                    angle.  Do not resize the image
                24. >>: output to the targa image format

************************************************************************/

#ifndef _TARGA_IMAGE_H_
#define _TARGA_IMAGE_H_

#include <iostream>

#include <SJC/SJC.h>

#include <SJCColor.h>

class SJCTargaImage
{

  // methods
 public:
  // Constructors 
  SJCTargaImage(void);
  SJCTargaImage(uint w, uint h);
  SJCTargaImage(uint w, uint h, ubyte *d);
  // Input constructor
  SJCTargaImage(std::istream &f);
  // Copy constructor
  SJCTargaImage(const SJCTargaImage& image);
  // destructor
  ~SJCTargaImage(void);
  
  // Output operator
  friend std::ostream& operator<<(std::ostream &o,
				  const SJCTargaImage &vf);
  
  // Convert the image to RGB format, by dividing out the alpha.
  // If flip is true, the image is flipped top to bottom
  bool	GetRGB(ubyte* data_out, const bool flip=false);
  bool	GetRGBA(ubyte* data_out, const bool flip=false);
  bool	GetA(ubyte* data_out, const bool flip=false);
  
  
  // Get the width and height
  uint	Width(void) const { return width; }
  uint	Height(void) const { return height; }
  
  
  // Transform to gray scale
  void	Grayscale(ubyte* data_out);
  
  // Compose operation
  // Composite the current image over the given image.
  void 	Comp_Over(SJCTargaImage* pImage);
  // Composite this image "in" the given image
  void 	Comp_In(SJCTargaImage* pImage);
  // Composite this image "out" the given image.
  void 	Comp_Out(SJCTargaImage* pImage);
  // Composite current image "atop" given image.
  void 	Comp_Atop(SJCTargaImage* pImage);
  // Composite this image with given image using exclusive or (XOR).
  void 	Comp_Xor(SJCTargaImage* pImage);
  
  // Calculate the difference bewteen this imag and the given one.  Image 
  // dimensions must be equal.
  void 	Difference(SJCTargaImage* pImage);
  
  // Filter operation
  // Perform 5x5 box filter on this image
  void 	Filter_Box(void);
  // Perform 5x5 Bartlett filter on this image
  void 	Filter_Bartlett(void);
  // Perform 5x5 Gaussian filter on this image
  void 	Filter_Gaussian(void);
  // Perform NxN Gaussian filter on this image
  void 	Filter_Gaussian_N(uint N);

  // Perform 5x5 edge detect (high pass) filter on this image
  void 	Filter_Edge(void);
  // Perform a 5x5 enhancement filter to this image
  void 	Filter_Enhance(void);
  
  // Resize operation
  void 	Resize(float scale);
  // Rotate the image clockwise by the given angle.  Do not resize the image
  void 	Rotate(float angleDegrees);

 protected:
  uint		width;        // width and height of the image
  uint		height;
  ubyte		*data;        // Data
 
 protected:
  // Transform single pixel from RGBA to RGB
  void 		RGBA_To_RGB(ubyte *data, ubyte *rgb);

  // Clear the data to black
  void 		ClearToBlack(void);
  
  // Apply the filter to the image
  void 		Filter_Image(float **filter, uint fw, uint fh,
			     bool do_alpha);
  // Reconstruct the image
  void 		Reconstruct(float, float, ubyte*);

};


#endif


