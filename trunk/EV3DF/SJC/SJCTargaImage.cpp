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
#include <SJCTargaImage.h>
#include <SJCException.h>

#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <math.h>

// constants
const int           RED             = 0;                // red channel
const int           GREEN           = 1;                // green channel
const int           BLUE            = 2;                // blue channel
const ubyte BACKGROUND[3]   = { 0, 0, 0 };      // background color


const int TGA_IMG_NODATA = 0;
const int TGA_IMG_UNC_PALETTED = 1;
const int TGA_IMG_UNC_TRUECOLOR = 2;
const int TGA_IMG_UNC_GRAYSCALE = 3;
const int TGA_IMG_RLE_PALETTED = 9;
const int TGA_IMG_RLE_TRUECOLOR = 10;
const int TGA_IMG_RLE_GRAYSCALE = 11;


const int TGA_LOWER_LEFT = 0;
const int TGA_LOWER_RIGHT = 1;
const int TGA_UPPER_LEFT = 2;
const int TGA_UPPER_RIGHT = 3;


const int HDR_LENGTH = 18;
const int HDR_IDLEN = 0;
const int HDR_CMAP_TYPE = 1;
const int HDR_IMAGE_TYPE = 2;
const int HDR_CMAP_FIRST = 3;
const int HDR_CMAP_LENGTH = 5;
const int HDR_CMAP_ENTRY_SIZE = 7;
const int HDR_IMG_SPEC_XORIGIN = 8;
const int HDR_IMG_SPEC_YORIGIN = 10;
const int HDR_IMG_SPEC_WIDTH = 12;
const int HDR_IMG_SPEC_HEIGHT = 14;
const int HDR_IMG_SPEC_PIX_DEPTH = 16;
const int HDR_IMG_SPEC_IMG_DESC = 17;



const int TGA_ERR_NONE = 0;
const int TGA_ERR_BAD_HEADER = 1;
const int TGA_ERR_OPEN_FAILS = 2;
const int TGA_ERR_BAD_FORMAT = 3;
const int TGA_ERR_UNEXPECTED_EOF = 4;
const int TGA_ERR_NODATA_IMAGE = 5;
const int TGA_ERR_COLORMAP_FOR_GRAY = 6;
const int TGA_ERR_BAD_COLORMAP_ENTRY_SIZE = 7;
const int TGA_ERR_BAD_COLORMAP = 8;
const int TGA_ERR_READ_FAILS = 9;
const int TGA_ERR_BAD_IMAGE_TYPE = 10;
const int TGA_ERR_BAD_DIMENSIONS = 11;


static int16
ttohs( int16 val )
{
#ifdef __BIG_ENDIAN__
  return( ((val & 0xFF) << 8) | ((val & 0xFF00) >> 8) );
#else
  return( val );
#endif 

}


static int32
ttohl( int32 val )
{
#ifdef __BIG_ENDIAN__
  return( ((val & 0x000000FF) << 24) |
	  ((val & 0x0000FF00) << 8)  |
	  ((val & 0x00FF0000) >> 8)  |
	  ((val & 0xFF000000) >> 24) );
#else
  return( val );
#endif 
}


//*************************************************************************
//
// * write the pixel to the data regarding how the
//   header says the data is ordered.
//==========================================================================
static void
tga_write_pixel_to_mem(ubyte * dat, ubyte img_spec, uint32 number, 
		       uint32 w, uint32 h, uint32 pixel)
//==========================================================================
{

  
  uint32 j;
  uint32 x, y;
  uint32 addy;
  
  switch( (img_spec & 0x30) >> 4 ) {

    case TGA_LOWER_RIGHT:
      x = w - 1 - (number % w);
      y = number / h;
      break;
    
    case TGA_UPPER_LEFT:
      x = number % w;
      y = h - 1 - (number / w);
      break;

    case TGA_UPPER_RIGHT:
      x = w - 1 - (number % w);
      y = h - 1 - (number / w);
      break;
	
    case TGA_LOWER_LEFT:
    default:
      x = number % w;
      y = number / w;
      break;

  }
  
  addy = (y * w + x) * 4;
  for( j = 0; j < 4; j++ ) {
    dat[addy + j] = (ubyte)((pixel >> (j * 8)) & 0xFF);
  }
    
}


//*************************************************************************
//
// * get the image data value out 
//==========================================================================
static uint32
tga_get_pixel(std::istream &tga, ubyte bytes_per_pix,
              ubyte *colormap, ubyte cmap_bytes_entry )
//==========================================================================
{
    

  uint32 tmp_col;
  uint32 tmp_int32;
  ubyte tmp_byte;
  
  uint32 j;
  
  tmp_int32 = 0;
  for( j = 0; j < (uint32)bytes_per_pix; j++ ) {
    tga.read((char*)&tmp_byte, sizeof(ubyte));
    if ( tga.gcount() != sizeof(ubyte) ) {
      tmp_int32 = 0;
    }
    else       {
      tmp_int32 += tmp_byte << (j * 8);
    }
  }
  
  if( colormap != NULL ) {
    // need to look up value to get real color 
    tmp_col = 0;
    for( j = 0; j < cmap_bytes_entry; j++ ) {
      tmp_col += colormap[cmap_bytes_entry * tmp_int32 + j] << (8 * j);
    }
  } else {
    tmp_col = tmp_int32;
  }
  
  return( tmp_col );
    
}
//*************************************************************************
//
// * This is not only responsible for converting from different depths
//   to other depths, it also switches BGR to RGB.
//   this thing will also premultiply alpha, on a pixel by pixel basis.
//==========================================================================
static uint32
tga_convert_color(uint32 pixel, uint32 bpp_in, ubyte alphabits)
//==========================================================================
{
    
  ubyte r, g, b, a;
  
  switch( bpp_in ) {
    
    case 32:
      if( alphabits == 0 ) {
	goto is_24_bit_in_disguise;
      }
      // 32-bit to 32-bit -- nop.
      break;
      
    case 24: is_24_bit_in_disguise:
      // 24-bit to 32-bit; (only force alpha to full)
      pixel |= 0xFF000000;
      break;

    case 15: is_15_bit_in_disguise:
      r = (ubyte)(((float)((pixel & 0x7C00) >> 10)) * 8.2258f);
      g = (ubyte)(((float)((pixel & 0x03E0) >> 5 )) * 8.2258f);
      b = (ubyte)(((float)(pixel & 0x001F)) * 8.2258f);
      // 15-bit to 32-bit; (force alpha to full)
      pixel = 0xFF000000 + (r << 16) + (g << 8) + b;
      break;
        
    case 16:
      if( alphabits == 1 ) {
	goto is_15_bit_in_disguise;
      }
      // 16-bit to 32-bit; (force alpha to full)
      r = (ubyte)(((float)((pixel & 0xF800) >> 11)) * 8.2258f);
      g = (ubyte)(((float)((pixel & 0x07E0) >> 5 )) * 4.0476f);
      b = (ubyte)(((float)(pixel & 0x001F)) * 8.2258f);
      pixel = 0xFF000000 + (r << 16) + (g << 8) + b;
      break;
      
    }
    
  // convert the 32-bit pixel from BGR to RGB.
  pixel = (pixel & 0xFF00FF00) + 
          ((pixel & 0xFF) << 16) + 
          ((pixel & 0xFF0000) >> 16);

  r = pixel & 0x000000FF;
  g = (pixel & 0x0000FF00) >> 8;
  b = (pixel & 0x00FF0000) >> 16;
  a = (pixel & 0xFF000000) >> 24;
  
  // not premultiplied alpha -- multiply.
  r = (ubyte)(((float)r / 255.0f) * ((float)a / 255.0f) * 255.0f);
  g = (ubyte)(((float)g / 255.0f) * ((float)a / 255.0f) * 255.0f);
  b = (ubyte)(((float)b / 255.0f) * ((float)a / 255.0f) * 255.0f);
  
  pixel = r + (g << 8) + (b << 16) + (a << 24);

  return( pixel );

}
//**************************************************************************
//
// *
//==========================================================================
void SJCTargaImage::
Filter_Image(float **filter, uint fw, uint fh, bool do_alpha)
//==========================================================================
{
  float   ***in_image;
  float   ***out_image;
  float   sums[4];
  
  in_image = new float**[4];
  in_image[0] = new float*[height];
  in_image[1] = new float*[height];
  in_image[2] = new float*[height];
  in_image[3] = new float*[height];

  out_image = new float**[4];
  out_image[0] = new float*[height];
  out_image[1] = new float*[height];
  out_image[2] = new float*[height];
  out_image[3] = new float*[height];
  
  uint m;
  for (uint i = 0 ; i < height ; i++)   {
    in_image[0][i] = new float[width];
    in_image[1][i] = new float[width];
    in_image[2][i] = new float[width];
    in_image[3][i] = new float[width];
    out_image[0][i] = new float[width];
    out_image[1][i] = new float[width];
    out_image[2][i] = new float[width];
    out_image[3][i] = new float[width];
    for (uint j = 0 ; j < width ; j++)       {
      in_image[0][i][j] = data[m++] / (float)255;
      in_image[1][i][j] = data[m++] / (float)255;
      in_image[2][i][j] = data[m++] / (float)255;
      in_image[3][i][j] = data[m++] / (float)255;

      if ( in_image[3][i][j] != 0.0 )	    {
	in_image[0][i][j] /= in_image[3][i][j];
	in_image[1][i][j] /= in_image[3][i][j];
	in_image[2][i][j] /= in_image[3][i][j];
      }
    }
  }

  for (uint i = 0 ; i < height ; i++)   {
    for (uint j = 0 ; j < width ; j++)     {
      sums[0] = sums[1] = sums[2] = sums[3] = 0.0;
      for (m = -fh / 2 ; m <= fh / 2 ; m++)          {
	uint        mi = i + m;

	if (mi < 0)
	  mi = -mi;
	if (mi >= height)
	  mi = height + height - mi - 2;
	
	for (uint n = -fw / 2 ; n <= fw / 2 ; n++)               {
	  uint        ni = j + n;
	  
	  if (ni < 0)
	    ni = -ni;
	  if (ni >= width)
	    ni = width + width - ni - 2;
	  
	  for (uint k = 0 ; k < 4 ; k++) {
	    sums[k] += in_image[k][mi][ni] * filter[m+fh/2][n+fw/2];
	  }
	}
      }

      out_image[0][i][j] = sums[0];
      out_image[1][i][j] = sums[1];
      out_image[2][i][j] = sums[2];
      if (do_alpha)
	out_image[3][i][j] = sums[3];
      else
	out_image[3][i][j] = in_image[3][i][j];
    }
  }

  m = 0;
  for (uint i = 0 ; i < height ; i++)   {
    for (uint j = 0 ; j < width ; j++)     {
      int        val;
	
      val = (int)floor(out_image[0][i][j] * out_image[3][i][j] * 255);
      if (val < 0)
	data[m++] = 0;
      else if (val > 255)
	data[m++] = 255;
      else
	data[m++] = val;
      val = (int)floor(out_image[1][i][j] * out_image[3][i][j] * 255);
      if (val < 0)
	data[m++] = 0;
      else if (val > 255)
	data[m++] = 255;
      else
	data[m++] = val;
      val = (int)floor(out_image[2][i][j] * out_image[3][i][j] * 255);
      if (val < 0)
	data[m++] = 0;
      else if (val > 255)
	data[m++] = 255;
      else
	data[m++] = val;
      val = (int)floor(out_image[3][i][j] * 255);
      if (val < 0)
	data[m++] = 0;
      else if (val > 255)
	data[m++] = 255;
      else
	data[m++] = val;
    }
    delete[] in_image[0][i];
    delete[] in_image[1][i];
    delete[] in_image[2][i];
    delete[] in_image[3][i];
    delete[] out_image[0][i];
    delete[] out_image[1][i];
    delete[] out_image[2][i];
    delete[] out_image[3][i];
  }
  
  delete[] in_image[0];
  delete[] in_image[1];
  delete[] in_image[2];
  delete[] in_image[3];
  delete[] in_image;
  delete[] out_image[0];
  delete[] out_image[1];
  delete[] out_image[2];
  delete[] out_image[3];
  delete[] out_image;
}



//**************************************************************************
//
// *
//==========================================================================
double Binomial(int n, int s)
//==========================================================================
{
  double        res;
  
  res = 1;
  for (int i = 1 ; i <= s ; i++)
    res = (n - i + 1) * res / i ;

  return res;
}// Binomial


//**************************************************************************
//
// *
//==========================================================================
void SJCTargaImage::
Reconstruct(float x, float y, ubyte* colorOutput)
//==========================================================================
{
  float   sums[4];
  int     val;
  
  sums[0] = sums[1] = sums[2] = sums[3] = 0.0f;
  
  for ( int m = (int)ceil(y - 2.0) ; m <= y + 2.0 ; m++)   {
    int mi = m;
    
    if (mi < 0)
      mi = -mi;
    if (mi >= (int)height)
      mi = height + height - mi - 2;
    
    for ( int n = (int)ceil(x - 2.0) ; n <= x + 2.0 ; n++){
      int        ni = n;
      
      if (ni < 0)
	ni = -ni;
      if (ni >= (int)width)
	ni = width + width - ni - 2;
      
      float   pixel[4];
      int     offset = (mi * width + ni) * 4;
      if ( data[offset + 3] )	{
	pixel[0] = (float)data[offset] / (float)data[offset+3];
	pixel[1] = (float)data[offset+1] / (float)data[offset+3];
	pixel[2] = (float)data[offset+2] / (float)data[offset+3];
	pixel[3] = (float)data[offset+3] / 255.0f;
      }
      else
	pixel[0] = pixel[1] = pixel[2] = pixel[3] = 0.0f;
      
      for ( int k = 0 ; k < 4 ; k++)
	sums[k] += pixel[k] * (2.0f - fabs(y-m)) * (2.0f - fabs(x-n));
    }
  }
  
  sums[0] /= 16.0f;
  sums[1] /= 16.0f;
  sums[2] /= 16.0f;
  sums[3] /= 16.0f;

  for ( unsigned int i = 0 ; i < 3 ; i++ )   {
    val = (int)floor(sums[i] * sums[3] * 255.0f);
    if (val < 0)
      colorOutput[i] = 0;
    else if (val > 255)
      colorOutput[i] = 255;
    else
      colorOutput[i] = val;
  }
  val = (int)floor(sums[3] * 255);
  if (val < 0)
    colorOutput[3] = 0;
  else if (val > 255)
    colorOutput[3] = 255;
  else
    colorOutput[3] = val;
  
}//Reconstruct

//**************************************************************************
//
// * Constructor.  Initialize member variables.
//==========================================================================
SJCTargaImage::SJCTargaImage(void) 
  : width(0), height(0), data(NULL)
//==========================================================================
{
}// SJCTargaImage
//**************************************************************************
//
// * Constructor.  Initialize member variables
//==========================================================================
SJCTargaImage::SJCTargaImage(uint w, uint h) : width(w), height(h)
{
  data = new ubyte[width * height * 4];
  ClearToBlack();
}// SJCTargaImage

//**************************************************************************
//
// * Constructor.  Initialize member variables to values given.
//==========================================================================
SJCTargaImage::SJCTargaImage(uint w, uint h, ubyte *d)
//==========================================================================
{
  width = w;
  height = h;
  data = new ubyte[width * height * 4];
  memcpy(data, d, sizeof(ubyte) * width * height * 4);
}

///////////////////////////////////////////////////////////////////////////////
//
//      Copy Constructor.  Initialize member to that of input
//
///////////////////////////////////////////////////////////////////////////////
//**************************************************************************
//
// * Copy Constructor.  Initialize member to that of input
//==========================================================================
SJCTargaImage::SJCTargaImage(const SJCTargaImage& image) 
//==========================================================================
{
  width = image.width;
  height = image.height;
  data = NULL; 
  if (image.data != NULL) {
    data = new ubyte[width * height * 4];
    memcpy(data, image.data, sizeof(ubyte) * width * height * 4);
  }
}


//**************************************************************************
//
// * Contructor from input file
//==========================================================================
SJCTargaImage::SJCTargaImage(std::istream &f)
//==========================================================================
{
  char   idlen;               // length of the image_id string below.
  char   cmap_type;           // paletted image <=> cmap_type
  char   image_type;          // can be any of the IMG_TYPE constants above.
  uint16 cmap_first;          // 
  uint16 cmap_length;         // how long the colormap is
  ubyte  cmap_entry_size;     // how big a palette entry is.
  uint16 img_spec_xorig;      // the x origin of the image in the image data.
  uint16 img_spec_yorig;      // the y origin of the image in the image data.
  uint16 img_spec_width;      // the width of the image.
  uint16 img_spec_height;     // the height of the image.
  char   img_spec_pix_depth;  // the depth of a pixel in the image.
  char   img_spec_img_desc;   // the image descriptor.
  
  ubyte* tga_hdr = 0;
  
  ubyte* colormap = 0;
  
  ubyte cmap_bytes_entry;
  uint32 cmap_bytes;
  
  uint32 tmp_col;
  uint32 tmp_int32;
  
  ubyte alphabits = 0;
  
  uint32 num_pixels;
  
  uint32 i;
  uint32 j;
  
  ubyte bytes_per_pix;
  
  ubyte true_bits_per_pixel;
  
  uint32 bytes_total = 0;
  
  ubyte packet_header;
  ubyte repcount;
  
  // read the header in. 
  tga_hdr = new ubyte[HDR_LENGTH];
  f.read((char*)tga_hdr, HDR_LENGTH);
  if ( f.gcount() != HDR_LENGTH )   {
    throw new SJCException("SJCTargaImage: Bad header data.\n");
    width = 0;
    height = 0;
    data = 0;
    return;
  }
  
  // byte order is important here. 
  idlen              = (ubyte)tga_hdr[HDR_IDLEN];
  
  image_type         = (ubyte)tga_hdr[HDR_IMAGE_TYPE];
  
  cmap_type          = (ubyte)tga_hdr[HDR_CMAP_TYPE];
  cmap_first         = ttohs( *(uint16 *)(&tga_hdr[HDR_CMAP_FIRST]) );
  cmap_length        = ttohs( *(uint16 *)(&tga_hdr[HDR_CMAP_LENGTH]) );
  cmap_entry_size    = (ubyte)tga_hdr[HDR_CMAP_ENTRY_SIZE];
  
  img_spec_xorig     = ttohs( *(uint16 *)(&tga_hdr[HDR_IMG_SPEC_XORIGIN]) );
  img_spec_yorig     = ttohs( *(uint16 *)(&tga_hdr[HDR_IMG_SPEC_YORIGIN]) );
  img_spec_width     = ttohs( *(uint16 *)(&tga_hdr[HDR_IMG_SPEC_WIDTH]) );
  img_spec_height    = ttohs( *(uint16 *)(&tga_hdr[HDR_IMG_SPEC_HEIGHT]) );
  img_spec_pix_depth = (ubyte)tga_hdr[HDR_IMG_SPEC_PIX_DEPTH];
  img_spec_img_desc  = (ubyte)tga_hdr[HDR_IMG_SPEC_IMG_DESC];
  
  delete[] tga_hdr;
  
  num_pixels = img_spec_width * img_spec_height;
  
  if( num_pixels == 0 ) {
    throw new SJCException("SJCTargaImage: Bad dimensions");
    width = height = 0;
    data = 0;
    return;
  }
    
  alphabits = img_spec_img_desc & 0x0F;
  
  // seek past the image id, if there is one 
  if( idlen )   {
    f.ignore(idlen);
  }
  

  // if this is a 'nodata' image, just jump out.
  if( image_type == TGA_IMG_NODATA ) {
    throw new SJCException("SJCTargaImage: No data image");
    width = height = 0;
    data = 0;
    return;
  }

  // deal with the colormap, if there is one. 
  if( cmap_type ) {
    
    switch( image_type ) {
      
      case TGA_IMG_UNC_PALETTED:
      case TGA_IMG_RLE_PALETTED:
	break;
            
      case TGA_IMG_UNC_TRUECOLOR:
      case TGA_IMG_RLE_TRUECOLOR:
	// this should really be an error, but some really old
	// crusty targas might actually be like this (created by
	// TrueVision, no less!) so, we'll hack our way through it.
	break;
	
      case TGA_IMG_UNC_GRAYSCALE:
      case TGA_IMG_RLE_GRAYSCALE:
	throw new SJCException("SJCTargaImage: Colormap for a gray image");
	width = height = 0;
	data = 0;
	return;
        }
        
    // ensure colormap entry size is something we support 
    if( !(cmap_entry_size == 15 || 
	  cmap_entry_size == 16 ||
	  cmap_entry_size == 24 ||
	  cmap_entry_size == 32) ) {
      throw new SJCException("SJCTargaImage: Bad colormap size");
      width = height = 0;
      data = 0;
      return;
    }
        
        
    // allocate memory for a colormap 
    if( cmap_entry_size & 0x07 ) {
      cmap_bytes_entry = (((8 - (cmap_entry_size & 0x07))
			   + cmap_entry_size) >> 3);
    } else {
      cmap_bytes_entry = (cmap_entry_size >> 3);
    }
        
    cmap_bytes = cmap_bytes_entry * cmap_length;
    colormap = new ubyte[cmap_bytes];
        
    for( i = 0; i < cmap_length; i++ ) {
            
      // seek ahead to first entry used 
      if( cmap_first != 0 ) {
	f.ignore(cmap_first * cmap_bytes_entry);
      }
      
      tmp_int32 = 0;
      ubyte val;
      for( j = 0; j < cmap_bytes_entry; j++ ) {
	f.read((char*)&val, 1);
	if ( f.gcount() != 1 )  {
	  throw new SJCException("SJCTargaImage: Bad colormap");
	  width = height = 0;
	  data = 0;
	  delete[] colormap;
	  return;
	}
	tmp_int32 = tmp_int32 | ( val << ( ( 3 - j ) * 8 ) );
      }

      // byte order correct.
      tmp_int32 = ttohl( tmp_int32 );
      
      for( j = 0; j < cmap_bytes_entry; j++ ) {
	colormap[i*cmap_bytes_entry+j] = (tmp_int32 >> (8 * j)) & 0xFF;
      }
      
    }

  }

  // compute num of bytes in an image data unit (either index or BGR triple)
  if( img_spec_pix_depth & 0x07 ) {
    bytes_per_pix = (((8 - (img_spec_pix_depth & 0x07))
		      + img_spec_pix_depth) >> 3);
  } else {
    bytes_per_pix = (img_spec_pix_depth >> 3);
  }

  // assume that there's one byte per pixel 
  if( bytes_per_pix == 0 ) {
    bytes_per_pix = 1;
  }
  
  // compute how many bytes of storage we need for the image 
  bytes_total = img_spec_width * img_spec_height * 4;
  
  data = new ubyte[bytes_total];
  
  // compute the true number of bits per pixel
  true_bits_per_pixel = cmap_type ? cmap_entry_size : img_spec_pix_depth;

  switch( image_type ) {

    case TGA_IMG_UNC_TRUECOLOR:
    case TGA_IMG_UNC_GRAYSCALE:
    case TGA_IMG_UNC_PALETTED:

      // FIXME: support grayscale 
      
      for( i = 0; i < num_pixels; i++ ) {
	
	// get the color value.
	tmp_col = tga_get_pixel(f, bytes_per_pix,
				colormap, cmap_bytes_entry );
	tmp_col = tga_convert_color(tmp_col, true_bits_per_pixel,alphabits);
	
	// now write the data out.
	tga_write_pixel_to_mem(data, img_spec_img_desc, 
			       i, img_spec_width, img_spec_height, tmp_col);
	
      }
      
      break;


    case TGA_IMG_RLE_TRUECOLOR:
    case TGA_IMG_RLE_GRAYSCALE:
    case TGA_IMG_RLE_PALETTED:

      // FIXME: handle grayscale..
      
      for( i = 0; i < num_pixels; ) {

	// a bit of work to do to read the data.. 
	f.read((char*)&packet_header, 1);
	if ( f.gcount() < 1 )  {
	  // well, just let them fill the rest with null pixels then...
	  packet_header = 1;
	}
	
	if( packet_header & 0x80 ) {
	  // run length packet 
	  
	  tmp_col = tga_get_pixel(f, bytes_per_pix, colormap,
				  cmap_bytes_entry );
	  tmp_col = tga_convert_color(tmp_col, true_bits_per_pixel,
				      alphabits);
	  
	  repcount = (packet_header & 0x7F) + 1;
	  
	  // write all the data out 
	  for( uint j = 0; j < repcount; j++ ) {
	    tga_write_pixel_to_mem(data, img_spec_img_desc, 
				   i + j, img_spec_width, 
				   img_spec_height, tmp_col);
	  }
	  
	  i += repcount;
	  
	} else {
	  // raw packet 
	  // get pixel from file 
	  
	  repcount = (packet_header & 0x7F) + 1;
	  
	  for( uint j = 0; j < repcount; j++ ) {
	    
	    tmp_col = tga_get_pixel(f, bytes_per_pix,
				    colormap, cmap_bytes_entry );
	    tmp_col = tga_convert_color(tmp_col, true_bits_per_pixel,
					alphabits);
	    
	    tga_write_pixel_to_mem(data, img_spec_img_desc, 
				   i + j, img_spec_width, img_spec_height, 
				   tmp_col);
	    
	  }

	  i += repcount;

	}
	
      }

      break;
    

    default:
      throw new SJCException("SJCTargaImage: Bad colormap");
      width = height = 0;
      delete[] data;
      data = 0;
      if ( colormap ) delete[] colormap;
      return;
    }
  
  if ( colormap ) delete[] colormap;
  
  width  = img_spec_width;
  height = img_spec_height;
  
}

//**************************************************************************
//
// * Destructor.  Free image memory.
//==========================================================================
SJCTargaImage::~SJCTargaImage(void)
//==========================================================================
{
  if (data)
    delete[] data;
}// ~SJCTargaImage


//**************************************************************************
//
// * Output to TGA format
//==========================================================================
std::ostream& operator<<(std::ostream &o, const SJCTargaImage &tga)
//==========================================================================
{
  uint32 i, j;
  uint32 oc, nc;
  
  enum RLE_STATE { INIT, NONE, RLP, RAWP };
  
  int state = INIT;
  
  uint32 size = tga.width * tga.height;
  
#ifdef __BIG_ENDIAN__
  uint16 shortwidth = ( ( tga.width & 0xFF ) << 8 )
    | ( ( tga.width & 0xFF00 ) >> 8 );
  uint16 shortheight = ( ( tga.height & 0xFF ) << 8 )
    | ( ( tga.height & 0xFF00 ) >> 8 );
#else
  uint16 shortwidth = (uint16)tga.width;
  uint16 shortheight = (uint16)tga.height;
#endif
  
  char repcount;
  
  float red, green, blue, alpha;
  
  int idx, row, column;
  
  // have to buffer a whole line for raw packets.
  ubyte * rawbuf = new ubyte[tga.width * 4];  
  
  char id[] = "written with libtarga";
  char idlen = 21;
  char zeroes[5] = { 0, 0, 0, 0, 0 };
  uint32 pixbuf;
  char one = 1;
  char cmap_type = 0;
  char img_type  = 10;  // 2 - uncompressed truecolor  10 - RLE truecolor
  uint16 xorigin  = 0;
  uint16 yorigin  = 0;
  char  pixdepth = 32; // bpp
  char img_desc  = 8;
    
    
  // write id length
  o.write(&idlen, 1);
  
  // write colormap type
  o.write(&cmap_type, 1);
  
  // write image type
  o.write(&img_type, 1);
    
  // write cmap spec.
  o.write(zeroes, 5);
  
  // write image spec.
  o.write((char*)&xorigin, 2);
  o.write((char*)&yorigin, 2);
  o.write((char*)&shortwidth, 2);
  o.write((char*)&shortheight, 2);
  o.write(&pixdepth, 1);
  o.write(&img_desc, 1);
  
  // write image id.
  o.write(id, idlen);
  
  // initial color values -- just to shut up the compiler.
  nc = 0;
  
  // color correction -- data is in RGB, need BGR.
  // also run-length-encoding.
  for( i = 0; i < size; i++ ) {
      
    idx = i * 4;

    row = i / tga.width;
    column = i % tga.width;
    
    // need to un-premultiply alpha.. 

    red     = tga.data[idx] / 255.0f;
    green   = tga.data[idx+1] / 255.0f;
    blue    = tga.data[idx+2] / 255.0f;
    alpha   = tga.data[idx+3] / 255.0f;
    
    if( alpha > 0.0001 ) {
      red /= alpha;
      green /= alpha;
      blue /= alpha;
    }
    
    // clamp to 1.0f 
    
    red = red > 1.0f ? 255.0f : red * 255.0f;
    green = green > 1.0f ? 255.0f : green * 255.0f;
    blue = blue > 1.0f ? 255.0f : blue * 255.0f;
    alpha = alpha > 1.0f ? 255.0f : alpha * 255.0f;
    
#ifdef __BIG_ENDIAN__
    pixbuf = (ubyte)alpha + (((ubyte)red) << 8) + 
      (((ubyte)green) << 16) + (((ubyte)blue) << 24);
#else
    pixbuf = (ubyte)blue + (((ubyte)green) << 8) + 
      (((ubyte)red) << 16) + (((ubyte)alpha) << 24);
#endif
    
    oc = nc;
    
    nc = pixbuf;
    
    
    switch( state ) {

      case INIT:
	// this is just used to make sure we have 2 pixel values to consider.
	state = NONE;
	break;
      

      case NONE:
	
	if ( column == 0 ) {
	  // write a 1 pixel raw packet for the old pixel, then go thru again.
	  repcount = 0;
	  o.write(&repcount, 1);
	  o.write((char*)&oc, 4);
	  state = NONE;
	  break;
	}
	
	if( nc == oc ) {
	  repcount = 0;
	  state = RLP;
	} else {
	  repcount = 0;
	  state = RAWP;
	  for( j = 0; j < 4; j++ ) {
	    rawbuf[(repcount * 4) + j] = *(((ubyte *)(&oc)) + j);
	  }
	}
	break;

	
      case RLP:
	repcount++;
	
	if( column == 0 ) {
	  // finish off rlp.
	  repcount |= 0x80;
	  o.write(&repcount, 1);
	  o.write((char*)&oc, 4);
	  state = NONE;
	  break;
	}
	
	if( repcount == 127 ) {
	  // finish off rlp.
	  repcount |= 0x80;
	  o.write(&repcount, 1);
	  o.write((char*)&oc, 4);
	  state = NONE;
	  break;
	}
	
	if( nc != oc ) {
	  // finish off rlp
	  repcount |= 0x80;
	  o.write( &repcount, 1);
	  o.write((char*)&oc, 4);
	  state = NONE;
	}
	break;
	
      case RAWP:
	repcount++;
	
	if( column == 0 ) {
	  // finish off rawp.
	  for( j = 0; j < 4; j++ ) {
	    rawbuf[(repcount * 4) + j] = *(((ubyte *)(&oc)) + j);
	  }
	  o.write(&repcount, 1);
	  o.write((char*)rawbuf, (repcount + 1) * 4);
	  state = NONE;
	  break;
	}
	
	if( repcount == 127 ) {
	  // finish off rawp.
	  for( j = 0; j < 4; j++ ) {
	    rawbuf[(repcount * 4) + j] = *(((ubyte *)(&oc)) + j);
	  }
	  o.write(&repcount, 1);
	  o.write((char*)rawbuf, (repcount + 1) * 4);
	  state = NONE;
	  break;
	}
	
	if( nc == oc ) {
	  // finish off rawp
	  repcount--;
	  o.write(&repcount, 1);
	  o.write((char*)rawbuf, (repcount + 1) * 4);
	  
	  // start new rlp
	  repcount = 0;
	  state = RLP;
	  break;
	}

	// continue making rawp
	for( j = 0; j < 4; j++ ) {
	  rawbuf[(repcount * 4) + j] = *(((ubyte *)(&oc)) + j);
	}
	
	break;
	
    }
       
    
  }


  // clean up state.

  switch( state ) {

    case INIT:
      break;

    case NONE:
      // write the last 2 pixels in a raw packet.
      o.write(&one, 1);
      o.write((char*)&oc, 4);
      o.write((char*)&nc, 4);
      break;
      
    case RLP:
      repcount++;
      repcount |= 0x80;
      o.write(&repcount, 1);
      o.write((char*)&oc, 4);
      break;

    case RAWP:
      repcount++;
      for( j = 0; j < 4; j++ ) {
	rawbuf[(repcount * 4) + j] = *(((ubyte *)(&oc)) + j);
      }
      o.write(&repcount, 1);
      o.write((char*)rawbuf, (repcount + 1) * 4);
      break;

  }

  delete[] rawbuf;
  
  return o;
}

//**************************************************************************
//
// * Get the RGB value
//==========================================================================
bool SJCTargaImage::
GetRGB(ubyte* data_out, const bool flip)
//==========================================================================
{

  if(!data)
    return false;
  
  // Divide out the alpha
  for ( uint i = 0 ; i < height ; i++ )   {
    int in_offset = i * width * 4;
    int out_offset = flip ? ( height - i - 1 ) * width * 3 : i * width * 3;

    for (uint j = 0 ; j < width ; j++){
      RGBA_To_RGB(data+(in_offset+j*4), data_out+(out_offset+j*3));
    }
  }
  return true;
  
}


//**************************************************************************
//
// * Get the RGBA value
//==========================================================================
bool SJCTargaImage::GetRGBA(ubyte* data_out, const bool flip)
{
  if(!data)
    return false;
  if ( flip )   {
    for ( uint i = 0 ; i < height ; i++ ){
      int in_offset = i * width * 4;
      int out_offset = ( height - i - 1 ) * width * 4;
      
      for (uint j = 0 ; j < width * 4 ; j++) {
	data_out[out_offset+j] = data[in_offset+j];
      }
    }
  }
  else
    memcpy(data_out, data, sizeof(ubyte) * width * height * 4);
  return true;
  
}


//**************************************************************************
//
// * Get the alpha value
//==========================================================================
bool SJCTargaImage::
GetA(ubyte* data_out, const bool flip)
//==========================================================================
{
  if(!data)
    return false;
  for ( uint i = 0 ; i < height ; i++ )    {
    int in_offset = i * width * 4;
    int out_offset = flip ? ( height - i - 1 ) * width : i * width;
    
    for (uint j = 0 ; j < width ; j++){
      data_out[out_offset+j] = data[in_offset+j*4+3];
    }
  }
  return true;
  
}


//**************************************************************************
//
// *
//==========================================================================
void
SJCTargaImage::Grayscale(ubyte *data_out)
//==========================================================================
{
    for (uint i = 0 ; i < width * height ; i++ )
    {
        data_out[i] = (int)floor(data[i*4]*0.299
			       + data[i*4+1]*0.587
			       + data[i*4+2]*0.114);
    }
}


//**************************************************************************
//
// * Composite the current image over the given image.
//==========================================================================
void SJCTargaImage::Comp_Over(SJCTargaImage* pImage)
//==========================================================================
{
  if (!pImage)   {
    throw new SJCException("SJCTargaImage::Comp_Over: null image\n");
  }

  if (width != pImage->width || height != pImage->height)  {
    throw new 
      SJCException("SJCTargaImage::Comp_Over: Images not the same size\n");
  }
  
  for (uint i = 0 ; i < width * height * 4 ; i += 4)  {
    double     f, g;
    int        j;
    
    f = 1.0;
    g = 1.0 - data[i+3] / (double)255;
    
    for (j = 0 ; j < 4 ; j++)     {
	int        val;
	
	val = (int)floor(f * data[i+j] + g * pImage->data[i+j]);
	if (val < 0)
	  data[i+j] = 0;
	else if (val > 255)
	  data[i+j] = 255;
	else
	  data[i+j] = val;
    }
  }
}

//**************************************************************************
//
// * Composite this image "in" the given image
//==========================================================================
void SJCTargaImage::Comp_In(SJCTargaImage* pImage)
//==========================================================================
{
  if (!pImage)   {
    throw new SJCException("SJCTargaImage::Comp_In: null image\n");
  }

  if (width != pImage->width || height != pImage->height)  {
    throw new 
      SJCException("SJCTargaImage::Comp_In: Images not the same size\n");
  }
  
  for (uint i = 0 ; i < width * height * 4 ; i += 4)   {
    double     f, g;
    int        j;
    
    f = pImage->data[i+3] / (double)255;
    g = 0.0;
    
    for (j = 0 ; j < 4 ; j++)       {
      int        val;
      
      val = (int)floor(f * data[i+j] + g * pImage->data[i+j]);
      if (val < 0)
	data[i+j] = 0;
      else if (val > 255)
	data[i+j] = 255;
      else
	data[i+j] = val;
    }
  }
}


//**************************************************************************
//
// * Composite this image "out" the given image.
//==========================================================================
void SJCTargaImage::Comp_Out(SJCTargaImage* pImage)
//==========================================================================
{
  if (!pImage)   {
    throw new SJCException("SJCTargaImage::Comp_Out: null image\n");
  }

  if (width != pImage->width || height != pImage->height)   {
    throw new 
      SJCException("SJCTargaImage::Comp_Out: Images not the same size\n");
  }

  for (uint i = 0 ; i < width * height * 4 ; i += 4)   {
    double        f, g;
    f = 1.0 - pImage->data[i+3] / (double)255;
    g = 0.0;

    for (int j = 0 ; j < 4 ; j++)       {
      int        val;
      
      val = (int)floor(f * data[i+j] + g * pImage->data[i+j]);
      if (val < 0)
	data[i+j] = 0;
      else if (val > 255)
	data[i+j] = 255;
      else
	data[i+j] = val;
    }
  }
}


//**************************************************************************
//
// * Composite current image "atop" given image.
//==========================================================================
void SJCTargaImage::Comp_Atop(SJCTargaImage* pImage)
//==========================================================================
{
  if (!pImage)    {
    throw new SJCException("SJCTargaImage::Comp_Atop: null image\n");
  }

  if (width != pImage->width || height != pImage->height)  {
    throw new 
      SJCException( "SJCTargaImage::Comp_Atop: Images not the same size\n");
  }

  
  for (uint i = 0 ; i < width * height * 4 ; i += 4)  {
    double        f, g;
    
    f = pImage->data[i+3] / (double)255;
    g = 1.0 - data[i+3] / (double)255;
    
    for (int j = 0 ; j < 4 ; j++)       {
      int        val;
      
      val = (int)floor(f * data[i+j] + g * pImage->data[i+j]);
      if (val < 0)
	data[i+j] = 0;
      else if (val > 255)
	data[i+j] = 255;
      else
	data[i+j] = val;
    }
  }
}


//**************************************************************************
//
// * Composite this image with given image using exclusive or (XOR).
//==========================================================================
void SJCTargaImage::Comp_Xor(SJCTargaImage* pImage)
//==========================================================================
{
  if (!pImage)  {
    throw new SJCException("SJCTargaImage::Comp_Xor: null image\n");
  }

  if (width != pImage->width || height != pImage->height)   {
    throw new 
      SJCException("SJCTargaImage::Comp_Xor: Images not the same size\n");
  }
  
  for (uint i = 0 ; i < width * height * 4 ; i += 4)   {
    double        f, g;
    
    f = 1.0 - pImage->data[i+3] / (double)255;
    g = 1.0 - data[i+3] / (double)255;
    
    for (int j = 0 ; j < 4 ; j++)       {
      int        val;
      
      val = (int)floor(f * data[i+j] + g * pImage->data[i+j]);
      if (val < 0)
	data[i+j] = 0;
      else if (val > 255)
	data[i+j] = 255;
      else
	data[i+j] = val;
    }
  }
}


//**************************************************************************
//
// * Calculate the difference bewteen this image and the given one.  Image 
//   dimensions must be equal.
//==========================================================================
void SJCTargaImage::Difference(SJCTargaImage* pImage)
//==========================================================================
{
  if (!pImage)   {
    throw new SJCException("SJCTargaImage::Difference: null image\n");
  }
  
  if (width != pImage->width || height != pImage->height)   {
    throw new 
      SJCException("SJCTargaImage::Difference: Images not the same size\n");
  }
  
  for (uint i = 0 ; i < width * height * 4 ; i += 4)  {
    ubyte        rgb1[3];
    ubyte        rgb2[3];
    
    RGBA_To_RGB(data + i, rgb1);
    RGBA_To_RGB(pImage->data + i, rgb2);
    
    data[i] = abs(rgb1[0] - rgb2[0]);
    data[i+1] = abs(rgb1[1] - rgb2[1]);
    data[i+2] = abs(rgb1[2] - rgb2[2]);
    data[i+3] = 255;
  }
}


//**************************************************************************
//
// * Perform 5x5 box filter on this image
//==========================================================================
void SJCTargaImage::Filter_Box()
//==========================================================================
{
  float   **filter;
  
  filter = new float*[5];
  for (int i = 0 ; i < 5 ; i++)   {
    filter[i] = new float[5];
    for (int j = 0 ; j < 5 ; j++)
      filter[i][j] = 1.f / 25.f;
  }
  
  Filter_Image(filter, 5, 5, true);
  
  for (int i = 0 ; i < 5 ; i++)
    delete[] filter[i];
  delete[] filter;
}


//**************************************************************************
//
// * Perform 5x5 Bartlett filter on this image
//==========================================================================
void SJCTargaImage::Filter_Bartlett()
//==========================================================================
{
  float   **filter;
  double  sum = 0.0;
  
  filter = new float*[5];
  filter[0] = new float[5];
  filter[0][0] = 1.0;
  filter[0][1] = 3.0;
  filter[0][2] = 5.0;
  filter[0][3] = 3.0;
  filter[0][4] = 1.0;
  sum = 13.0;
  for (int i = 1 ; i < 5 ; i++)   {
    filter[i] = new float[5];
    for (int j = 0 ; j < 5 ; j++)     {
      filter[i][j] = filter[0][i] * filter[0][j];
      sum += filter[i][j];
    }
  }
  for (int i = 0 ; i < 5 ; i++)
    for (int j = 0 ; j < 5 ; j++)
      filter[i][j] /= (float)sum;

  Filter_Image(filter, 5, 5, true);
  
  for (int i = 0 ; i < 5 ; i++)
    delete[] filter[i];
  delete[] filter;
}


//**************************************************************************
//
// * Perform 5x5 Gaussian filter on this image
//==========================================================================
void SJCTargaImage::Filter_Gaussian()
//==========================================================================
{
  float   **filter;
  int            i, j;
  double  sum = 0.0;
  
  filter = new float*[5];
  sum = 0.0;
  for (i = 0 ; i < 5 ; i++)   {
    filter[i] = new float[5];
    for (j = 0 ; j < 5 ; j++)     {
      filter[i][j] = (float)(Binomial(4,i) * Binomial(4,j));
      sum += filter[i][j];
    }
  }
  for (i = 0 ; i < 5 ; i++)
    for (j = 0 ; j < 5 ; j++)
      filter[i][j] /= (float)sum;
  
  Filter_Image(filter, 5, 5, true);
  
  for (i = 0 ; i < 5 ; i++)
    delete[] filter[i];
  delete[] filter;
}

//**************************************************************************
//
// * Perform NxN Gaussian filter on this image
//==========================================================================
void SJCTargaImage::Filter_Gaussian_N( unsigned int N )
//==========================================================================
{
  float   **filter;
  int            i, j;
  double  sum = 0.0;
  
  filter = new float*[N];
  sum = 0.0;
  for (i = 0 ; i < (int) N ; i++)  {
    filter[i] = new float[N];
    for (j = 0 ; j < (int) N ; j++)     {
      // without the casting to floats, we have an overflow for
      // large kernels...definitely the wrong behavior
      filter[i][j] = (float) Binomial(N-1,i) * (float) Binomial(N-1,j);
      sum += filter[i][j];
    }
  }
  for (i = 0 ; i < (int) N ; i++) {
    for (j = 0 ; j < (int) N ; j++) {
      filter[i][j] /= (float)sum;
    }
  }

  Filter_Image(filter, N, N, true);
   
  for (i = 0 ; i < (int) N ; i++)
    delete[] filter[i];
  delete[] filter;
}


//**************************************************************************
//
// * Perform 5x5 edge detect (high pass) filter on this image
//==========================================================================
void SJCTargaImage::Filter_Edge()
//==========================================================================
{
  float   **filter;
  double  sum = 0.0;
  
  filter = new float*[5];
  sum = 0.0;
  for (int i = 0 ; i < 5 ; i++)   {
    filter[i] = new float[5];
    for (int j = 0 ; j < 5 ; j++)     {
      filter[i][j] = (float)(Binomial(4,i) * Binomial(4,j));
      sum += filter[i][j];
    }
  }
  for (int i = 0 ; i < 5 ; i++)
    for (int j = 0 ; j < 5 ; j++)       {
      if (i == 2 && j == 2)
	filter[i][j] = 1.f - filter[i][j] / (float)sum;
      else
	filter[i][j] /= (float)-sum;
    }

  Filter_Image(filter, 5, 5, false);

  for (int i = 0 ; i < 5 ; i++)
    delete[] filter[i];
  delete[] filter;
}// Filter_Edge


//**************************************************************************
//
// * Perform a 5x5 enhancement filter to this image
//==========================================================================
void SJCTargaImage::Filter_Enhance()
//==========================================================================
{
    float   **filter;
    double  sum = 0.0;

    filter = new float*[5];
    sum = 0.0;
    for (int i = 0 ; i < 5 ; i++)
    {
        filter[i] = new float[5];
        for (int j = 0 ; j < 5 ; j++)
        {
            filter[i][j] = (float)(Binomial(4,i) * Binomial(4,j));
            sum += filter[i][j];
        }
    }
    for (int i = 0 ; i < 5 ; i++)
        for (int j = 0 ; j < 5 ; j++)
        {
            if (i == 2 && j == 2)
                filter[i][j] = 2.f - filter[i][j] / (float)sum;
            else
                filter[i][j] /= (float)-sum;
        }

    Filter_Image(filter, 5, 5, false);

    for (int i = 0 ; i < 5 ; i++)
        delete[] filter[i];
    delete[] filter;
}// Filter_Enhance

//**************************************************************************
//
// * Scale the image dimensions by the given factor. Return success
//   of operation.
//==========================================================================
void SJCTargaImage::Resize(float scale)
//==========================================================================
{
  ubyte       *new_data;
  int                 new_width, new_height;
  int			i, j;
  
  new_width = (int)floor(width * scale);
  new_height = (int)floor(height * scale);
  new_data = new ubyte[new_width * new_height * 4];
  
  for (i = 0 ; i < new_height ; i++)   {
    float        in_y = i / scale;
    
    for (j = 0 ; j < new_width ; j++)       {
      float   in_x = j / scale;
      
      Reconstruct(in_x, in_y, &(new_data[(i * new_width + j) * 4]));
    }
  }
  
  width = new_width;
  height = new_height;
  delete[] data;
  data = new_data;
}

//**************************************************************************
//
// * Rotate the image clockwise by the given angle.  Do not resize the image.
//==========================================================================
void SJCTargaImage::Rotate(float angleDegrees)
//==========================================================================
{
  float 	    theta = angleDegrees * M_PI / 180.0f;
  float 	    xc = width * 0.5f;
  float 	    yc = height * 0.5f;
  ubyte   *newData = new ubyte[width * height * 4];
  
  for (uint x = 0; x < width; ++x)   {
    for (uint y = 0; y < height; ++y)     {
      // Figure out where we came from
      float xi = (x - xc) * cos (-theta) - (y - yc) * sin (-theta) + xc;
      float yi = (x - xc) * sin (-theta) + (y - yc) * cos (-theta) + yc;
      if ( xi < 0.0f || xi > width - 1.0f  ||
	   yi < 0.0f || yi > height - 1.0f )    {
	newData[((width * y + x) * 4)] = 0;
	newData[((width * y + x) * 4)+1] = 0;
	newData[((width * y + x) * 4)+2] = 0;
	newData[((width * y + x) * 4)+3] = 0;
      }
      else
	Reconstruct(xi, yi, newData + ((width * y + x) * 4));
    }
  }// for
  
  delete[] data;
  data = newData;
}

//**************************************************************************
//
// * Given a single RGBA pixel return the single RGB equivalent composited 
//   with a black background via the second argument
//==========================================================================
void SJCTargaImage::RGBA_To_RGB(ubyte *data, ubyte *rgb)
//==========================================================================
{
  ubyte  alpha = data[3];

  if (alpha == 0)   {
    rgb[0] = BACKGROUND[0];
    rgb[1] = BACKGROUND[1];
    rgb[2] = BACKGROUND[2];
  }
  else   {
    float	alpha_scale = (float)255 / (float)alpha;
    int	val;
    int	i;
    
    for (i = 0 ; i < 3 ; i++)    {
      val = (int)floor(data[i] * alpha_scale);
      if (val < 0)
	rgb[i] = 0;
      else if (val > 255)
	rgb[i] = 255;
      else
		    rgb[i] = val;
    }
  }
}


//**************************************************************************
//
// * Clear the image to all black
//==========================================================================
void SJCTargaImage::ClearToBlack(void)
//==========================================================================
{
  memset(data, 0, width * height * 4);
}

