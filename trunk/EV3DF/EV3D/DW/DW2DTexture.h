﻿//
//
//  Generated by StarUML(tm) C++ Add-In
//
//  @ Project : Untitled
//  @ File Name : Textrue.h
//  @ Date : 2010/3/25
//  @ Author : 
//
//


#if !defined(_TEXTRUE_H)
#define _TEXTRUE_H
#include <windows.h>
#include <gl/gl.h>

class DW2DTexture {
public:
	enum {TEXTURE_OK, SIZE_ERROR, TEXTURE_ERROR};
	DW2DTexture():w(0),h(0){}
	DW2DTexture(int iw,int ih):w(iw),h(ih){}
	int w,h;
	int WriteIn(int gl_format, int gl_datatype, unsigned char* pData);
	GLuint* UseGLTexture();
private:
	GLuint m_gltexture;
};

#endif  //_TEXTRUE_H
