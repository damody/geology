﻿//
//
//  Generated by StarUML(tm) C++ Add-In
//
//  @ Project : Untitled
//  @ File Name : FlaneMesh.h
//  @ Date : 2010/4/10
//  @ Author : 
//
//


#if !defined(_FLANEMESH_H)
#define _FLANEMESH_H
#include <windows.h>
#include <gl/GL.h>
#include "ColorTable.h"
#include "Color4.h"
#include "../3D/Vector4.h"
#include <cassert>
#include <vector>

class FlaneMesh {
public:
	enum {USE_TEXTURE=1, USE_VERTEX};
	FlaneMesh():m_pCt(NULL),
		m_pColor(NULL),
		m_pVertexData(NULL),
		m_psIndices(NULL),
		m_DrawType(USE_VERTEX){}
	~FlaneMesh();
	void SetSize(int iw, int ih);
	void SetColorTable(ColorTable* ct);
	ColorTable* GetColorTablbe();
	void DrawFace();
	void SetDrawMethod(int enumx);
/*
//source m_p[0] |----------| m_p[1]
//              |          |
//              |          |
//       m_p[2] |----------| m_p[3]
*/
	bool GenerateGrids(int x_grids, int y_grids, Vector4& source, Vector4& w_step, Vector4& h_step);
	bool ConvertGetData(double* pd, int num = 0);
	bool DataToColor(ColorTable* pct, double* pd, Color4* pc4, int num);
private:
	Vector4		m_p[4];
	int		m_DrawType;
	std::vector<unsigned char>	m_pColor;
	std::vector<float>		m_pVertexData;
	std::vector<unsigned int>	m_psIndices;
	ColorTable*	m_pCt;
	int m_w, m_h, m_num_vertices, m_num_indices, m_num_triangles;
};

#endif  //_FLANEMESH_H
