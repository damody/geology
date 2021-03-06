﻿//
//
//  Generated by StarUML(tm) C++ Add-In
//
//  @ Project : Untitled
//  @ File Name : MathCube.cpp
//  @ Date : 2010/3/18
//  @ Author : 
//
//

#include "MathCube.h"
#include "../Isosurface/MarchCube.h"
#include "../3D/Gut.h"
#include <Windows.h>
#include <gl/gl.h>
#include <gl/glu.h>
#pragma warning(disable:4127) // content error
#pragma warning(disable:4239) // because & is fast

Pos g_Cube[] = {
	{0,0,0},//0 000
	{0,0,1},//1  001
	{0,1,0},//2  010
	{0,1,1},//3   011
	{1,0,0},//4  100
	{1,0,1},//5   101
	{1,1,0},//6   110
	{1,1,1},//7    111
};

unsigned short g_QuadCubeIndex[] = {
	0,1,3,2,
	2,3,7,6,
	6,7,5,4,
	4,5,1,0,
	0,2,6,4,
	1,3,7,5
};

unsigned short g_FaceIndex[6][4] = {
	{1,0,2,3},
	{0,4,6,2},
	{4,5,7,6},
	{5,1,3,7},
	{2,6,7,3},
	{0,4,5,1},
};

unsigned short g_QuadXIndex[] = {
	0,2,7,5,
	4,6,3,1
};

float g_Face[] = {
	0,1,1,1,0,0,1,0,
	0,1,1,1,0,0,1,0,
	0,1,1,1,0,0,1,0,
	0,1,1,1,0,0,1,0
};

MathCube::~MathCube()
{
	ReleaseResources();
	delete m_pCtable;
}

void MathCube::ReleaseResources()
{
	memset(m_persent, INT_MAX, sizeof(double)*3);
	ZeroMemory(m_faceComputed, sizeof(bool)*6);
	if (m_pTriMesh)
		delete m_pTriMesh;
	m_pTriMesh = NULL;
}

MathCube::MathCube() :m_init(false),m_size(NULL),m_precise(0),m_pTriMesh(NULL),m_SJCScalarField3d(NULL)
{
	m_pCtable = new ColorTable();
	ZeroMemory(m_chipdata, sizeof(double*)*3);
	ZeroMemory(m_facedata, sizeof(double*)*6);
	ZeroMemory(m_faceComputed, sizeof(bool)*6);
	memset(m_persent, INT_MAX, sizeof(double)*3);
	m_scalar = 1.0;
	m_moveX = 0;
	m_moveY = 0;
	m_moveZ = 0;
}

void MathCube::initWorld()
{
	m_ObjectMatrix.Identity();
	m_ObjectMatrix.TranslateX(-m_SJCScalarField3d->BoundMaxX()/2);
	m_ObjectMatrix.TranslateY(-m_SJCScalarField3d->BoundMaxY()/2);
	m_ObjectMatrix.TranslateZ(-m_SJCScalarField3d->BoundMaxZ()/2);
	m_eye(100.0f, 100.0f, 100.0f); 
	m_lookat(1.1f, 1.1f, 1.1f);
	m_up(0.0f, 0.0f, 1.0f);
	m_ViewMatrix = GutMatrixLookAtRH(m_eye, m_lookat, m_up);
}

void MathCube::SetColorTable(ColorTable* ct)
{
	if (m_pCtable == NULL && m_pCtable != ct)
		delete m_pCtable;
	m_pCtable = ct;
}

void MathCube::SetProjectionMatrix( Matrix4x4* wm /*= NULL*/ )
{
	if (wm != NULL)
		m_ProjectionMatrix = *wm;
	else
	{
		m_ProjectionMatrix = GutMatrixPerspectiveRH_OpenGL(45.0f, 1.0f, 0.1f, 10000.0f);
		// 閮剖?閬?頧??拚
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf( (float *) &m_ProjectionMatrix);
	}
}

void MathCube::SetViewMatrix(Matrix4x4* vm /*= NULL*/ ) 
{
	if (vm != NULL)
		m_ViewMatrix = *vm;
	else
		m_ViewMatrix = GutMatrixLookAtRH(m_eye, m_lookat, m_up);
}

void MathCube::SetObjectMatrix(Matrix4x4* dm /*= NULL*/ ) 
{
	if (dm != NULL)
		m_ObjectMatrix = *dm;
	else
		m_ObjectMatrix.Identity();
}

void MathCube::SetSize( int x, int y, int z )
{
	x;y;z;
}

void MathCube::SetData( SJCScalarField3d* sf3d, int precise, double isolevel )
{
	// ?脣?鞈???
	m_SJCScalarField3d = sf3d;
	// ?湔isosurface
	if (m_pTriMesh)
		delete m_pTriMesh;
	m_pTriMesh = MarchCubes(sf3d, isolevel);
	if (!m_pTriMesh->faces.empty() || !m_pTriMesh->tstrips.empty())
		m_pTriMesh->need_normals();
	clen = 0.5f * m_pTriMesh->feature_size();
	draw_curv = false;
	m_pTriMesh->need_tstrips();
	// 敺拙?雿宏
	m_ObjectMatrix.TranslateX(-m_moveX);
	m_ObjectMatrix.TranslateY(-m_moveY);
	m_ObjectMatrix.TranslateZ(-m_moveZ);
	// 雿宏?喃葉敹?
	m_moveX = -m_SJCScalarField3d->BoundMaxX()/2;
	m_moveY = -m_SJCScalarField3d->BoundMaxY()/2;
	m_moveZ = -m_SJCScalarField3d->BoundMaxZ()/2;
	m_ObjectMatrix.TranslateX(m_moveX);
	m_ObjectMatrix.TranslateY(m_moveY);
	m_ObjectMatrix.TranslateZ(m_moveZ);
	memset(m_persent, INT_MAX, sizeof(double)*3);
	ZeroMemory(m_faceComputed, sizeof(bool)*6);
	m_precise = precise;
	precise *= precise;	
	// add color
	m_histogram = DWHistogram<double>(m_SJCScalarField3d->begin(), m_SJCScalarField3d->size());
	m_pCtable->clear();
	m_pCtable->push_back(m_histogram.GetPersentValue(1),Color4(255, 0, 0,0));	// 蝝?
	m_pCtable->push_back(m_histogram.GetPersentValue(0.75),Color4(255, 128, 0,0));	// 璈?
	m_pCtable->push_back(m_histogram.GetPersentValue(0.625),Color4(255, 255, 0,0));	// 暺?
	m_pCtable->push_back(m_histogram.GetPersentValue(0.5),Color4(0, 255, 0,0));	// 蝬?
	m_pCtable->push_back(m_histogram.GetPersentValue(0.375),Color4(0, 255, 255,0));	// ??
	m_pCtable->push_back(m_histogram.GetPersentValue(0.25),Color4(0, 0, 255,0));	// ??
	m_pCtable->push_back(m_histogram.GetPersentValue(0.125),Color4(102, 0, 255,0));	// ??
	m_pCtable->push_back(m_histogram.GetPersentValue(0),Color4(167, 87, 168,0));	// 蝝?
	int x = m_SJCScalarField3d->NumX()+1;
	int y = m_SJCScalarField3d->NumY()+1;
	int z = m_SJCScalarField3d->NumZ()+1;
	if (m_chipdata[USE_X].size() < uint(y*z*precise))
	{
		m_chipdata[USE_X].resize(y*z*precise);
		m_facedata[0].resize(y*z*precise);
		m_facedata[1].resize(y*z*precise);
	}
	if (m_chipdata[USE_Y].size() < uint(x*z*precise))
	{
		m_chipdata[USE_Y].resize(x*z*precise);
		m_facedata[2].resize(x*z*precise);
		m_facedata[3].resize(x*z*precise);
	}
	if (m_chipdata[USE_Z].size() < uint(x*y*precise))
	{
		m_chipdata[USE_Z].resize(x*y*precise);
		m_facedata[4].resize(x*y*precise);
		m_facedata[5].resize(x*y*precise);
	}
}

void MathCube::RenderFace(int index)
{
	if (index<1 || index>6) return;
	--index;
	if (m_faceComputed[index] == false) 
	{
		m_faceComputed[index] = true;
		m_faceFmesh[index].SetColorTable(m_pCtable);
		double XL = m_SJCScalarField3d->MaxX() + m_SJCScalarField3d->DX();
		double YL = m_SJCScalarField3d->MaxY() + m_SJCScalarField3d->DY();
		double ZL = m_SJCScalarField3d->MaxZ() + m_SJCScalarField3d->DZ();
		double move;
		switch (index)
		{
		case 0: // X?
		case 1:
			{
				if (index==0)
					move = 0;
				else
					move = XL;
				float w = m_SJCScalarField3d->NumZ();
				float h = m_SJCScalarField3d->NumY();
				Vector4 source(move, 0, 0);
				Vector4 w_length(0, 0, ZL);
				Vector4 h_length(0, YL, 0);
				w*=m_precise;h*=m_precise;
				m_faceFmesh[index].SetSize(w, h);
				m_faceFmesh[index].GenerateGrids(w, h, source, w_length, h_length);
				double dws = ZL / w;
				double dhs = YL / h;
				int count = 0, cw=0, ch=0;
				for (double j = 0;ch <= h;j+=dhs, ch++)
				{
					cw = 0;
					for (double i = 0;cw <= w;i+=dws, cw++)
					{
						m_facedata[index][count++] = m_SJCScalarField3d->Value(move, j, i);
					}
				}
				m_faceFmesh[index].ConvertGetData(&(m_facedata[index][0]), (w+1)*(h+1));
			}
			break;
		case 2: // Y?
		case 3:
			{
				if (index==2)
					move = 0;
				else
					move = YL;
				int w = m_SJCScalarField3d->NumZ();
				int h = m_SJCScalarField3d->NumX();
				Vector4 source(0, move, 0);
				Vector4 w_length(0, 0, ZL);
				Vector4 h_length(XL, 0, 0);
				w*=m_precise;h*=m_precise;
				m_faceFmesh[index].SetSize(w, h);
				m_faceFmesh[index].GenerateGrids(w, h, source, w_length, h_length);				
				double dws = ZL / w;
				double dhs = XL / h;
				int count = 0, cw=0, ch=0;
				for (double j = 0;ch <= h;j+=dhs, ch++)
				{
					cw = 0;
					for (double i = 0;cw <= w;i+=dws, cw++)
					{
						m_facedata[index][count++] = m_SJCScalarField3d->Value(j, move, i);
					}
				}
				m_faceFmesh[index].ConvertGetData(&(m_facedata[index][0]), (w+1)*(h+1));
			}
			break;
		case 4: // Z?
		case 5:
			{
				if (index==4)
					move = 0;
				else
					move = ZL;
				int w = m_SJCScalarField3d->NumX();
				int h = m_SJCScalarField3d->NumY();
				w*=m_precise;h*=m_precise;
				Vector4 source(0, 0,move);
				Vector4 w_length(XL, 0, 0);
				Vector4 h_length(0, YL, 0);
				m_faceFmesh[index].SetSize(w, h);
				m_faceFmesh[index].GenerateGrids(w, h, source, w_length, h_length);
				double dws = XL / w;
				double dhs = YL / h;
				int count = 0, cw=0, ch=0;
				for (double j = 0;ch <= h;j+=dhs, ch++)
				{
					cw = 0;
					for (double i = 0;cw <= w;i+=dws, cw++)
					{
						m_facedata[index][count++] = m_SJCScalarField3d->Value(i, j, move);
					}
				}
				m_faceFmesh[index].ConvertGetData(&(m_facedata[index][0]), (w+1)*(h+1));
			}
			break;
		}
	}

	m_faceFmesh[index].DrawFace();
}
void MathCube::RenderRange(float min, float max) 
{
	min;
	max;
}

void MathCube::RenderX() 
{

}

void MathCube::RenderChip( const AXIS index, int persent)
{
	if (m_persent[index] != persent)
	{
		m_persent[index] = persent;
		m_chipFmesh[index].SetColorTable(m_pCtable);
		double XL = m_SJCScalarField3d->MaxX() + m_SJCScalarField3d->DX();
		double YL = m_SJCScalarField3d->MaxY() + m_SJCScalarField3d->DY();
		double ZL = m_SJCScalarField3d->MaxZ() + m_SJCScalarField3d->DZ();
		switch (index)
		{
		case USE_X: // X?
			{
				double move = XL*persent/1000;
				int w = m_SJCScalarField3d->NumZ();
				int h = m_SJCScalarField3d->NumY();
				w*=m_precise;h*=m_precise;
				Vector4 source(move, 0, 0);
				Vector4 w_length(0, 0, ZL);
				Vector4 h_length(0, YL, 0);
				m_chipFmesh[0].SetSize(w, h);
				m_chipFmesh[0].GenerateGrids(w, h, source, w_length, h_length);
				double dws = ZL / w;
				double dhs = YL / h;
				int count = 0, cw=0, ch=0;
				for (double j = 0;ch <= h;j+=dhs, ch++)
				{
					cw = 0;
					for (double i = 0;cw <= w;i+=dws, cw++, ++count)
					{
						m_chipdata[0][count] = m_SJCScalarField3d->Value(move, j, i);
					}
				}
				m_chipFmesh[0].ConvertGetData(&(m_chipdata[0][0]), (w+1)*(h+1));
			}
			break;
		case USE_Y: // Y?
			{
				double move = YL*persent/1000.0;
				int w = m_SJCScalarField3d->NumZ();
				int h = m_SJCScalarField3d->NumX();
				w*=m_precise;h*=m_precise;
				Vector4 source(0, move, 0);
				Vector4 w_length(0, 0, ZL);
				Vector4 h_length(XL, 0, 0);
				m_chipFmesh[1].SetSize(w, h);
				m_chipFmesh[1].GenerateGrids(w, h, source, w_length, h_length);				
				double dws = ZL / w;
				double dhs = XL / h;
				int count = 0, cw=0, ch=0;
				for (double j = 0;ch <= h;j+=dhs, ch++)
				{
					cw = 0;
					for (double i = 0;cw <= w;i+=dws, cw++)
					{
						m_chipdata[1][count++] = m_SJCScalarField3d->Value(j, move, i);
					}
				}
				m_chipFmesh[1].ConvertGetData(&(m_chipdata[1][0]), (w+1)*(h+1));
			}
			break;
		case USE_Z: // Z?
			{
				double move = ZL*persent/1000.0;
				int w = m_SJCScalarField3d->NumX();
				int h = m_SJCScalarField3d->NumY();
				w*=m_precise;h*=m_precise;
				Vector4 source(0, 0, move);
				Vector4 w_length(XL, 0, 0);
				Vector4 h_length(0, YL, 0);
				m_chipFmesh[2].SetSize(w, h);
				m_chipFmesh[2].GenerateGrids(w, h, source, w_length, h_length);
				double dws = XL / w;
				double dhs = YL / h;
				int count = 0, cw=0, ch=0;
				for (double j = 0;ch <= h;j+=dhs, ch++)
				{
					cw = 0;
					for (double i = 0;cw <= w;i+=dws, cw++)
					{
						m_chipdata[2][count++] = m_SJCScalarField3d->Value(i, j, move);
					}
				}
				m_chipFmesh[2].ConvertGetData(&(m_chipdata[2][0]), (w+1)*(h+1));
			}
			break;
		}
	}
	m_chipFmesh[index].DrawFace();
}

void MathCube::SetRotateX( float ro )
{
	m_ObjectMatrix.RotateX(ro);
}

void MathCube::SetRotateY( float ro )
{
	m_ObjectMatrix.RotateY(ro);
}

void MathCube::SetRotateZ( float ro )
{
	m_ObjectMatrix.RotateZ(ro);
}

void MathCube::RenderCube()
{   
	for (int i=1;i<=6;i++)
		RenderFace(i);
	draw_edges = true;
	draw_2side = true;
	draw_curv = false;
	shiny = true;
	lit = true;
	setup_lighting();
	DrawMesh();
	cls();
}

void MathCube::RenderAxis()
{
	GLUquadricObj *quadObj1 = gluNewQuadric();
	GLUquadricObj *quadObj2 = gluNewQuadric();
	GLUquadricObj *quadObj3 = gluNewQuadric();
	GLubyte draw_list;
	draw_list = glGenLists(1);
	glNewList(draw_list, GL_COMPILE);
	{//?怠???
		gluQuadricDrawStyle(quadObj1,GLU_FILL);
		gluQuadricNormals(quadObj1,GL_FLAT);
		gluQuadricOrientation(quadObj1,GLU_OUTSIDE);
		gluQuadricTexture(quadObj1,GL_FALSE);
		gluCylinder(quadObj1, 2.0f, 0.0f, 10, 15, 5);
		gluQuadricDrawStyle(quadObj2,GLU_FILL);
		gluQuadricNormals(quadObj2,GL_FLAT);
		gluQuadricOrientation(quadObj2,GLU_OUTSIDE);
		gluQuadricTexture(quadObj2,GL_FALSE);
		gluDisk(quadObj2, 0, 2.0f, 15, 5);
		glEndList();
	}
	glPushMatrix();
	glColor3ub(255,0,0);
	glTranslatef(m_SJCScalarField3d->BoundMaxX(),0,0);
	glRotatef(90, 0.0f, 1.0f, 0.0f);
	glCallList(draw_list);
	glPopMatrix();
	glEndList();
	glPushMatrix();
	glColor3ub(0,255,0);
	glTranslatef(0,m_SJCScalarField3d->BoundMaxY(),0);
	glRotatef(-90, 1.0f, 0.0f, 0.0f);
	glCallList(draw_list);
	glPopMatrix();
	glEndList();
	glPushMatrix();
	glColor3ub(0,0,255);
	glTranslatef(0,0,m_SJCScalarField3d->BoundMaxZ());
	glRotatef(90, 0.0f, 0.0f, 1.0f);
	glCallList(draw_list);
	glPopMatrix();
	draw_list = glGenLists(1);
	glNewList(draw_list, GL_COMPILE);
	{//?怠???
		gluQuadricDrawStyle(quadObj3,GLU_FILL);
		gluQuadricNormals(quadObj3,GL_FLAT);
		gluQuadricOrientation(quadObj3,GLU_OUTSIDE);
		gluQuadricTexture(quadObj3,GL_FALSE);
		gluCylinder(quadObj3, 1.0f, 1.0f, 1.0f, 15, 5);
		gluDeleteQuadric(quadObj1);
		gluDeleteQuadric(quadObj2);
		gluDeleteQuadric(quadObj3);
	}
	glEndList();
	glPushMatrix();
	glColor3ub(255,0,0);
	glScalef(m_SJCScalarField3d->BoundMaxX(),1,1);
	glRotatef(90, 0.0f, 1.0f, 0.0f);
	glCallList(draw_list);
	glPopMatrix();
	glEndList();
	glPushMatrix();
	glColor3ub(0,255,0);
	glScalef(1,m_SJCScalarField3d->BoundMaxY(),1);
	glRotatef(-90, 1.0f, 0.0f, 0.0f);
	glCallList(draw_list);
	glPopMatrix();
	glEndList();
	glPushMatrix();
	glColor3ub(0,0,255);
	glScalef(1,1,m_SJCScalarField3d->BoundMaxZ());
	glRotatef(90, 0.0f, 0.0f, 1.0f);
	glCallList(draw_list);
	glPopMatrix();
}

void MathCube::SetRotate( float x,float y )
{
	float rz = x * 0.01f;
	float rx = y * -0.01f;
	::Matrix4x4 rotation_matrix;
	rotation_matrix.RotateZ_Replace(rz);
	rotation_matrix.RotateX(rx);
	m_ObjectMatrix = m_ObjectMatrix * rotation_matrix;
}

void MathCube::SetDistance( float dis )
{
	m_ObjectMatrix.Scale(1/m_scalar, 1/m_scalar, 1/m_scalar);
	m_ObjectMatrix.TranslateX(-m_moveX*m_scalar);
	m_ObjectMatrix.TranslateY(-m_moveY*m_scalar);
	m_ObjectMatrix.TranslateZ(-m_moveZ*m_scalar);
	m_scalar += dis * 0.001f;
	float x = m_scalar*m_moveX;
	float y = m_scalar*m_moveY;
	float z = m_scalar*m_moveZ;
	m_ObjectMatrix.TranslateX(x);
	m_ObjectMatrix.TranslateY(y);
	m_ObjectMatrix.TranslateZ(z);
	m_ObjectMatrix.Scale(m_scalar, m_scalar, m_scalar);
	//m_ObjectMatrix.Scale_Replace(scale, scale, scale);
	
	// 	m_ObjectMatrix.Translate_Replace(m_SJCScalarField3d->BoundMaxX()/2*scale,
	// 		m_SJCScalarField3d->BoundMaxY()/2*scale,
	// 		m_SJCScalarField3d->BoundMaxZ()/2*scale);
}

void MathCube::Resize( int width, int height )
{
	// 雿輻?啁?閬?憭批???啁?蝜芸?閫??摨?
	glViewport(0, 0, width, height);
	// ?蔣?拚, ?身瘞游像頝??湔??閬?.
	float aspect = (float) height / (float) width;
	Matrix4x4 projection_matrix = GutMatrixPerspectiveRH_OpenGL(45.0f, aspect, 0.1f, 10000.0f);
	// 閮剖?閬?頧??拚
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf( (float *) &projection_matrix);
}

void MathCube::RenderStart()
{
	// 皜?恍
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glTexParameteri( GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glEnable(GL_TEXTURE_2D);
	// 閮剖?閬??孱L_MODELVIEW?拚
	glMatrixMode(GL_MODELVIEW);
	// 閮剖?頧??拚

	Matrix4x4 world_view_matrix = m_ObjectMatrix * m_ViewMatrix;
	glLoadMatrixf( (float *) &world_view_matrix);
}

void MathCube::RenterFlat( Pos* pos )
{
	pos;
}

// int MathCube::Load2Dface(int index, int w,int h, double* data )
// {
// 	isTextureOK = true;
// 	Color4* colorAry = new Color4[w*h];
// 	for (int i=0;i<w;i++)
// 		for (int j=0;j<h;j++)
// 			colorAry[i*h+j] = m_Ctable.GetColor4((float)data[i*h+j]);
// 	m_tex = new DW2DTexture(w,h);
// 	m_tex->WriteIn(GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*)colorAry);
// 	return 0;
// }

void MathCube::SetIdentity()
{
	m_ObjectMatrix.Identity();
}

void MathCube::SetScalarX( float ro )
{
	m_ObjectMatrix.Scale(ro,1,1);
}

void MathCube::SetScalarY( float ro )
{
	m_ObjectMatrix.Scale(1,ro,1);
}

void MathCube::SetScalarZ( float ro )
{
	m_ObjectMatrix.Scale(1,1,ro);
}

void MathCube::SetScalar( float x, float y, float z )
{
	m_ObjectMatrix.Scale(x,y,z);
}

void MathCube::SetEyeMove( float ix, float iy, bool leftdown )
{
	Vector4 move(0,0,0);
	if (leftdown)
		move[0] -= ix;
	else
		move[1] -= ix;
	move[2] -= iy;
	m_eye += move;
	m_lookat += move;
	m_ViewMatrix = GutMatrixLookAtRH(m_eye, m_lookat, m_up);
}

// Draw the mesh
void MathCube::DrawMesh()
{
	if (m_pTriMesh->vertices.size()<1) return;
	glPushMatrix();

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);

	if (draw_2side) {
		glDisable(GL_CULL_FACE);
	} else {
		glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);
	}

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT,
		sizeof(m_pTriMesh->vertices[0]),
		&m_pTriMesh->vertices[0][0]);
	if (!m_pTriMesh->normals.empty()) {
		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT,
			sizeof(m_pTriMesh->normals[0]),
			&m_pTriMesh->normals[0][0]);
	} else {
		glDisableClientState(GL_NORMAL_ARRAY);
	}

	if (!m_pTriMesh->colors.empty()) {
		glEnableClientState(GL_COLOR_ARRAY);
		glColorPointer(3, GL_UNSIGNED_BYTE,
			sizeof(m_pTriMesh->colors[0]),
			&m_pTriMesh->colors[0][0]);
		glEnable(GL_COLOR_MATERIAL);
	} else {
		glDisableClientState(GL_COLOR_ARRAY);
		glDisable(GL_COLOR_MATERIAL);
	}

	if (m_pTriMesh->tstrips.empty()) {
		// No triangles - draw as points
		glPointSize(1);
		glDrawArrays(GL_POINTS, 0, m_pTriMesh->vertices.size());
		glPopMatrix();
		return;
	}

	if (draw_edges || draw_curv) {
		glPolygonOffset(10.0f, 10.0f);
		glEnable(GL_POLYGON_OFFSET_FILL);
	}

	draw_tstrips();

	glDisable(GL_POLYGON_OFFSET_FILL);
	if (draw_edges) {
		glPolygonMode(GL_FRONT, GL_LINE);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisable(GL_COLOR_MATERIAL);
		GLfloat global_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
		GLfloat light0_diffuse[] = { 0.8, 0.8, 0.8, 0.0 };
		GLfloat light1_diffuse[] = { -0.2, -0.2, -0.2, 0.0 };
		GLfloat light0_specular[] = { 0.0f, 0.0f, 0.0f, 0.0f };
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
		glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
		glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
		GLfloat mat_diffuse[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat_diffuse);
		glColor3f(0, 0, 1); // Used iff unlit
		draw_tstrips();
		glPolygonMode(GL_FRONT, GL_FILL);
	}

	if (draw_curv) {
		glDisable(GL_LIGHTING);
		m_pTriMesh->need_curvatures();
		int nv = m_pTriMesh->vertices.size();
		glColor3f(1,0,0);
		glBegin(GL_LINES);
		int i;
		for (i = 0; i < nv; i++) {
			glVertex3fv(m_pTriMesh->vertices[i]);
			glVertex3fv(m_pTriMesh->vertices[i] +
				clen * m_pTriMesh->normals[i]);
		}
		glColor3f(0.7,0.7,0);
		for (i = 0; i < nv; i++) {
			glVertex3fv(m_pTriMesh->vertices[i] -
				clen * m_pTriMesh->pdir1[i]);
			glVertex3fv(m_pTriMesh->vertices[i] +
				clen * m_pTriMesh->pdir1[i]);
		}
		glColor3f(0,1,0);
		for (i = 0; i < nv; i++) {
			glVertex3fv(m_pTriMesh->vertices[i] -
				clen * m_pTriMesh->pdir2[i]);
			glVertex3fv(m_pTriMesh->vertices[i] +
				clen * m_pTriMesh->pdir2[i]);
		}
		glEnd();
		//glDrawArrays(GL_POINTS, 0, nv);
	}

	glPopMatrix();
}

// Draw triangle strips.  They are stored as length followed by values.
void MathCube::draw_tstrips()
{
	const int *t = &m_pTriMesh->tstrips[0];
	const int *end = t + m_pTriMesh->tstrips.size();
	while (likely(t < end)) {
		int striplen = *t++;
		glDrawElements(GL_TRIANGLE_STRIP, striplen, GL_UNSIGNED_INT, t);
		t += striplen;
	}
}

// Clear the screen
void MathCube::cls()
{
	glDisable(GL_DITHER);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_NORMALIZE);
	glDisable(GL_LIGHTING);
	glDisable(GL_NORMALIZE);
	glDisable(GL_COLOR_MATERIAL);
	// 	glClearColor(0.08, 0.08, 0.08, 0);
	// 	glClearDepth(1);
	// 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

}


// Set up lights and materials
void MathCube::setup_lighting()
{
	if (!lit || m_pTriMesh->normals.empty()) {
		glDisable(GL_LIGHTING);
		glColor3f(0.85, 0.85, 0.85);
		return;
	}

	GLfloat mat_diffuse[4] = { 1, 1, 1, 1 };
	GLfloat mat_specular[4] = { 0.18, 0.18, 0.18, 0.18 };
	if (!shiny) {
		mat_specular[0] = mat_specular[1] =
			mat_specular[2] = mat_specular[3] = 0.0f;
	}
	GLfloat mat_shininess[] = { 64 };
	GLfloat global_ambient[] = { 0.02, 0.02, 0.05, 0.05 };
	GLfloat light0_ambient[] = { 1, 1, 1, 1 };
	GLfloat light0_diffuse[] = { 0.85, 0.85, 0.8, 0.85 };
	GLfloat light1_diffuse[] = { -0.01, -0.01, -0.03, -0.03 };
	GLfloat light0_specular[] = { 0.85, 0.85, 0.85, 0.85 };
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, draw_2side);
	if (!m_pTriMesh->colors.empty()) {
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glEnable(GL_COLOR_MATERIAL);
	}
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glEnable(GL_NORMALIZE);
}

void MathCube::ResetMarchCubeLevel( double isolevel )
{
	if (m_SJCScalarField3d == NULL) return;
	if (m_pTriMesh)
		delete m_pTriMesh;
	m_pTriMesh = MarchCubes(m_SJCScalarField3d, isolevel);
	if (!m_pTriMesh->faces.empty() || !m_pTriMesh->tstrips.empty())
		m_pTriMesh->need_normals();
	clen = 0.5f * m_pTriMesh->feature_size();
	draw_curv = false;
	m_pTriMesh->need_tstrips();
}

void MathCube::RenderBondingBox()
{
	float maxX = (float)m_SJCScalarField3d->BoundMaxX();
	float maxY = (float)m_SJCScalarField3d->BoundMaxY();
	float maxZ = (float)m_SJCScalarField3d->BoundMaxZ();
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glBegin(GL_LINES);
	glColor3ub(255,255,255);
	glVertex3f(0,0,0);glVertex3f(maxX,0,0);
	glVertex3f(maxX,0,0);glVertex3f(maxX,maxY,0);
	glVertex3f(maxX,maxY,0);glVertex3f(0,maxY,0);
	glVertex3f(0,maxY,0);glVertex3f(0,0,0);
	glVertex3f(0,maxY,0);glVertex3f(0,maxY,maxZ);
	glVertex3f(0,maxY,maxZ);glVertex3f(0,0,maxZ);
	glVertex3f(0,0,maxZ);glVertex3f(0,0,0);
	glVertex3f(0,0,0);glVertex3f(0,0,maxZ);
	glVertex3f(0,0,maxZ);glVertex3f(maxX,0,maxZ);
	glVertex3f(maxX,0,maxZ);glVertex3f(maxX,maxY,maxZ);
	glVertex3f(maxX,maxY,maxZ);glVertex3f(0,maxY,maxZ);
	glVertex3f(0,maxY,maxZ);glVertex3f(0,0,maxZ);
	glVertex3f(maxX,0,0);glVertex3f(maxX,0,maxZ);
	glVertex3f(0,maxY,0);glVertex3f(0,maxY,maxZ);
	glVertex3f(maxX,maxY,0);glVertex3f(maxX,maxY,maxZ);
	glEnd();
}
