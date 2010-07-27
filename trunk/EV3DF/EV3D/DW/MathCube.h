//
//
//  Generated by StarUML(tm) C++ Add-In
//
//  @ Project : Untitled
//  @ File Name : MathCube.h
//  @ Date : 2010/3/18
//  @ Author : 
//
//


#if !defined(_MATHCUBE_H)
#define _MATHCUBE_H
#include "../3D/Matrix4x4.h"
#include "ConvStr.h"
#include "Color4.h"
#include "ColorTable.h"
#include "SJCVector3.h"
#include "SJCScalarField3.h"
#include "FlaneMesh.h"
#include "DWHistogram.h"
#include "TriMesh.h"
#include <map>
#include <vector>

#include "wx/wx.h"
class wxTextCtrl;

extern Pos g_Cube[];
extern unsigned short g_QuadCubeIndex[];
extern unsigned short g_QuadXIndex[];
extern unsigned short g_FaceIndex[6][4];
extern float g_Face[];

class MathCube {
public:
	enum AXIS {
		USE_X, USE_Y, USE_Z 
	};
	MathCube();
	~MathCube();
	// init matrixs
	void initWorld();
	// set data to mathcube
	void SetData(SJCScalarField3d* sf3d, int precise, double isolevel);
	// not yet implement start
	void RenderRange(float min, float max);
	void RenderX();
	void RenterFlat(Pos* pos);
	void SetSize(int x, int y, int z);
	// not yet implement end
	void RenderCube();
	void RenderFace(int index);
	void SetProjectionMatrix(Matrix4x4* wm = NULL);
	void SetViewMatrix(Matrix4x4* vm = NULL);
	void SetObjectMatrix(Matrix4x4* dm = NULL);	
	void SetRotate(float x,float y);
	void SetDistance(float z);
	void SetRotateX(float ro);
	void SetRotateY(float ro);
	void SetRotateZ(float ro);
	void SetIdentity();
	void SetScalarX(float ro);
	void SetScalarY(float ro);
	void SetScalarZ(float ro);
	void SetScalar(float x, float y, float z);
	void Resize(int width, int height);
	void RenderStart();
	void RenderAxis();
	void RenderChip(const AXIS index, int persent);
	void SetColorTable(ColorTable* ct);
	ColorTable* GetColorTable() {return m_pCtable;}
	void SetEyeMove(float x, float y, bool leftdown = false);
	void ReleaseResources();
	void DrawMesh();
	void ResetMarchCubeLevel(double isolevel);
	void draw_tstrips();
	void cls();
	void setup_lighting();
	void RenderBondingBox();
private:
	std::vector<double>	m_chipdata[3];
	std::vector<double>	m_facedata[6];
	double		m_persent[3];
	bool		m_faceComputed[6];
	SJCScalarField3d* m_SJCScalarField3d;
	FlaneMesh	m_chipFmesh[3];
	FlaneMesh	m_faceFmesh[6];
	Matrix4x4	m_ProjectionMatrix;
	Matrix4x4	m_ViewMatrix;
	Matrix4x4	m_ObjectMatrix;
	Vector4		m_lookat, m_eye, m_up;
	int		m_size;
	int		m_precise;
	bool		m_init;
	ColorTable*	m_pCtable;
	TriMesh*	m_pTriMesh;
	DWHistogram<double>	m_histogram;
	bool draw_edges;
	bool draw_curv;
	bool draw_2side;
	bool shiny;
	bool lit;
	float clen;
	double m_scalar,m_moveX,m_moveY,m_moveZ;
};

#endif  //_MATHCUBE_H