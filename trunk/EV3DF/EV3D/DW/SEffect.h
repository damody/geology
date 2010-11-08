#pragma once
#include <string>
#include "Color4.h"
#include "TShape.hpp"
#include "ColorTable.h"
#include "SolidDefine.h"
#pragma warning(disable:4201)

class SEffect
{
public:
	enum {
		BOUNDING_BOX,
		VERTEX,
		CONTOUR,
		AXES,
		PLANE_CHIP,
		RULER,
		CONTOUR_CHIP,
		VOLUME_RENDERING
	};
public:
	bool GetVisable() {return m_Visable;}
	int  GetType() {return m_Type;}
public:
	static SEffect_Sptr New( int type );
protected:
	bool	m_Visable;	///< 能見度
	int	m_Type;		///< effect種類
	SEffect():m_Visable(true){}
protected:
	friend SolidDoc;
	friend SolidView;
	friend SolidCtrl;
};

struct Bounding_Box_Setting : public SEffect
{
	Bounding_Box_Setting():
	m_Color(Color4(255,255,255,255)),
	m_ThickDegree(1){}
	Color4		m_Color;
	float		m_ThickDegree;
};
struct Vertex_Setting : public SEffect
{
	float		m_MaxValue,
			m_MinValue,
			m_Size;
};
struct Contour_Setting : public SEffect
{
	Contour_Setting():m_ContourValue(0){}
	float		m_ContourValue;
	Color4		m_Color;
};
struct Axes_Setting : public SEffect
{
	Color4		m_XColor,
			m_YColor,
			m_ZColor;
};
struct Ruler_Setting : public SEffect
{
	Color4		m_Color;
	float		m_TargetAxes,
			m_Scalar,
			m_ThickDegree;
	Posf		m_StartPoint,
			m_EndPoint;
};
struct PlaneChip_Setting : public SEffect
{
	PlaneChip_Setting():m_Percent(0),m_Axes(0){}
	int		m_Axes;
	float		m_Percent;
	float		m_Alpha;
};
struct ContourChip_Setting : public SEffect
{
	int		m_Axes;
	float		m_Percent,
			m_ContourValue;
};
struct VolumeRender_Setting : public SEffect
{
	ColorTable	m_ColorTable;
	float		m_MaxValue,
			m_MinValue;
};

