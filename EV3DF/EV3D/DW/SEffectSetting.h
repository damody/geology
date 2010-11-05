#pragma once
#include <string>
#include "Color4.h"
#include "TShape.hpp"
#include "ColorTable.h"
#include "SEffect.h"
#pragma warning(disable:4201)

class SEffect_Setting
{
public:
	SEffect_Setting(std::string name):m_TargetName(name)
	{
	}
	void ChangeTarget(std::string name)
	{
		m_TargetName = name;
	}
	const char* GetTargetName()
	{
		return m_TargetName.c_str();
	}
	bool m_visable;	///< 能見度
	int  m_Type;	///< effect種類

private:
	
	/*! 對象名稱
	*/
	std::string	m_TargetName;
};

struct Bounding_Box_Setting : public SEffect_Setting
{
	Bounding_Box_Setting(std::string name):SEffect_Setting(name),
	m_Color(Color4(255,255,255,255)),
	m_ThickDegree(1){}
	Color4		m_Color;
	float		m_ThickDegree;
};
struct Vertex_Setting : public SEffect_Setting
{
	Vertex_Setting(std::string name):SEffect_Setting(name){}
	float		m_MaxValue,
			m_MinValue,
			m_Size;
};
struct Isosuface_Setting : public SEffect_Setting
{
	Isosuface_Setting(std::string name):SEffect_Setting(name){}
	float		m_ContourValue;
	Color4		m_Color;
};
struct Axes_Setting : public SEffect_Setting
{
	Axes_Setting(std::string name):SEffect_Setting(name){}
	Color4		m_XColor,
			m_YColor,
			m_ZColor;
};
struct Ruler_Setting : public SEffect_Setting
{
	Ruler_Setting(std::string name):SEffect_Setting(name){}
	Color4		m_Color;
	float		m_TargetAxes,
			m_Scalar,
			m_ThickDegree;
	Posf		m_StartPoint,
			m_EndPoint;
};
struct PlaneChip_Setting : public SEffect_Setting
{
	PlaneChip_Setting(std::string name):SEffect_Setting(name){}
	int		m_Axes;
	float		m_Percent;
};
struct ContourChip_Setting : public SEffect_Setting
{
	ContourChip_Setting(std::string name):SEffect_Setting(name){}
	int		m_Axes;
	float		m_Percent,
			m_ContourValue;
};
struct VolumeRender_Setting : public SEffect_Setting
{
	VolumeRender_Setting(std::string name):SEffect_Setting(name){}
	ColorTable	m_ColorTable;
	float		m_MaxValue,
			m_MinValue;
};

