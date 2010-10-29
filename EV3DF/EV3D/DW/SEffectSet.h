#pragma once
#include <string>
#include "Color4.h"
#include "TShape.hpp"
#include "ColorTable.h"
#pragma warning(disable:4201)

struct Bounding_Box_Set
{
	std::string	m_TargetName;
	Color4		m_Color;
	float		m_ThickDegree;
};
struct Vertex_Set
{
	std::string	m_TargetName;
	float		m_MaxValue,
			m_MinValue,
			m_Size;
};
struct Isosuface_Set
{
	std::string	m_TargetName;
	float		m_ContourValue;
	Color4		m_Color;
};
struct Axes_Set
{
	std::string	m_TargetName;
	Color4		m_XColor,
			m_YColor,
			m_ZColor;
};
struct Ruler_Set
{
	std::string	m_TargetName;
	Color4		m_Color;
	float		m_TargetAxes,
			m_Scalar,
			m_ThickDegree;
	Posf		m_StartPoint,
			m_EndPoint;
};
struct PlaneChip_Set
{
	std::string	m_TargetName;
	int		m_Axes;
	float		m_Percent;
};
struct ContourChip_Set
{
	std::string	m_TargetName;
	int		m_Axes;
	float		m_Percent,
			m_ContourValue;
};
struct VolumeRender_Set
{
	std::string	m_TargetName;
	ColorTable	m_ColorTable;
	float		m_MaxValue,
			m_MinValue;
};

