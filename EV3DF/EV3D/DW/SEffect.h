#pragma once
#include "BoxArea.h"
#include "Color4.h"
#include "TShape.hpp"
#include "ColorTable.h"
#include "SEffectSet.h"
#include <string>
/**
儲存與設定效果用的類別
*/


class SEffect
{
public:
	enum EffectType
	{
		BOUNDING_BOX,
		VERTEX,
		ISOSURFACE,
		AXES,
		PLANE_CHIP,
		RULER,
		CONTOUR_CHIP,
		VOLUME_RENDER
	};
	int m_type;
	BoxArea m_BoxArea;
	union
	{
		Bounding_Box_Set	*m_BBoxSet;
		Vertex_Set		*m_VertexSet;
		Isosuface_Set		*m_IsosufaceSet;
		Axes_Set		*m_AxesSet;
		Ruler_Set		*m_RulerSet;
		PlaneChip_Set		*m_PlaneChipSet;
		ContourChip_Set		*m_ContourChipSet;
		VolumeRender_Set	*m_VolumeRenderSet;
	};
	SEffect(void);
	~SEffect(void);
};
