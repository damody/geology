#pragma once
#include <boost/shared_ptr.hpp>
#include "BoxArea.h"
#include "Color4.h"
#include "TShape.hpp"
#include "ColorTable.h"
#include "SolidDefine.h"
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
	int		m_type;
	BoxArea		m_BoxArea;
	SEffect_Setting* m_Setting;
	virtual ~SEffect(void);
private:
	SEffect(void);
	
};
