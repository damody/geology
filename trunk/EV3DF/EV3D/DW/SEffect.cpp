﻿#include "SEffect.h"
#include <cassert>

SEffect_Sptr SEffect::New( int type )
{
	SEffect_Sptr res;
	switch (type)
	{
	case BOUNDING_BOX:
		res = SEffect_Sptr(new Bounding_Box_Setting);
		break;
	case VERTEX:
		res = SEffect_Sptr(new Vertex_Setting);
		break;
	case CONTOUR:
		res = SEffect_Sptr(new Contour_Setting);
		break;
	case AXES:
		res = SEffect_Sptr(new Axes_Setting);
		break;
	case PLANE_CHIP:
		res = SEffect_Sptr(new PlaneChip_Setting);
		break;
	case RULER:
		res = SEffect_Sptr(new Ruler_Setting);
		break;
	case CONTOUR_CHIP:
		res = SEffect_Sptr(new ContourChip_Setting);
		break;
	case VOLUME_RENDERING:
		res = SEffect_Sptr(new VolumeRender_Setting);
		break;
	}
	res->m_Type = type;
	assert(res.get() != NULL);
	return res;
}