#include "SEffect.h"
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
	case AXES_TWD97_TO_WGS84:
	case AXES:
		res = SEffect_Sptr(new Axes_Setting);
		break;
	case CLIP_PLANE:
		res = SEffect_Sptr(new ClipPlane_Setting);
		break;
	case RULER:
		res = SEffect_Sptr(new Ruler_Setting);
		break;
	case CLIP_CONTOUR:
		res = SEffect_Sptr(new ClipContour_Setting);
		break;
	case VOLUME_RENDERING:
		res = SEffect_Sptr(new VolumeRender_Setting);
		break;
	}
	res->m_Type = type;
	assert(res.get() != NULL);
	return res;
}
