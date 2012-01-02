// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
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
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
