
#pragma once
#include "SolidDefine.h"

/**
顯示單元
*/
class SolidView
{
public:
	enum { CHANGE_OK,      ///< 改變渲染對象成功
		CHANGE_FAIL	///< 改變渲染對象失敗
	};
public:
	int		GetType();
	void		SetVisable(bool show);
	int		GetVisable();
	SolidDoc_Sptr	GetParentDoc()	{ return m_ParentDoc; }
	SolidCtrl	*GetParentCtrl(){ return m_ParentCtrl; }
	void		SetEffect(SEffect_Sptr setting);
	SEffect_Sptr	GetEffect()	{ return m_SEffect; }
	void		Update();
	void		SetColorTable();
	void		Init_BoundingBox();
	void		Init_Vertex();
	void		Init_Contour();
	void		Init_Axes();
	void		Init_ClipPlane();
	void		Init_Ruler();
	void		Init_ClipContour();
	void		Init_VolumeRendering();
private:
	SolidView(SolidCtrl *ParentCtrl, SolidDoc_Sptr Doc);
private:
	SolidDoc_Sptr			m_ParentDoc;
	SolidCtrl			*m_ParentCtrl;
	vtkActor_Sptr			m_actor;
	vtkPolyDataMapper_Sptr		m_polydataMapper;
	SEffect_Sptr			m_SEffect;
	vtkImagePlaneWidget_Sptr	m_ImagePlane;
	vtkContourFilter_Sptr		m_ContourFilter;
	vtkVolume_Sptr			m_volume;
	vtkCubeAxesActor_Sptr		m_CubeAxesActor;
	vtkScalarBarActor_Sptr		m_ScalarBarActor;
private:
	friend	SolidCtrl;
	friend	SolidDoc;
};
