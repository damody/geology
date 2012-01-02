// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
#pragma once
#include "SolidDefine.h"

/**
顯示單元
*/
class SolidView
{
public:
	enum { 
		CHANGE_OK,      ///< 改變渲染對象成功
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
	// update information to vtk view
	void		Update();
	// TODO: you need to Implement this for each SEffect
	void		SetColorTable();
	// initialize different SEffect
	void		Init_BoundingBox();
	void		Init_Vertex();
	void		Init_Contour();
	// TODO: this function is not good for general data
	void		Init_Axes();
	// TODO: this function is not good for general data
	void		Init_Axes_TWD97_TO_WGS84();
	void		Init_ClipPlane();
	// TODO: you need to Implement
	void		Init_Ruler();
	// TODO: you need to Implement
	void		Init_ClipContour();
	void		Init_VolumeRendering();
private:
	// can't create by user
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
	vtkColorTransferFunction*	GetColorTable();
private:
	friend	SolidCtrl;
	friend	SolidDoc;
};
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)