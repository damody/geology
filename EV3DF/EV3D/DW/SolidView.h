#pragma once
#include "SolidDefine.h"
/**
顯示單元
*/
class SolidView
{
public:
	enum {
		CHANGE_OK,	///< 改變渲染對象成功
		CHANGE_FAIL	///< 改變渲染對象失敗
	};
	
public:	
	int		GetType();
	void		SetVisable(bool show);
	int		GetVisable();
	SolidDoc_Sptr	GetParentDoc(){return m_ParentDoc;}
	SolidCtrl*	GetParentCtrl(){return m_ParentCtrl;}
	void		SetEffect(SEffect_Sptr setting);
	void		Update();
	void		SetColorTable();
private:
	SolidView(SolidCtrl *ParentCtrl, SolidDoc_Sptr Doc);
private:
	SolidDoc_Sptr		m_ParentDoc;
	SolidCtrl		*m_ParentCtrl;
	vtkActor_Sptr		m_actor;
	vtkPolyDataMapper_Sptr	m_polydataMapper;
	SEffect_Sptr		m_SEffect;
	vtkCamera_Sptr		m_Camera;
	vtkImagePlaneWidget_Sptr m_ImagePlane;
private:	
	friend SolidCtrl;
	friend SolidDoc;
};
