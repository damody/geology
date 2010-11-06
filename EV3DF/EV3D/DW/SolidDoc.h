﻿#pragma once
#include "SolidDefine.h"
/**

*/
class SolidDoc
{
public:
	SolidDoc_Sptr	GetParentDoc() const {return m_ParentDoc;}
	SolidCtrl*	GetParentCtrl() const {return m_ParentCtrl;}
	bool		HasPolyData() const;
	bool		HasImageData() const;
	void		RmAllView();
	SolidView_Sptrs	GetViewList() const;
	int		SetPolyData(vtkPolyData_Sptr polyData);
	int		SetImageData(vtkImageData_Sptr imageData);
	int		GridingData(const BoxArea_Sptr area);
private:
	SolidDoc(BoxArea_Sptr area);
	Histogramd		m_histogram;
	SolidDoc_Sptr		m_ParentDoc;
	SolidCtrl		*m_ParentCtrl;
	SolidView_Sptrs		m_SolidViews;	/// 所有相關的views
	vtkPolyData_Sptr	m_PolyData;	/// 離散 data
	vtkImageData_Sptr	m_ImageData;	/// griding data
	BoxArea_Sptr		m_area;		/// 範圍
	vtkAxesActor_Sptr	m_Axes;
	vtkOrientationMarkerWidget_Sptr	m_Axes_widget;
	friend SolidCtrl;
	friend SolidView;
};
