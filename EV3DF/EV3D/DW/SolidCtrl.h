﻿#pragma once
#include <vector>
#include <boost/shared_ptr.hpp>
#include "BoxArea.h"
#include "SolidDefine.h"
#include "SJCScalarField3.h"
#include "DWHistogram.h"
/**
control unit
*/
class SolidCtrl
{
public:
	enum InterpolationMethod
	{
		INVERSE,
		NEAREST
	};
	SolidCtrl()
	{
		m_RenderWindow = vtkSmartNew;
		m_Renderer = vtkSmartNew;
		m_WindowInteractor = vtkOnlyNew;
		m_Camera = vtkSmartNew;
		m_Axes_widget = vtkSmartNew;
		m_Axes = vtkSmartNew;
		m_RenderWindow->AddRenderer(m_Renderer);
		m_WindowInteractor->SetRenderWindow(m_RenderWindow);
		m_Renderer->SetActiveCamera(m_Camera);
		m_Renderer->SetBackground(.1, .2, .3);
		m_Axes_widget->SetOutlineColor( 0.8300, 0.6700, 0.5300 );
		m_Axes_widget->SetOrientationMarker( m_Axes );
		m_Axes_widget->SetInteractor( m_WindowInteractor );
		m_Axes_widget->On();
	}
// Solid需要
// 	SolidCtrl(vtkRenderWindow_Sptr rw, vtkRenderWindowInteractor_Sptr iren)
// 	{
// 		m_RenderWindow = rw;
// 		m_WindowInteractor);
// 		m_WindowInteractor->SetRenderWindow(m_RenderWindow);
// 	}
	SolidView_Sptrs		m_SolidViewPtrs;
	SolidDoc_Sptrs		m_SolidDocPtrs;
	vtkRenderWindow_Sptr	m_RenderWindow;
	vtkRenderer_Sptr	m_Renderer;
	vtkAxesActor_Sptr	m_Axes;
	vtkPolyData_Sptr	m_polydata;
	vtkImageData_Sptr	m_imagedata;
	vtkCamera_Sptr		m_Camera;
	BoxArea_Sptr		m_area;
	vtkRenderWindowInteractor_Sptr	m_WindowInteractor;
	vtkOrientationMarkerWidget_Sptr	m_Axes_widget;
	SJCScalarField3d	*m_sf3d;
public:
	int SetData(SJCScalarField3d* sf3d, InterpolationMethod method = NEAREST);
	void RmAllView();
	void RmView(SolidView_Sptr view);
	void RmDoc(SolidDoc_Sptr doc);
	void ReSetViewDirection();
	void Render()
	{
		m_RenderWindow->Render();
	}
	void SetHwnd( HWND hwnd )
	{
		m_RenderWindow->SetParentId(hwnd);
	}
	void ReSize( int w, int h )
	{
		m_RenderWindow->SetSize(w, h);
	}
	SolidView_Sptr	NewSEffect(SEffect_Sptr effect);
private:
	SolidDoc_Sptr	NewDoc();
	SolidView_Sptr	NewView(SEffect_Sptr& area, SolidDoc_Sptr& doc);
};
