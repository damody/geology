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
	SolidCtrl()
	{
		vtkSmartNew(m_RenderWindow);
		vtkSmartNew(m_Renderer);
		vtkSmartNew(m_WindowInteractor);
		m_RenderWindow->AddRenderer(m_Renderer);
		m_WindowInteractor->SetRenderWindow(m_RenderWindow);
		m_Renderer->SetBackground(.1, .2, .3);
	}
	SolidCtrl(vtkRenderWindow_Sptr rw, vtkRenderWindowInteractor_Sptr iren)
	{
		m_RenderWindow = rw;
		vtkSmartNew(m_WindowInteractor);
		m_WindowInteractor->SetRenderWindow(m_RenderWindow);
	}
	SolidView_Sptrs		m_SolidViewPtrs;
	SolidDoc_Sptrs		m_SolidDocPtrs;
	vtkRenderWindow_Sptr	m_RenderWindow;
	vtkRenderer_Sptr	m_Renderer;
	vtkAxesActor_Sptr	m_Axes;
	vtkPolyData_Sptr	m_polydata;
	vtkImageData_Sptr	m_imagedata;
	vtkRenderWindowInteractor_Sptr	m_WindowInteractor;
	vtkOrientationMarkerWidget_Sptr	m_Axes_widget;
	SJCScalarField3d	*m_sf3d;
public:
	int SetData(SJCScalarField3d* sf3d);
	int RmView(SolidView_Sptr& view);
	int RmDoc(SolidDoc_Sptr& doc);
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
