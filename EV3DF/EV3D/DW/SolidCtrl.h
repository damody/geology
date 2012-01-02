// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
#pragma once
#include <vector>
#include <boost/shared_ptr.hpp>
#include "SolidDefine.h"
#include "SJCScalarField3.h"
#include "DWHistogram.h"

/**
control unit
Control each data and view, has origin data
*/
class SolidCtrl
{
public:
	enum InterpolationMethod
	{
		INVERSE_DISTANCE, ///< use interpolate to get grid data
		NEAREST_NEIGHBOR
	};
	SolidCtrl():m_sf3d(NULL)
	{
		m_RenderWindow = vtkSmartNew;
		m_Renderer = vtkSmartNew;
		m_WindowInteractor = vtkOnlyNew;
		m_Camera = vtkSmartNew;
		m_Axes_widget = vtkSmartNew;
		m_Axes = vtkSmartNew;
		m_style = vtkSmartNew;
		m_RenderWindow->AddRenderer(m_Renderer);
		m_WindowInteractor->SetRenderWindow(m_RenderWindow);
		m_WindowInteractor->SetInteractorStyle( m_style );
		m_WindowInteractor->EnableRenderOn();
		m_Renderer->SetActiveCamera(m_Camera);
		m_Renderer->SetBackground(.0, .0, .0);
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
	vtkRenderWindowInteractor_Sptr	m_WindowInteractor;
	vtkOrientationMarkerWidget_Sptr	m_Axes_widget;
	vtkInteractorStyleTrackballCamera_Sptr	m_style;
	SJCScalarField3d	*m_sf3d;
	vtkBounds		m_bounds;
public:
	// set data to use
	int SetGridedData(SJCScalarField3d* sf3d);
	int SetGridedData(vtkImageData_Sptr image);
	int SetGridedData(vtkPolyData_Sptr poly, int nx, int ny, int nz);
	int SetUnGridData(vtkPolyData_Sptr poly, InterpolationMethod method = NEAREST_NEIGHBOR);
	// not good
	// TODO: Remove this function and add General function to add data
	void AddTaiwan();
	void AddTaiwan(char* datafilename, int col, int raw);
	// remove all SEffect View
	void RmAllView();
	// remove view you want
	void RmView(SolidView_Sptr view);
	// remove doc in this ctrl
	void RmDoc(SolidDoc_Sptr doc);
	void ResetViewDirection();
	// render frame
	void Render();
	void SetCamera(vtkCamera_Sptr camera)
	{
		camera->DeepCopy(m_Camera);
	}
	// you need to set hwnd to render on window
	void SetHwnd( HWND hwnd )
	{
		m_RenderWindow->SetParentId(hwnd);
	}
	// if window resize you need to call this function
	void ReSize( int w, int h )
	{
		m_RenderWindow->SetSize(w, h);
	}
	// if your want to add new view by use SEffect
	SolidView_Sptr	NewSEffect(SEffect_Sptr effect);
private:
	SolidDoc_Sptr	NewDoc();
	SolidView_Sptr	NewView(SEffect_Sptr& area, SolidDoc_Sptr& doc);
};
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)