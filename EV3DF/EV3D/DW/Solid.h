#pragma once

#include "SJCVector3.h"
#include "SJCScalarField3.h"
#include "ColorTable.h"
#include "DWHistogram.h"
#include <windows.h>

#include <vtkFloatArray.h>
#include <vtkCellData.h>
#include <vtkScalarsToColors.h>
#include <vtkLookupTable.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkContourFilter.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkOutlineFilter.h>
#include <vtkImagePlaneWidget.h>
#include <vtkCamera.h>
#include <vtkImageShiftScale.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkProperty.h>
#include <vtkActor.h>
#include "SEffect.h"

// use vtk to implement function like mathcube
/**
資料單元
*/

class Solid
{
public:
	enum AXIS {
		USE_X, USE_Y, USE_Z 
	};
	Solid();
	~Solid();
	void SetHwnd(HWND hwnd);
	// set data to mathcube
	void SetData(SJCScalarField3d* sf3d);
	void AddEffect();
	void SetIsoSurface(double isolevel, bool show = true);
	void SetSlice(AXIS axes, double percent);
	void EnableSlice(AXIS axes);
	void DisableSlice(AXIS axes);
	void SetVertex(bool show = true);
	void SetVolume();
	void ReSize(int w, int h);
	void Render();
	void SetColorTable(ColorTable* ct);
	ColorTable* GetColorTable() {return m_pCtable;}
private:
	ColorTable*	m_pCtable;
	SJCScalarField3d* m_SJCScalarField3d;
	DWHistogram<double>	m_histogram;
	// vtk classm
	vtkSmartPointer<vtkPolyData>		m_polydata; // save orgin data
	vtkSmartPointer<vtkImageData>		m_ImageData, m_vol; // save orgin data
	vtkSmartPointer<vtkOrientationMarkerWidget> m_AxesWidget; // axes on this widget
	vtkSmartPointer<vtkRenderer>		m_Renderer; // vtk use this to render
	vtkSmartPointer<vtkRenderWindow>	m_RenderWindow; // use by flow
	vtkSmartPointer<vtkRenderWindowInteractor> m_iren; // use by flow
	vtkSmartPointer<vtkVertexGlyphFilter>	m_VertexFilter; // show vertex
	vtkSmartPointer<vtkContourFilter>	m_contour; // show isosurface
	vtkSmartPointer<vtkPolyDataMapper>	m_contour_mapper, m_vertex_mapper, m_outlineMapper;
	vtkSmartPointer<vtkActor>		m_contour_actor, m_vertex_actor, m_outlineActor;
	vtkSmartPointer<vtkLookupTable>		m_lut, m_chiplut;
	vtkSmartPointer<vtkSmartVolumeMapper>	m_volumeMapper;
	vtkSmartPointer<vtkVolumeProperty>	m_volumeProperty;
	vtkSmartPointer<vtkVolume>		m_volume;
	vtkSmartPointer<vtkPiecewiseFunction>	m_CompositeOpacity;
	vtkSmartPointer<vtkColorTransferFunction> m_ColorTransferFun;
	vtkSmartPointer<vtkOutlineFilter>	m_outline;
	vtkSmartPointer<vtkCamera>		m_camera;
	vtkSmartPointer<vtkOrientationMarkerWidget> m_axes_widget;
	vtkSmartPointer<vtkAxesActor>		m_axes; // to show axes
	vtkSmartPointer<vtkImagePlaneWidget>	m_planeWidgetX, m_planeWidgetY, m_planeWidgetZ;
	vtkSmartPointer<vtkFloatArray>		m_volcolors;
	vtkSmartPointer<vtkPoints>		m_points;
	vtkSmartPointer<vtkUnsignedCharArray>	m_colors;
	vtkSmartPointer<vtkImageShiftScale>	m_ImageShiftScale;

};
