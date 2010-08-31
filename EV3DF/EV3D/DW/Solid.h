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

// use vtk to implement function like mathcube

class Solid
{
public:
	enum AXIS {
		USE_X, USE_Y, USE_Z 
	};
	Solid();
	void SetHwnd(HWND hwnd);
	// set data to mathcube
	void SetData(SJCScalarField3d* sf3d);
	void SetIsoSurface(double isolevel, bool show = true);
	void SetVertex(bool show = true);
	void ReSize(int w, int h);
	void Render();
	void SetColorTable(ColorTable* ct);
	ColorTable* GetColorTable() {return m_pCtable;}
private:
	ColorTable*	m_pCtable;
	SJCScalarField3d* m_SJCScalarField3d;
	vtkSmartPointer<vtkPolyData> m_polydata; // save orgin data
	vtkSmartPointer<vtkImageData> m_ImageData; // save orgin data
	vtkSmartPointer<vtkAxesActor> axes; // to show axes
	vtkSmartPointer<vtkOrientationMarkerWidget> m_AxesWidget; // axes on this widget
	vtkSmartPointer<vtkRenderer> m_Renderer; // vtk use this to render
	vtkSmartPointer<vtkRenderWindow> m_RenderWindow; // use by flow
	vtkSmartPointer<vtkRenderWindowInteractor> m_iren; // use by flow
	vtkSmartPointer<vtkVertexGlyphFilter> m_VertexFilter; // show vertex
	vtkSmartPointer<vtkContourFilter> m_contour; // show isosurface
	vtkSmartPointer<vtkPolyDataMapper> m_contour_mapper, m_vertex_mapper;
	vtkSmartPointer<vtkActor> m_contour_actor, m_vertex_actor;
	vtkSmartPointer<vtkLookupTable> m_lut;
	DWHistogram<double>	m_histogram;
};