#pragma once
#include <cassert>
#include <windows.h>
#include <tchar.h>
#define MESSAGE(x) MessageBox(NULL, _T(x), _T("MESSAGE"), 0);
#define FFassert(crash, res) if (crash) \
	assert(res); \
	else \
	MESSAGE(res);

#include <vtkSmartPointer.h>
#include <vtkAppendPolyData.h>
#include <vtkLine.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkScalarsToColors.h>
#include <vtkLookupTable.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPointData.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkContourFilter.h>
#include <vtkSmartVolumeMapper.h>
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

#define VTK_SMART_POINTER(x) \
	typedef vtkSmartPointer< x >	x##_Sptr; \
	typedef std::vector< x##_Sptr >	x##_Sptrs;

VTK_SMART_POINTER(vtkAppendPolyData)
VTK_SMART_POINTER(vtkLine)
VTK_SMART_POINTER(vtkFloatArray)
VTK_SMART_POINTER(vtkCellArray)
VTK_SMART_POINTER(vtkCellData)
VTK_SMART_POINTER(vtkScalarsToColors)
VTK_SMART_POINTER(vtkLookupTable)
VTK_SMART_POINTER(vtkPolyDataMapper)
VTK_SMART_POINTER(vtkPolyData)
VTK_SMART_POINTER(vtkImageData)
VTK_SMART_POINTER(vtkAxesActor)
VTK_SMART_POINTER(vtkOrientationMarkerWidget)
VTK_SMART_POINTER(vtkRenderer)
VTK_SMART_POINTER(vtkRenderWindow)
VTK_SMART_POINTER(vtkRenderWindowInteractor)
VTK_SMART_POINTER(vtkPointData)
VTK_SMART_POINTER(vtkVertexGlyphFilter)
VTK_SMART_POINTER(vtkContourFilter)
VTK_SMART_POINTER(vtkSmartVolumeMapper)
VTK_SMART_POINTER(vtkVolume)
VTK_SMART_POINTER(vtkVolumeProperty)
VTK_SMART_POINTER(vtkOutlineFilter)
VTK_SMART_POINTER(vtkImagePlaneWidget)
VTK_SMART_POINTER(vtkCamera)
VTK_SMART_POINTER(vtkImageShiftScale)
VTK_SMART_POINTER(vtkUnsignedCharArray)
VTK_SMART_POINTER(vtkPiecewiseFunction)
VTK_SMART_POINTER(vtkColorTransferFunction)
VTK_SMART_POINTER(vtkProperty)
VTK_SMART_POINTER(vtkActor)
VTK_SMART_POINTER(vtkPoints)

template <class T>
void vtkSmartNew(vtkSmartPointer<T>& Ptr)
{
	Ptr = vtkSmartPointer<T>::New();
	assert(Ptr.GetPointer() != 0);
}

template <class T>
void vtkSmartNew_Initialize(vtkSmartPointer<T>& Ptr)
{
	Ptr = vtkSmartPointer<T>::New();
	assert(Ptr.GetPointer() != 0);
	Ptr->Initialize();
}

// percomplied wx
#include "wx/wx.h"
#include "wx/wxprec.h"
#include "wx/aui/framemanager.h"
#include "wx/frame.h"
#include "wx/gbsizer.h"
#include "wx/filepicker.h"
#include "wx/glcanvas.h"
