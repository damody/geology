#pragma once
class Solid;
class SolidDoc;
class SolidView;
class SolidCtrl;
class SEffect;
class ColorTable;

#include "DWHistogram.h"
#include "Interpolation/vtkBounds.h"
#include <boost/shared_ptr.hpp>
#include <vector>
#include <cassert>

/*! 使用這個巨集將會得到
SHARE_PTR(x, y)
boost::shared_ptr<x> y_Sptr;
std::vector<t_Sptr> y_Sptrs;
*/
#define SHARE_PTR(x) \
	typedef boost::shared_ptr< x >	x##_Sptr; \
	typedef std::vector< x##_Sptr >	x##_Sptrs;
template <class T>
void shareNew(boost::shared_ptr<T>& Ptr)
{
	Ptr = boost::shared_ptr<T>(new T);
	assert(Ptr.get() != 0);
}
static struct
{
	template <class T>
	operator boost::shared_ptr<T>()
	{
		return boost::shared_ptr<T>::New();
	}
}SharePtrNew;

/*!
typedef boost::shared_ptr<SolidDoc>		SolidDoc_Sptr;
typedef std::vector<SolidDoc_Sptr>		SolidDoc_Sptrs;
*/
SHARE_PTR(Solid)
SHARE_PTR(SolidDoc)
SHARE_PTR(SolidView)
SHARE_PTR(SolidCtrl)
SHARE_PTR(SEffect)
SHARE_PTR(ColorTable)

// enum {
// 	SET_OK,
// 	SET_FAIL
// };

#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>
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
VTK_SMART_POINTER(vtkDoubleArray)
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
template <typename T, typename U>
class has_member_Initialize_tester
{
private:
	template <U> struct helper;
	template <typename T> static char check(helper<&T::Initialize> *);
	template <typename T> static char (&check(...))[2];
public:	
	enum { value = (sizeof(check<T>(0)) == sizeof(char)) };
};

template <char Doit, class T>
struct static_Check_To_Initialize
{
	static void Do(T& ic){ic;}
};
template<class T>
struct static_Check_To_Initialize<1, T>
{
	static void Do(T& ic){ic->Initialize();}
};

static struct
{
	template <class T>
	operator vtkSmartPointer<T>()
	{
		vtkSmartPointer<T> ptr = vtkSmartPointer<T>::New();
		static_Check_To_Initialize
			<
				has_member_Initialize_tester< T, void (T::*)()>::value, 
				vtkSmartPointer<T>
			>::Do(ptr);
		return ptr;
	}
}vtkSmartNew;

static struct
{
	template <class T>
	operator vtkSmartPointer<T>()
	{
		return vtkSmartPointer<T>::New();
	}
}vtkOnlyNew;

/* old vtkSmartNew
template <class T>
void vtkSmartNew_Initialize(vtkSmartPointer<T>& Ptr)
{
	Ptr = vtkSmartPointer<T>::New();
	assert(Ptr.GetPointer() != 0);
	Ptr->Initialize();
}
*/

#include <windows.h>
#include <tchar.h>
#define MESSAGE(x) MessageBox(NULL, _T(x), _T("MESSAGE"), 0);

