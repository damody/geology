
#pragma once
class Solid;
class	SolidDoc;
class	SolidView;
class	SolidCtrl;
class	SEffect;
class	ColorTable;
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
	typedef boost::shared_ptr<x>	   x##_Sptr; \
	typedef std::vector<x##_Sptr>			x##_Sptrs;
template<class T> void shareNew(boost::shared_ptr<T> &Ptr)
{
	Ptr = boost::shared_ptr<T> (new T);
	assert(Ptr.get() != 0);
}

static struct
{
	template<class T> operator boost::shared_ptr<T> ()
	{
		return boost::shared_ptr<T> (new T);
	}
}
SharePtrNew;

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
#include <vtkCubeAxesActor.h>
#include <vtkImageMapToColors.h>
#include <vtkPolyDataNormals.h>
#include <vtkInteractorStyleTrackballCamera.h>
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
#include <vtkDelaunay2D.h>
#include <vtkScalarBarActor.h>
#define VTKSMART_PTR(x) \
	typedef vtkSmartPointer<x> x##_Sptr; \
	typedef std::vector<x##_Sptr> x##_Sptrs;
VTKSMART_PTR(vtkScalarBarActor)
VTKSMART_PTR(vtkDelaunay2D)
VTKSMART_PTR(vtkCubeAxesActor)
VTKSMART_PTR(vtkPolyDataNormals)
VTKSMART_PTR(vtkImageMapToColors)
VTKSMART_PTR(vtkInteractorStyleTrackballCamera)
VTKSMART_PTR(vtkDoubleArray)
VTKSMART_PTR(vtkCellData)
VTKSMART_PTR(vtkScalarsToColors)
VTKSMART_PTR(vtkLookupTable)
VTKSMART_PTR(vtkPolyDataMapper)
VTKSMART_PTR(vtkPolyData)
VTKSMART_PTR(vtkImageData)
VTKSMART_PTR(vtkAxesActor)
VTKSMART_PTR(vtkOrientationMarkerWidget)
VTKSMART_PTR(vtkRenderer)
VTKSMART_PTR(vtkRenderWindow)
VTKSMART_PTR(vtkRenderWindowInteractor)
VTKSMART_PTR(vtkPointData)
VTKSMART_PTR(vtkVertexGlyphFilter)
VTKSMART_PTR(vtkContourFilter)
VTKSMART_PTR(vtkSmartVolumeMapper)
VTKSMART_PTR(vtkVolume)
VTKSMART_PTR(vtkVolumeProperty)
VTKSMART_PTR(vtkOutlineFilter)
VTKSMART_PTR(vtkImagePlaneWidget)
VTKSMART_PTR(vtkCamera)
VTKSMART_PTR(vtkImageShiftScale)
VTKSMART_PTR(vtkUnsignedCharArray)
VTKSMART_PTR(vtkPiecewiseFunction)
VTKSMART_PTR(vtkColorTransferFunction)
VTKSMART_PTR(vtkProperty)
VTKSMART_PTR(vtkActor)
VTKSMART_PTR(vtkPoints)
template<typename T, typename U> class has_member_Initialize_tester
{
private:
	template<U> struct helper;
	template<typename T> static char check(helper < &T::Initialize > *);
	template<typename T> static char (&check(...))[2];
public:
	enum { value = (sizeof (check<T> (0)) == sizeof (char)) };
};
template<char Doit, class T> struct static_Check_To_Initialize
{
	static void	Do(T &ic)	{ ic; }
};
template<class T> struct static_Check_To_Initialize<1, T>
{
	static void	Do(T &ic)	{ ic->Initialize(); }
};
static struct
{
	template<class T> operator vtkSmartPointer<T> ()
	{
		vtkSmartPointer<T> ptr = vtkSmartPointer<T>::New();
		static_Check_To_Initialize<has_member_Initialize_tester<T, void(T:: *) ()>::value, vtkSmartPointer<T> >::Do(ptr);
		return ptr;
	}
}
vtkSmartNew;
static struct
{
	template<class T> operator vtkSmartPointer<T> ()
	{
		return vtkSmartPointer<T>::New();
	}
}
vtkOnlyNew;

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
#define MESSAGE(x)	MessageBox(NULL, _T(x), _T("MESSAGE"), 0);
