#include "StdVtk.h"
#include "SolidCtrl.h"
#include "SolidDoc.h"
#include "SolidView.h"
#include "SEffect.h"

SolidDoc_Sptr	SolidCtrl::NewDoc()
{
	BoxArea_Sptr area(new BoxArea);
	area->m_rangeX = m_sf3d->m_dLengthX;
	area->m_rangeY = m_sf3d->m_dLengthY;
	area->m_rangeZ = m_sf3d->m_dLengthZ;
	area->m_numX = m_sf3d->NumX();
	area->m_numY = m_sf3d->NumY();
	area->m_numZ = m_sf3d->NumZ();
	SolidDoc_Sptr tmpPtr(new SolidDoc(area));
	m_SolidDocPtrs.push_back(tmpPtr);
	tmpPtr->m_WindowInteractor->SetRenderWindow(m_RenderWindow);
	return tmpPtr;
}

SolidView_Sptr	SolidCtrl::NewView( SEffect_Sptr& effect, SolidDoc_Sptr& doc )
{
	SolidView_Sptr tmpPtr(new SolidView(doc));
	tmpPtr->SetRenderTarget(m_Renderer);
	tmpPtr->SetEffect(effect);
	tmpPtr->m_ParentCtrl = this;
	m_SolidViewPtrs.push_back(tmpPtr);
	return tmpPtr;
}

int SolidCtrl::SetData( SJCScalarField3d* sf3d )
{
	m_sf3d = sf3d;
	// 先載入資料
	vtkPolyData_Sptr polydata;
	vtkSmartNew_Initialize(polydata);
	vtkImageData_Sptr imagedata;
	vtkSmartNew_Initialize(imagedata);
	vtkPoints_Sptr points;
	vtkSmartNew_Initialize(points);
 	vtkFloatArray_Sptr point_array;
 	vtkSmartNew_Initialize(point_array);
	point_array->SetName("value");
	const uint x_len = sf3d->NumX(),
		y_len = sf3d->NumY(),
		z_len = sf3d->NumZ();
	uint i, j, k,  kOffset, jOffset, offset;
	for(k=0;k<z_len;k++)
	{
		kOffset = k*y_len*x_len;
		for(j=0; j<y_len; j++)
		{
			jOffset = j*x_len;
			for(i=0;i<x_len;i++)
			{
				offset = i + jOffset + kOffset;
				point_array->InsertTuple1(offset, sf3d->Value(i, j, k));
				points->InsertNextPoint(i, j, k);
			}
		}
	}
	uint count = point_array->GetNumberOfTuples();
	// 如果資料被Griding過了就直接放到imagedata
	bool isGrided = x_len* y_len* z_len == count;
	if (isGrided) 
	{
		imagedata->SetDimensions(x_len, y_len, z_len);
		imagedata->GetPointData()->SetScalars(point_array);
	}
	else
	{
		// Griding資料
	}
	polydata->SetPoints(points);
	polydata->GetPointData()->SetScalars(point_array);
	// 把資料跟bounding box建出來
	
	SolidDoc_Sptr spDoc = NewDoc();
	int res = spDoc->SetPolyData(polydata);
	assert(res == SET_OK);
	if (isGrided)
	{
		res = spDoc->SetImageData(imagedata);
		assert(res == SET_OK);
	}
	spDoc->m_histogram = Histogramd(sf3d->begin(), sf3d->size());

	SEffect_Sptr Setting = SEffect::New(SEffect::VERTEX);
	SolidView_Sptr spView = NewView(Setting, spDoc);
	SEffect_Sptr Setting2 = SEffect::New(SEffect::BOUNDING_BOX);
	SolidView_Sptr spView2 = NewView(Setting2, spDoc);
	SEffect_Sptr Setting3 = SEffect::New(SEffect::PLANE_CHIP);
	//SolidView_Sptr spView3 = NewView(Setting3, spDoc);
	return 0;
}
