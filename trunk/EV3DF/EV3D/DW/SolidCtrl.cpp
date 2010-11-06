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
	tmpPtr->SetPolyData(m_polydata);
	if (m_imagedata.GetPointer() != 0)
	{
		tmpPtr->SetImageData(m_imagedata);
	}
	tmpPtr->m_histogram = Histogramd(m_sf3d->begin(), m_sf3d->size());
	tmpPtr->m_ParentCtrl = this;
	m_SolidDocPtrs.push_back(tmpPtr);
	return tmpPtr;
}

SolidView_Sptr	SolidCtrl::NewView( SEffect_Sptr& effect, SolidDoc_Sptr& doc )
{
	SolidView_Sptr tmpPtr(new SolidView(this, doc));
	tmpPtr->SetEffect(effect);
	m_SolidViewPtrs.push_back(tmpPtr);
	return tmpPtr;
}

int SolidCtrl::SetData( SJCScalarField3d* sf3d )
{
	vtkSmartNew(m_Axes_widget);
	vtkSmartNew(m_Axes);
	m_Axes_widget->SetOutlineColor( 0.8300, 0.6700, 0.5300 );
	m_Axes_widget->SetOrientationMarker( m_Axes );
	m_Axes_widget->SetInteractor( m_WindowInteractor );
	m_Axes_widget->On();
	m_sf3d = sf3d;
	// 先載入資料
	vtkSmartNew_Initialize(m_polydata);
	vtkSmartNew_Initialize(m_imagedata);
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
		m_imagedata->SetDimensions(x_len, y_len, z_len);
		m_imagedata->GetPointData()->SetScalars(point_array);
	}
	else
	{
		// Griding資料
	}
	m_polydata->SetPoints(points);
	m_polydata->GetPointData()->SetScalars(point_array);
	// 把資料跟bounding box建出來
	
	SolidDoc_Sptr spDoc = NewDoc();
	spDoc->SetPolyData(m_polydata);
	if (isGrided)
	{
		spDoc->SetImageData(m_imagedata);
	}
	spDoc->m_histogram = Histogramd(sf3d->begin(), sf3d->size());
	return 0;
}

SolidView_Sptr SolidCtrl::NewSEffect( SEffect_Sptr effect )
{
	SolidDoc_Sptr spDoc = NewDoc();
	spDoc->SetPolyData(m_polydata);
	if (m_imagedata.GetPointer() != 0)
	{
		 spDoc->SetImageData(m_imagedata);
	}
	SolidView_Sptr spView = NewView(effect, spDoc);
	return spView;
}
