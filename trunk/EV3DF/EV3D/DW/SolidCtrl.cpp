#include "StdVtk.h"
#include "SolidCtrl.h"
#include "SolidDoc.h"
#include "SolidView.h"
#include "SEffect.h"
#include "Interpolation/vtkInverseDistanceFilter.h"
#include "Interpolation/vtkNearestNeighborFilter.h"

SolidDoc_Sptr	SolidCtrl::NewDoc() // 新資料集
{
	BoxArea_Sptr area;
	shareNew(area);
	*area = *m_area;
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

SolidView_Sptr	SolidCtrl::NewView( SEffect_Sptr& effect, SolidDoc_Sptr& doc ) // 新的資料虛擬化View
{
	SolidView_Sptr tmpPtr(new SolidView(this, doc));
	tmpPtr->SetEffect(effect);
	m_SolidViewPtrs.push_back(tmpPtr);
	return tmpPtr;
}

int SolidCtrl::SetData( SJCScalarField3d* sf3d, InterpolationMethod method /*= NEAREST*/ )
{
	RmAllView();
	m_sf3d = sf3d;
	shareNew(m_area);
	m_area->m_rangeX = m_sf3d->m_dLengthX;
	m_area->m_rangeY = m_sf3d->m_dLengthY;
	m_area->m_rangeZ = m_sf3d->m_dLengthZ;
	m_area->m_numX = m_sf3d->NumX();
	m_area->m_numY = m_sf3d->NumY();
	m_area->m_numZ = m_sf3d->NumZ();
	// 先載入資料
	m_polydata = vtkSmartNew;
	m_imagedata = vtkSmartNew;
	vtkPoints_Sptr points = vtkSmartNew;
 	vtkFloatArray_Sptr point_array = vtkSmartNew;
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
		switch (method)
		{
		case NEAREST:
			break;
		case INVERSE:
			break;
			
		}
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
	m_Camera->SetPosition(0, 0, (m_area->m_numX+m_area->m_numY+m_area->m_numZ)/3.0);
	m_Camera->SetFocalPoint(m_area->m_numX/2.0, m_area->m_numY/2.0, m_area->m_numZ/2.0);
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

void SolidCtrl::ReSetViewDirection()
{
	m_Camera->SetPosition(0, 0, (m_area->m_numX+m_area->m_numY+m_area->m_numZ)/3.0);
	m_Camera->SetFocalPoint(m_area->m_numX/2.0, m_area->m_numY/2.0, m_area->m_numZ/2.0);
}

void SolidCtrl::RmView( SolidView_Sptr view )
{
	for (SolidView_Sptrs::iterator it = m_SolidViewPtrs.begin();
		it != m_SolidViewPtrs.end();
		it++)
	{
		if (*it == view)
		{
			(*it)->SetVisable(false);
			m_SolidViewPtrs.erase(it);
			SolidView_Sptr tmp;
			tmp.swap(view);
			break;
		}
	}
}

void SolidCtrl::RmDoc( SolidDoc_Sptr doc )
{
	for (SolidDoc_Sptrs::iterator it = m_SolidDocPtrs.begin();
		it != m_SolidDocPtrs.end();
		it++)
	{
		if (*it == doc)
		{
			m_SolidDocPtrs.erase(it);
			doc->RmAllView();
			SolidDoc_Sptr tmp;
			tmp.swap(doc);
			break;
		}
	}
}

void SolidCtrl::RmAllView()
{
	for (SolidView_Sptrs::iterator it = m_SolidViewPtrs.begin();
		it != m_SolidViewPtrs.end();
		it++)
	{
		(*it)->SetVisable(false);
		SolidView_Sptr tmp;
		tmp.swap(*it);
	}
	m_SolidViewPtrs.clear();
}
