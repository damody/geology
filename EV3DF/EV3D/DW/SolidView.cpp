﻿#include "StdVtk.h"
#include "SolidView.h"
#include "SolidDoc.h"
#include "SolidCtrl.h"
#include "SEffect.h"
#include "BoxArea.h"

SolidView::SolidView(SolidCtrl *ParentCtrl, SolidDoc_Sptr Doc):m_ParentCtrl(ParentCtrl)
{
	vtkSmartNew(m_actor);
	vtkSmartNew(m_polydataMapper);
	m_ParentDoc = Doc;
}

void SolidView::SetVisable( bool show )
{
	m_SEffect->m_Visable = show;
	if (show)
	{
		switch (m_SEffect->GetType())
		{
		case SEffect::CLIP_PLANE:
			m_ImagePlane->On();
			break;
		case SEffect::RULER:
			break;
		case SEffect::VOLUME_RENDERING:
			break;
		default:
			GetParentCtrl()->m_Renderer->AddActor(m_actor);
		}
	}
	else
	{
		switch (m_SEffect->GetType())
		{
		case SEffect::CLIP_PLANE:
			m_ImagePlane->Off();
			break;
		case SEffect::RULER:
			break;
		case SEffect::VOLUME_RENDERING:
			break;
		default:
			GetParentCtrl()->m_Renderer->RemoveActor(m_actor);
		}
	}
}

void SolidView::SetEffect( SEffect_Sptr effect )
{
	m_SEffect = effect;
	BoxArea_Sptr area = GetParentDoc()->m_area;
	switch (m_SEffect->GetType())
	{
	case SEffect::BOUNDING_BOX:
		{
			Init_BoundingBox();
		}
		break;
	case SEffect::VERTEX:
		{
			Init_Vertex();
		}
		break;
	case SEffect::CONTOUR:
		{
			Init_Contour();
		}
		break;
	case SEffect::AXES:
		{
			MESSAGE("not Implementation");
		}
		break;
	case SEffect::CLIP_PLANE:
		{
			Init_ClipPlane();
		}
		break;
	case SEffect::RULER:
		{
			MESSAGE("not Implementation");
		}
		break;
	case SEffect::CLIP_CONTOUR:
		{
			MESSAGE("not Implementation");
		}
		break;
	case SEffect::VOLUME_RENDERING:
		{
			Init_VolumeRendering();
		}
		break;
	}
	SetVisable(m_SEffect->m_Visable);
}

void SolidView::Update()
{
	switch (m_SEffect->GetType())
	{
	case SEffect::BOUNDING_BOX:
		{
			Init_BoundingBox();
		}
		break;
	case SEffect::VERTEX:
		{
			Init_Vertex();
		}
		break;
	case SEffect::CONTOUR:
		{
			Contour_Setting* setting = (Contour_Setting*)m_SEffect.get();
			m_ContourFilter->SetValue(1, setting->m_ContourValue);
			m_ContourFilter->Update();
		}
		break;
	case SEffect::AXES:
		{
			MESSAGE("not Implementation");
		}
		break;
	case SEffect::CLIP_PLANE:
		{
			double numvalue;
			ClipPlane_Setting* setting = (ClipPlane_Setting*)m_SEffect.get();
			switch (setting->m_Axes)
			{
			case 0:
				numvalue = GetParentDoc()->m_area->m_numX;
				m_ImagePlane->SetPlaneOrientationToXAxes();
				break;
			case 1:
				numvalue = GetParentDoc()->m_area->m_numY;
				m_ImagePlane->SetPlaneOrientationToYAxes();
				break;
			case 2:
				numvalue = GetParentDoc()->m_area->m_numZ;
				m_ImagePlane->SetPlaneOrientationToZAxes();
				break;
			}
			m_ImagePlane->SetSlicePosition(setting->m_Percent * numvalue / 100.0);
			setting->m_Percent = m_ImagePlane->GetSlicePosition()/numvalue*100.0;
		}
		break;
	case SEffect::RULER:
		{
			MESSAGE("not Implementation");
		}
		break;
	case SEffect::CLIP_CONTOUR:
		{
			MESSAGE("not Implementation");
		}
		break;
	case SEffect::VOLUME_RENDERING:
		{
			Init_VolumeRendering();
		}
		break;
	}
	SetVisable(m_SEffect->m_Visable);
	GetParentCtrl()->Render();
}

int SolidView::GetVisable()
{
	return m_SEffect->m_Visable;
}

int SolidView::GetType()
{
	return m_SEffect->m_Type;
}

void SolidView::Init_BoundingBox()
{
	vtkOutlineFilter_Sptr bounding_box;
	vtkSmartNew(bounding_box);
	bounding_box->SetInput(GetParentDoc()->m_ImageData);
	m_polydataMapper->SetInputConnection(bounding_box->GetOutputPort());
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Vertex()
{
	vtkUnsignedCharArray_Sptr colors;
	vtkSmartNew_Initialize(colors);
	colors->SetNumberOfComponents(3);
	colors->SetName("Colors");
	vtkLookupTable_Sptr lut;
	vtkSmartNew(lut);
	lut->SetTableRange(GetParentDoc()->m_histogram.GetPersentValue(0.01), 
		GetParentDoc()->m_histogram.GetPersentValue(0.99));
	lut->Build();
	int point_size = GetParentDoc()->m_PolyData->GetNumberOfPoints();
	vtkFloatArray* data_ary = (vtkFloatArray*)(GetParentDoc()->m_PolyData->GetPointData()->GetScalars("value"));
	for (int i = 0;i < point_size;i++)
	{
		double dcolor[3];
		lut->GetColor(data_ary->GetValue(i), dcolor);
		unsigned char color[3];
		for(unsigned int j = 0; j < 3; j++)
		{
			color[j] = static_cast<unsigned char>(255.0 * dcolor[j]);
		}
		colors->InsertNextTupleValue(color);
	}
	vtkVertexGlyphFilter_Sptr vertexGlyphFilter;
	vtkSmartNew(vertexGlyphFilter);
	vtkPolyData_Sptr colorpolydata;
	vtkSmartNew(colorpolydata);
	colorpolydata->SetPoints(GetParentDoc()->m_PolyData->GetPoints());
	colorpolydata->GetPointData()->SetScalars(colors);
	vertexGlyphFilter->SetInput(colorpolydata);
	vertexGlyphFilter->Update();
	m_polydataMapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());			
	m_polydataMapper->SetLookupTable(lut);
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Contour()
{
	vtkLookupTable_Sptr lut;
	vtkSmartNew(lut);
	lut->SetTableRange(GetParentDoc()->m_histogram.GetPersentValue(0.01), 
		GetParentDoc()->m_histogram.GetPersentValue(0.99));
	lut->Build();
	vtkSmartNew(m_ContourFilter);
	m_ContourFilter->SetInput(GetParentDoc()->m_ImageData);
	m_polydataMapper->SetInputConnection(m_ContourFilter->GetOutputPort());
	m_polydataMapper->SetLookupTable(lut);
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Axes()
{

}

void SolidView::Init_ClipPlane()
{
	vtkSmartNew(m_ImagePlane);
	m_ImagePlane->SetLeftButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	//m_ImagePlane->SetMiddleButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_ImagePlane->SetRightButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_ImagePlane->SetInteractor(GetParentCtrl()->m_WindowInteractor);
	m_ImagePlane->RestrictPlaneToVolumeOn();
	m_ImagePlane->SetInput(GetParentDoc()->m_ImageData);
	m_ImagePlane->SetPlaneOrientationToXAxes();
	vtkLookupTable_Sptr lut;
	vtkSmartNew(lut);
	lut->SetTableRange(GetParentDoc()->m_histogram.GetPersentValue(0.01), 
		GetParentDoc()->m_histogram.GetPersentValue(0.99));
	lut->Build();
	m_ImagePlane->SetLookupTable(lut);
	m_ImagePlane->On();
}

void SolidView::Init_Ruler()
{

}

void SolidView::Init_ClipContour()
{

}

void SolidView::Init_VolumeRendering()
{
	vtkPiecewiseFunction_Sptr PiecewiseFunction;
	vtkSmartNew(PiecewiseFunction);
	vtkColorTransferFunction_Sptr ColorTransferFunction;
	vtkSmartNew(ColorTransferFunction);
	vtkImageShiftScale_Sptr ImageShiftScale;
	vtkSmartNew(ImageShiftScale);
	vtkSmartVolumeMapper_Sptr SmartVolumeMapper;
	vtkSmartNew(SmartVolumeMapper);
	vtkVolumeProperty_Sptr VolumeProperty;
	vtkSmartNew(VolumeProperty);
	vtkVolume_Sptr Volume;
	vtkSmartNew(Volume);
}
