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
		case SEffect::PLANE_CHIP:
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
		case SEffect::PLANE_CHIP:
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
	case SEffect::PLANE_CHIP:
		{
			Init_PlaneChip();
		}
		break;
	case SEffect::RULER:
		{
			MESSAGE("not Implementation");
		}
		break;
	case SEffect::CONTOUR_CHIP:
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
	vtkContourFilter_Sptr ContourFilter;
	vtkSmartNew(ContourFilter);
	ContourFilter->SetInput(GetParentDoc()->m_ImageData);
	m_polydataMapper->SetInputConnection(ContourFilter->GetOutputPort());
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Axes()
{

}

void SolidView::Init_PlaneChip()
{
	vtkSmartNew(m_ImagePlane);
	m_ImagePlane->SetLeftButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_ImagePlane->SetMiddleButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
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

void SolidView::Init_ContourChip()
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
