﻿
#include "StdWxVtk.h"
#include "SolidView.h"
#include "SolidDoc.h"
#include "SolidCtrl.h"
#include "SEffect.h"
#include "CoordinateTransform.h"

SolidView::SolidView(SolidCtrl * ParentCtrl, SolidDoc_Sptr Doc) :
	m_ParentCtrl(ParentCtrl)
{
	m_actor = vtkSmartNew;
	m_polydataMapper = vtkSmartNew;
	m_ParentDoc = Doc;
}
void SolidView::SetVisable(bool show)
{
	m_SEffect->m_Visable = show;
	if (show)
	{
		switch (m_SEffect->GetType())
		{
		case SEffect::CLIP_PLANE:	m_ImagePlane->On(); break;
		case SEffect::RULER:		break;
		case SEffect::VOLUME_RENDERING: GetParentCtrl()->m_Renderer->AddViewProp(m_volume); 
			GetParentCtrl()->m_Renderer->AddActor2D(m_ScalarBarActor); break;
		case SEffect::AXES_TWD97_TO_WGS84:
		case SEffect::AXES:		GetParentCtrl()->m_Renderer->AddActor(m_CubeAxesActor); break;
		default:			GetParentCtrl()->m_Renderer->AddActor(m_actor);
		}
	}
	else
	{
		switch (m_SEffect->GetType())
		{
		case SEffect::CLIP_PLANE:	m_ImagePlane->Off(); break;
		case SEffect::RULER:		break;
		case SEffect::VOLUME_RENDERING: GetParentCtrl()->m_Renderer->RemoveViewProp(m_volume); 
			GetParentCtrl()->m_Renderer->RemoveActor2D(m_ScalarBarActor); break;
		case SEffect::AXES_TWD97_TO_WGS84:
		case SEffect::AXES:		GetParentCtrl()->m_Renderer->RemoveActor(m_CubeAxesActor); break;
		default:			GetParentCtrl()->m_Renderer->RemoveActor(m_actor);
		}
	}
}

void SolidView::SetEffect(SEffect_Sptr effect)
{
	m_SEffect = effect;
	switch (m_SEffect->GetType())
	{
	case SEffect::BOUNDING_BOX:	{ Init_BoundingBox(); }break;
	case SEffect::VERTEX:		{ Init_Vertex(); }break;
	case SEffect::CONTOUR:		{ Init_Contour(); }break;
	case SEffect::AXES:		{ Init_Axes(); }break;
	case SEffect::AXES_TWD97_TO_WGS84:	{ Init_Axes_TWD97_TO_WGS84(); } break;
	case SEffect::CLIP_PLANE:	{ Init_ClipPlane(); }break;
	case SEffect::RULER:		{ Init_Ruler(); }break;
	case SEffect::CLIP_CONTOUR:	{ Init_ClipContour(); }break;
	case SEffect::VOLUME_RENDERING: { Init_VolumeRendering(); }break;
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
			Contour_Setting *setting = (Contour_Setting *) m_SEffect.get();
			m_ContourFilter->SetValue(1, setting->m_ContourValue);
			m_ContourFilter->Update();
		}
		break;

	case SEffect::AXES:
		{ }
		break;

	case SEffect::CLIP_PLANE:
		{
			double			numvalue;
			ClipPlane_Setting	*setting = (ClipPlane_Setting *) m_SEffect.get();
			switch (setting->m_Axes)
			{
			case 0:
				numvalue = GetParentDoc()->m_bounds.Xlen();
				m_ImagePlane->SetPlaneOrientationToXAxes();
				break;

			case 1:
				numvalue = GetParentDoc()->m_bounds.Ylen();
				m_ImagePlane->SetPlaneOrientationToYAxes();
				break;

			case 2:
				numvalue = GetParentDoc()->m_bounds.Zlen();
				m_ImagePlane->SetPlaneOrientationToZAxes();
				break;

			default:
				assert(0 && "setting->m_Axes");
			}
			m_ImagePlane->SetSlicePosition(setting->m_Percent * numvalue / 100.0);
			setting->m_Percent = m_ImagePlane->GetSlicePosition() / numvalue * 100.0;
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
	vtkOutlineFilter_Sptr	bounding_box = vtkSmartNew;
	bounding_box->SetInput(GetParentDoc()->m_ImageData);
	m_polydataMapper->SetInputConnection(bounding_box->GetOutputPort());
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Vertex()
{
	vtkUnsignedCharArray_Sptr	colors = vtkSmartNew;
	colors->SetNumberOfComponents(3);
	colors->SetName("Colors");

	vtkLookupTable_Sptr	lut = vtkSmartNew;
	lut->SetTableRange
	(
		GetParentDoc()->m_histogram.GetPersentValue(0.01),
		GetParentDoc()->m_histogram.GetPersentValue(0.99)
	);
	lut->Build();

	double	p[6];
	GetParentDoc()->m_PolyData->GetBounds(p);

	int		point_size = GetParentDoc()->m_PolyData->GetNumberOfPoints();
	vtkDoubleArray	*data_ary = (vtkDoubleArray *)
		(GetParentDoc()->m_PolyData->GetPointData()->GetScalars("value"));
	for (int i = 0; i < point_size; i++)
	{
		double	dcolor[3];
		lut->GetColor(data_ary->GetValue(i), dcolor);

		unsigned char	color[3];
		for (unsigned int j = 0; j < 3; j++)
		{
			color[j] = static_cast<unsigned char>(255.0 * dcolor[j]);
		}

		colors->InsertNextTupleValue(color);
	}

	vtkVertexGlyphFilter_Sptr	vertexGlyphFilter = vtkSmartNew;
	vtkPolyData_Sptr		colorpolydata = vtkSmartNew;
	colorpolydata->SetPoints(GetParentDoc()->m_PolyData->GetPoints());
	colorpolydata->GetPointData()->SetScalars(colors);
	vertexGlyphFilter->SetInput(colorpolydata);
	m_polydataMapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());
	m_polydataMapper->SetLookupTable(lut);
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Contour()
{
	vtkLookupTable_Sptr	lut = vtkSmartNew;
	lut->SetTableRange
	(
		GetParentDoc()->m_histogram.GetPersentValue(0.01),
		GetParentDoc()->m_histogram.GetPersentValue(0.99)
	);
	lut->Build();
	m_ContourFilter = vtkSmartNew;
	m_ContourFilter->SetInput(GetParentDoc()->m_ImageData);
	m_polydataMapper->SetInputConnection(m_ContourFilter->GetOutputPort());

	vtkColorTransferFunction_Sptr	colorTransferFunction = vtkSmartNew;
	colorTransferFunction->AddRGBPoint(0.0, 1.0 / 2, 0.0, 0.0);
	colorTransferFunction->AddRGBPoint(20.0, 1.0 / 2, 165 / 255 / 2.0, 0.0);
	colorTransferFunction->AddRGBPoint(40.0, 1.0 / 2, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(50.0, 0.0, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(60.0, 0.0, 0.5 / 2, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(70.0, 0.0, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(80.0, 139 / 255.0 / 2, 0.0, 1.0 / 2);
	m_polydataMapper->SetLookupTable(colorTransferFunction);

	vtkPolyDataNormals_Sptr sGridPolyDataNormal = vtkSmartNew;
	sGridPolyDataNormal->SetInput(m_ContourFilter->GetOutput());
	sGridPolyDataNormal->Update();
	m_polydataMapper->SetInput(sGridPolyDataNormal->GetOutput());
	m_polydataMapper->Update();
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Axes_TWD97_TO_WGS84()
{
	m_CubeAxesActor = vtkSmartNew;
	double boundingTWD[6];
	double boundingWGS[6];
	GetParentDoc()->m_ImageData->GetBounds(boundingTWD);
	GetParentDoc()->m_ImageData->GetBounds(boundingWGS);
	//xmin zmin
	CoordinateTransform::TWD97_To_lonlat(350920-boundingTWD[0], boundingTWD[4], boundingWGS+0, boundingWGS+4);
	CoordinateTransform::TWD97_To_lonlat(350920-boundingTWD[1], boundingTWD[5], boundingWGS+1, boundingWGS+5);
	
	m_CubeAxesActor->SetBounds(GetParentDoc()->m_ImageData->GetBounds());
	m_CubeAxesActor->SetCamera(GetParentCtrl()->m_Renderer->GetActiveCamera());
	m_CubeAxesActor->SetBounds(boundingTWD);
	m_CubeAxesActor->SetXAxisRange(-boundingWGS[0], -boundingWGS[1]);
	m_CubeAxesActor->SetYAxisRange(boundingWGS[2]/10, boundingWGS[3]/10);
	m_CubeAxesActor->SetZAxisRange(boundingWGS[4], boundingWGS[5]);
	m_CubeAxesActor->SetDrawXGridlines(0);
	m_CubeAxesActor->SetXTitle("E,lon");
	m_CubeAxesActor->SetYTitle("Height");
	m_CubeAxesActor->SetZTitle("N,lot");
	m_CubeAxesActor->SetXLabelFormat("%-#f");
	m_CubeAxesActor->SetYLabelFormat("%-#f");
	m_CubeAxesActor->SetZLabelFormat("%-#f");
	m_CubeAxesActor->SetLabelScaling(false,0,0,0);
	m_CubeAxesActor->SetFlyModeToStaticEdges();
}

void SolidView::Init_Axes()
{
	m_CubeAxesActor = vtkSmartNew;
	double bounding[6];
	GetParentDoc()->m_ImageData->GetBounds(bounding);
	m_CubeAxesActor->SetBounds(GetParentDoc()->m_ImageData->GetBounds());
	m_CubeAxesActor->SetCamera(GetParentCtrl()->m_Renderer->GetActiveCamera());
	m_CubeAxesActor->SetBounds(bounding);
	m_CubeAxesActor->SetXAxisRange(-350920,-150120);
	m_CubeAxesActor->SetXTitle("E,lon");
	m_CubeAxesActor->SetYTitle("Height");
	m_CubeAxesActor->SetZTitle("N,lot");
	m_CubeAxesActor->SetXLabelFormat("%-#f");
	m_CubeAxesActor->SetYLabelFormat("%-#f");
	m_CubeAxesActor->SetZLabelFormat("%-#f");
	m_CubeAxesActor->SetLabelScaling(false,4,4,4);
	m_CubeAxesActor->SetFlyModeToStaticEdges();
}

void SolidView::Init_ClipPlane()
{
	m_ImagePlane = vtkSmartNew;
	m_ImagePlane->SetLeftButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	//m_ImagePlane->SetMiddleButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_ImagePlane->SetRightButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_ImagePlane->SetInteractor(GetParentCtrl()->m_WindowInteractor);
	m_ImagePlane->RestrictPlaneToVolumeOn();
	m_ImagePlane->SetInput(GetParentDoc()->m_ImageData);
	m_ImagePlane->SetPlaneOrientationToXAxes();

	vtkColorTransferFunction_Sptr	colorTransferFunction = vtkSmartNew;
	colorTransferFunction->AddRGBPoint(649.0, 1.0 / 2, 0.0, 0.0);
	colorTransferFunction->AddRGBPoint(350.0, 1.0 / 2, 165 / 255 / 2.0, 0.0);
	colorTransferFunction->AddRGBPoint(300.0, 1.0 / 2, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(250.0, 0.0, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(200.0, 0.0, 0.5 / 2, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(150.0, 0.0, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(100.0, 139 / 255.0 / 2, 0.0, 1.0 / 2);
	m_ImagePlane->GetColorMap()->SetLookupTable(colorTransferFunction);
	m_ImagePlane->On();
}

void SolidView::Init_Ruler()
{
	MESSAGE("not Implementation");
}

void SolidView::Init_ClipContour()
{
	MESSAGE("not Implementation");
}

void SolidView::Init_VolumeRendering()
{
	vtkPiecewiseFunction_Sptr	compositeOpacity = vtkSmartNew;
	vtkColorTransferFunction_Sptr	colorTransferFunction = vtkSmartNew;
	vtkSmartVolumeMapper_Sptr	volumeMapper = vtkOnlyNew;
	vtkVolumeProperty_Sptr		volumeProperty = vtkSmartNew;
	m_ScalarBarActor = vtkSmartNew;
	volumeMapper->SetBlendModeToComposite();		// composite first
	volumeMapper->SetRequestedRenderMode(vtkSmartVolumeMapper::GPURenderMode);
	volumeMapper->SetInputConnection(GetParentDoc()->m_ImageData->GetProducerPort());
	volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);
	compositeOpacity->AddPoint(649.0, 0.0001);
	compositeOpacity->AddPoint(450.0, 0.0005);
	compositeOpacity->AddPoint(400.0, 0.0001);
	compositeOpacity->AddPoint(300.0, 0.0005);
	compositeOpacity->AddPoint(250.0, 0.0005);
	compositeOpacity->AddPoint(170.0, 0.0001);
	compositeOpacity->AddPoint(143.0, 0.0001);
	volumeProperty->SetScalarOpacity(compositeOpacity);	// composite first.
	volumeProperty->SetDiffuse(0.2);
	volumeProperty->ShadeOff();
	volumeProperty->SetDisableGradientOpacity(1);
	colorTransferFunction->AddRGBPoint(649.0, 1.0 / 2, 0.0, 0.0);
	colorTransferFunction->AddRGBPoint(450.0, 1.0 / 2, 165 / 255 / 2.0, 0.0);
	colorTransferFunction->AddRGBPoint(350.0, 1.0 / 2, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(280.0, 0.0, 1.0 / 2, 0.0);
	colorTransferFunction->AddRGBPoint(230.0, 0.0, 0.5 / 2, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(180.0, 0.0, 0.0, 1.0 / 2);
	colorTransferFunction->AddRGBPoint(100.0, 139 / 255.0 / 2, 0.0, 1.0 / 2);
	m_ScalarBarActor->SetLookupTable(colorTransferFunction);
	m_ScalarBarActor->SetNumberOfLabels(4);
 	m_ScalarBarActor->SetMaximumWidthInPixels(60);
	m_ScalarBarActor->SetMaximumHeightInPixels(300);
	GetParentCtrl()->m_Renderer->AddActor2D(m_ScalarBarActor);
	volumeProperty->SetColor(colorTransferFunction);
	m_volume = vtkSmartNew;
	m_volume->SetMapper(volumeMapper);
	m_volume->SetProperty(volumeProperty);
	GetParentCtrl()->m_Renderer->AddViewProp(m_volume);
}
