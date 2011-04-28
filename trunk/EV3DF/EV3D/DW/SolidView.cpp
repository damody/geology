#include "StdVtk.h"
#include "SolidView.h"
#include "SolidDoc.h"
#include "SolidCtrl.h"
#include "SEffect.h"
#include <vtkCubeAxesActor.h>

SolidView::SolidView(SolidCtrl *ParentCtrl, SolidDoc_Sptr Doc):m_ParentCtrl(ParentCtrl)
{
	m_actor = vtkSmartNew;
	m_polydataMapper = vtkSmartNew;
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
		case SEffect::AXES:
			GetParentCtrl()->m_Renderer->AddActor(m_CubeAxesActor);
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
			GetParentCtrl()->m_Renderer->RemoveViewProp(m_volume);
			break;
		case SEffect::AXES:
			GetParentCtrl()->m_Renderer->RemoveActor(m_CubeAxesActor);
			break;
		default:
			GetParentCtrl()->m_Renderer->RemoveActor(m_actor);
		}
	}
}

void SolidView::SetEffect( SEffect_Sptr effect )
{
	m_SEffect = effect;
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
			Init_Axes();
		}
		break;
	case SEffect::CLIP_PLANE:
		{
			Init_ClipPlane();
		}
		break;
	case SEffect::RULER:
		{
			Init_Ruler();
		}
		break;
	case SEffect::CLIP_CONTOUR:
		{
			Init_ClipContour();
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
			
		}
		break;
	case SEffect::CLIP_PLANE:
		{
			double numvalue;
			ClipPlane_Setting* setting = (ClipPlane_Setting*)m_SEffect.get();
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
			setting->m_Percent = m_ImagePlane->GetSlicePosition()/numvalue*100.0;
			vtkProperty_Sptr proerty = vtkSmartNew;
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
			//Init_VolumeRendering();
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
	vtkOutlineFilter_Sptr bounding_box = vtkSmartNew;
	bounding_box->SetInput(GetParentDoc()->m_ImageData);
	m_polydataMapper->SetInputConnection(bounding_box->GetOutputPort());
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Vertex()
{
	vtkUnsignedCharArray_Sptr colors = vtkSmartNew;
	colors->SetNumberOfComponents(3);
	colors->SetName("Colors");
	vtkLookupTable_Sptr lut = vtkSmartNew;
	lut->SetTableRange(GetParentDoc()->m_histogram.GetPersentValue(0.01), 
		GetParentDoc()->m_histogram.GetPersentValue(0.99));
	lut->Build();
	double p[6];
	GetParentDoc()->m_PolyData->GetBounds(p);
	int point_size = GetParentDoc()->m_PolyData->GetNumberOfPoints();
	vtkDoubleArray* data_ary = (vtkDoubleArray*)(GetParentDoc()->m_PolyData->GetPointData()->GetScalars("value"));
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
	vtkVertexGlyphFilter_Sptr vertexGlyphFilter = vtkSmartNew;
	vtkPolyData_Sptr colorpolydata = vtkSmartNew;
	colorpolydata->SetPoints(GetParentDoc()->m_PolyData->GetPoints());
	colorpolydata->GetPointData()->SetScalars(colors);
	vertexGlyphFilter->SetInput(colorpolydata);
	m_polydataMapper->SetInputConnection(vertexGlyphFilter->GetOutputPort());			
	m_polydataMapper->SetLookupTable(lut);
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Contour()
{
	vtkLookupTable_Sptr lut = vtkSmartNew;
	lut->SetTableRange(GetParentDoc()->m_histogram.GetPersentValue(0.01), 
		GetParentDoc()->m_histogram.GetPersentValue(0.99));
	lut->Build();
	m_ContourFilter = vtkSmartNew;
	m_ContourFilter->SetInput(GetParentDoc()->m_ImageData);
	m_polydataMapper->SetInputConnection(m_ContourFilter->GetOutputPort());
	vtkColorTransferFunction_Sptr colorTransferFunction = vtkSmartNew;
	colorTransferFunction->AddRGBPoint(0.0  ,1.0/2,0.0,0.0);
	colorTransferFunction->AddRGBPoint(20.0  ,1.0/2, 165/255/2.0,0.0);
	colorTransferFunction->AddRGBPoint(40.0  ,1.0/2, 1.0/2,0.0);
	colorTransferFunction->AddRGBPoint(50.0  ,0.0, 1.0/2,0.0);
	colorTransferFunction->AddRGBPoint(60.0  ,0.0, 0.5/2, 1.0/2);
	colorTransferFunction->AddRGBPoint(70.0  ,0.0, 0.0, 1.0/2);
	colorTransferFunction->AddRGBPoint(80.0  ,139/255.0/2, 0.0, 1.0/2);
	m_polydataMapper->SetLookupTable(colorTransferFunction);
	vtkPolyDataNormals_Sptr sGridPolyDataNormal = vtkSmartNew;
	sGridPolyDataNormal->SetInput(m_ContourFilter->GetOutput());
	sGridPolyDataNormal->Update();
	m_polydataMapper->SetInput(sGridPolyDataNormal->GetOutput());
	m_polydataMapper->Update();
	m_actor->SetMapper(m_polydataMapper);
}

void SolidView::Init_Axes()
{
	m_CubeAxesActor = vtkSmartNew;
	m_CubeAxesActor->SetBounds(GetParentDoc()->m_ImageData->GetBounds());
	m_CubeAxesActor->SetCamera(GetParentCtrl()->m_Renderer->GetActiveCamera());
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
	vtkColorTransferFunction_Sptr colorTransferFunction = vtkSmartNew;
	colorTransferFunction->AddRGBPoint(649.0  ,1.0/2,0.0,0.0);
	colorTransferFunction->AddRGBPoint(580.0  ,1.0/2, 165/255/2.0,0.0);
	colorTransferFunction->AddRGBPoint(500.0  ,1.0/2, 1.0/2,0.0);
	colorTransferFunction->AddRGBPoint(400.0  ,0.0, 1.0/2,0.0);
	colorTransferFunction->AddRGBPoint(330.0  ,0.0, 0.5/2, 1.0/2);
	colorTransferFunction->AddRGBPoint(220.0  ,0.0, 0.0, 1.0/2);
	colorTransferFunction->AddRGBPoint(143.0  ,139/255.0/2, 0.0, 1.0/2);
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
	vtkPiecewiseFunction_Sptr compositeOpacity = vtkSmartNew;
	vtkColorTransferFunction_Sptr colorTransferFunction = vtkSmartNew;
	vtkImageShiftScale_Sptr ImageShiftScale = vtkSmartNew;
	vtkSmartVolumeMapper_Sptr volumeMapper = vtkOnlyNew;
	vtkVolumeProperty_Sptr volumeProperty = vtkSmartNew;

	volumeMapper->SetBlendModeToComposite(); // composite first
	volumeMapper->SetInputConnection(GetParentDoc()->m_ImageData->GetProducerPort());

	volumeProperty->ShadeOff();
	volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);

	compositeOpacity->AddPoint(649.0,0.0);
	compositeOpacity->AddPoint(450.0,0.5);
	compositeOpacity->AddPoint(143.0,0.0);
	volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.

	colorTransferFunction->AddRGBPoint(649.0  ,1.0/2,0.0,0.0);
	colorTransferFunction->AddRGBPoint(580.0  ,1.0/2, 165/255/2.0,0.0);
	colorTransferFunction->AddRGBPoint(500.0  ,1.0/2, 1.0/2,0.0);
	colorTransferFunction->AddRGBPoint(400.0  ,0.0, 1.0/2,0.0);
	colorTransferFunction->AddRGBPoint(330.0  ,0.0, 0.5/2, 1.0/2);
	colorTransferFunction->AddRGBPoint(220.0  ,0.0, 0.0, 1.0/2);
	colorTransferFunction->AddRGBPoint(143.0  ,139/255.0/2, 0.0, 1.0/2);
	volumeProperty->SetColor(colorTransferFunction);

	m_volume = vtkSmartNew;
	m_volume->SetMapper(volumeMapper);
	m_volume->SetProperty(volumeProperty);
	GetParentCtrl()->m_Renderer->AddViewProp(m_volume);
}
