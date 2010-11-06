#include "StdVtk.h"
#include "SolidView.h"
#include "SolidDoc.h"
#include "SolidCtrl.h"
#include "SEffect.h"
#include "BoxArea.h"

SolidView::SolidView(SolidCtrl *ParentCtrl, SolidDoc_Sptr Doc):m_ParentCtrl(ParentCtrl)
{
	vtkSmartNew(m_actor);
	vtkSmartNew(m_polydataMapper);
	vtkSmartNew(m_Camera);
	GetParentCtrl()->m_Renderer->SetActiveCamera(m_Camera);
	m_ParentDoc = Doc;
}

void SolidView::SetVisable( bool show )
{
	m_SEffect->m_Visable = show;
	if (show)
		GetParentCtrl()->m_Renderer->AddActor(m_actor);
	else
		GetParentCtrl()->m_Renderer->RemoveActor(m_actor);
}

void SolidView::SetEffect( SEffect_Sptr effect )
{
	m_SEffect = effect;
	BoxArea_Sptr area = GetParentDoc()->m_area;
	m_Camera->SetPosition(0, 0, (area->m_numX+area->m_numY+area->m_numZ)/3.0);
	m_Camera->SetFocalPoint(area->m_numX/2.0, area->m_numY/2.0, area->m_numZ/2.0);
	switch (m_SEffect->GetType())
	{
	case SEffect::BOUNDING_BOX:
		{
			vtkOutlineFilter_Sptr bounding_box;
			vtkSmartNew(bounding_box);
			bounding_box->SetInput(GetParentDoc()->m_ImageData);
			m_polydataMapper->SetInputConnection(bounding_box->GetOutputPort());
			m_actor->SetMapper(m_polydataMapper);
		}
		break;
	case SEffect::VERTEX:
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
		break;
	case SEffect::CONTOUR:
		{
			vtkContourFilter_Sptr ContourFilter;
			vtkSmartNew(ContourFilter);
			ContourFilter->SetInput(GetParentDoc()->m_ImageData);
			m_polydataMapper->SetInputConnection(ContourFilter->GetOutputPort());
			m_actor->SetMapper(m_polydataMapper);
		}
		break;
	case SEffect::AXES:
		{
			MESSAGE("not Implementation");
		}
		break;
	case SEffect::PLANE_CHIP:
		{
			vtkImagePlaneWidget_Sptr ImagePlane;
			vtkSmartNew(ImagePlane);
// 			ImagePlane->SetLeftButtonAction(-1);
// 			ImagePlane->SetMiddleButtonAction(-1);
// 			ImagePlane->SetRightButtonAction(-1);
			ImagePlane->SetInteractor(GetParentCtrl()->m_WindowInteractor);
			ImagePlane->RestrictPlaneToVolumeOn();
			ImagePlane->SetInput(GetParentDoc()->m_ImageData);
			ImagePlane->SetPlaneOrientationToXAxes();
			vtkLookupTable_Sptr lut;
			vtkSmartNew(lut);
			lut->SetTableRange(GetParentDoc()->m_histogram.GetPersentValue(0.01), 
				GetParentDoc()->m_histogram.GetPersentValue(0.99));
			lut->Build();
			ImagePlane->SetLookupTable(lut);
			ImagePlane->SetSlicePosition(10*GetParentDoc()->m_area->m_numX/100.0);
			ImagePlane->On();
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
	case SEffect::VOLUME_RENDER:
		{
			MESSAGE("not Implementation");
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
