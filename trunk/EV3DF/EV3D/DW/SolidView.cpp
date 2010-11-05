#include "StdVtk.h"
#include "SolidView.h"
#include "SolidDoc.h"
#include "SEffectSetting.h"

SolidView::SolidView(SolidDoc_Sptr& Doc)
:m_visable(true)
{
	vtkSmartNew(m_actor);
	vtkSmartNew(m_ltable);
	vtkSmartNew(m_polydataMapper);
	m_ParentDoc = Doc;
	
}

SolidView::~SolidView(void)
{
}


void SolidView::SetRenderTarget( vtkRenderer_Sptr renderer )
{
	m_Renderer = renderer;
}

void SolidView::SetVisable( bool show )
{
	m_visable = show;
	if (show)
		m_Renderer->AddActor(m_actor);
	else
		m_Renderer->RemoveActor(m_actor);
}

void SolidView::SetSetting( SEffect_Setting_Sptr setting )
{
	m_SEffect_Setting = setting;
	switch (m_SEffect_Setting->m_Type)
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
	}
	SetVisable(m_visable);
}
