#include "StdVtk.h"
#include "Solid.h"
#include <vtkUnsignedCharArray.h>
#include "ConvStr.h"

Solid::Solid()
{
	// vtk init
	m_polydata = vtkSmartPointer<vtkPolyData>::New();
	m_ImageData = vtkSmartPointer<vtkImageData>::New();
	axes = vtkSmartPointer<vtkAxesActor>::New();
	m_AxesWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
	m_Renderer = vtkSmartPointer<vtkRenderer>::New();
	m_RenderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	m_iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	m_VertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	m_contour = vtkSmartPointer<vtkContourFilter>::New();
	m_contour_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_vertex_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_contour_actor = vtkSmartPointer<vtkActor>::New();
	m_vertex_actor = vtkSmartPointer<vtkActor>::New();
	m_lut = vtkSmartPointer<vtkLookupTable>::New();
	m_volumeMapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
	m_volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
	m_volume = vtkSmartPointer<vtkVolume>::New();



	m_volumeMapper=vtkSmartVolumeMapper::New();
	m_volumeMapper->SetBlendModeToComposite(); // composite first
	


	// ofher init
	m_pCtable = new ColorTable();

	// set flow
	m_contour->SetInput(m_ImageData);
	m_contour_mapper->SetInputConnection(m_contour->GetOutputPort());
	m_contour_actor->SetMapper(m_contour_mapper);
	// set flow
	m_VertexFilter->SetInput(m_polydata);
	m_vertex_mapper->SetInputConnection(m_VertexFilter->GetOutputPort());
	m_vertex_actor->SetMapper(m_vertex_mapper);

	m_vertex_mapper->SetLookupTable(m_lut);
	m_contour_mapper->SetLookupTable(m_lut);

	m_Renderer->AddActor(m_vertex_actor);
	m_Renderer->AddActor(m_contour_actor);
	m_Renderer->SetBackground(.1, .2, .3);
	m_RenderWindow->AddRenderer(m_Renderer);
	m_iren->SetRenderWindow(m_RenderWindow);
}

void Solid::SetData( SJCScalarField3d* sf3d )
{
	// backup to use
	m_SJCScalarField3d = sf3d;
	// color
	m_histogram = DWHistogram<double>(m_SJCScalarField3d->begin(), m_SJCScalarField3d->size());
	m_lut->SetTableRange(m_histogram.GetPersentValue(0.01), m_histogram.GetPersentValue(0.99));
	m_lut->Build();
	// data
	m_polydata->Initialize();
	m_ImageData->Initialize();
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkFloatArray> volcolors = vtkSmartPointer<vtkFloatArray>::New();
	vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
	colors->SetNumberOfComponents(3);
	colors->SetName("Colors");
	uint i, j, k,  kOffset, jOffset, offset;
	const uint x_len = sf3d->NumX(),
		y_len = sf3d->NumY(),
		z_len = sf3d->NumZ();
	m_ImageData->SetDimensions(x_len, y_len, z_len);
	for(k=0;k<z_len;k++)
	{
		kOffset = k*y_len*x_len;
		for(j=0; j<y_len; j++)
		{
			jOffset = j*x_len;
			for(i=0;i<x_len;i++)
			{
				offset = i + jOffset + kOffset;
				points->InsertNextPoint(i, j, k);
				double dcolor[3];
				m_lut->GetColor(sf3d->Value(i, j, k), dcolor);
				volcolors->InsertTuple1(offset, sf3d->Value(i, j, k));
				unsigned char color[3];
				for(unsigned int j = 0; j < 3; j++)
				{
					color[j] = static_cast<unsigned char>(255.0 * dcolor[j]);
				}
				//OutputDebugString( (ConvStr::GetWstr(color[0])+L" "+ConvStr::GetWstr(color[1])+L" "+ConvStr::GetWstr(color[2])+L"\n").c_str() );
				colors->InsertNextTupleValue(color);
			}
		}
	}
	m_ImageData->GetPointData()->SetScalars(volcolors);
	m_polydata->SetPoints(points);
	m_polydata->GetPointData()->SetScalars(colors);

	m_pCtable->clear();
	m_pCtable->push_back(m_histogram.GetPersentValue(1),Color4(255, 0, 0,0));	// ¬õ
	m_pCtable->push_back(m_histogram.GetPersentValue(0.75),Color4(255, 128, 0,0));	// ¾í
	m_pCtable->push_back(m_histogram.GetPersentValue(0.625),Color4(255, 255, 0,0));	// ¶À
	m_pCtable->push_back(m_histogram.GetPersentValue(0.5),Color4(0, 255, 0,0));	// ºñ
	m_pCtable->push_back(m_histogram.GetPersentValue(0.375),Color4(0, 255, 255,0));	// «C
	m_pCtable->push_back(m_histogram.GetPersentValue(0.25),Color4(0, 0, 255,0));	// ÂÅ
	m_pCtable->push_back(m_histogram.GetPersentValue(0.125),Color4(102, 0, 255,0));	// ÀQ
	m_pCtable->push_back(m_histogram.GetPersentValue(0),Color4(167, 87, 168,0));	// µµ
}

void Solid::SetIsoSurface( double isolevel, bool show /*= true*/ )
{
	// set data
	m_contour->SetInput(m_ImageData);
	m_contour->SetValue(1, isolevel);
	m_contour->Update();
	if (show)
	{
		m_Renderer->AddActor(m_contour_actor);
	}
	else
	{
		m_Renderer->RemoveActor(m_contour_actor);
	}
}

void Solid::SetVertex( bool show /*= true*/ )
{
	// set data
	m_VertexFilter->SetInput(m_polydata);
	m_VertexFilter->Update();
	if (show)
	{
		m_Renderer->AddActor(m_vertex_actor);
	}
	else
	{
		m_Renderer->RemoveActor(m_vertex_actor);
	}
}

void Solid::ReSize( int w, int h )
{
	m_RenderWindow->SetSize(w, h);
}

void Solid::SetHwnd( HWND hwnd )
{
	m_RenderWindow->SetParentId(hwnd);
}

void Solid::Render()
{
	m_RenderWindow->Render();
}

void Solid::SetColorTable( ColorTable* ct )
{
	if (m_pCtable == NULL && m_pCtable != ct)
		delete m_pCtable;
	m_pCtable = ct;
}

Solid::~Solid()
{
	if (m_pCtable)
		delete m_pCtable;
}

void Solid::SetVolume()
{
	double range[2] = {0.0, 125.0};

	vtkImageShiftScale *t=vtkImageShiftScale::New();
	t->SetInput(m_ImageData);
	t->SetShift(-range[0]);
	double magnitude=range[1]-range[0];
	if(magnitude==0.0)
	{
		magnitude=1.0;
	}
	t->SetScale(255.0/magnitude);
	t->SetOutputScalarTypeToUnsignedChar();
	t->Update();
	m_volumeMapper->SetInputConnection(t->GetOutputPort());
}