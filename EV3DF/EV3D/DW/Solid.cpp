#include "StdVtk.h"
#include "Solid.h"

#include "ConvStr.h"


Solid::Solid()
{
	// vtk init
	m_points = vtkSmartPointer<vtkPoints>::New();
	m_volcolors = vtkSmartPointer<vtkFloatArray>::New();
	m_colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
	m_polydata = vtkSmartPointer<vtkPolyData>::New();
	m_ImageData = vtkSmartPointer<vtkImageData>::New();
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
	m_chiplut = vtkSmartPointer<vtkLookupTable>::New();
	m_volumeMapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
	m_volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
	m_volume = vtkSmartPointer<vtkVolume>::New();
	m_outline = vtkSmartPointer<vtkOutlineFilter>::New();
	m_volumeMapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
	m_outlineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_outlineActor = vtkSmartPointer<vtkActor>::New();
	m_camera = vtkSmartPointer<vtkCamera>::New();
	m_axes_widget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
	m_axes = vtkSmartPointer<vtkAxesActor>::New();
	m_planeWidgetX = vtkSmartPointer<vtkImagePlaneWidget>::New();
	m_planeWidgetY = vtkSmartPointer<vtkImagePlaneWidget>::New();
	m_planeWidgetZ = vtkSmartPointer<vtkImagePlaneWidget>::New();
	m_vol = vtkSmartPointer<vtkImageData>::New();
	m_ImageShiftScale = vtkSmartPointer<vtkImageShiftScale>::New();
	m_CompositeOpacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
	m_ColorTransferFun = vtkSmartPointer<vtkColorTransferFunction>::New();

	m_chiplut->Build();
	m_volumeMapper->SetBlendModeToComposite(); // composite first
	// ofher init
	m_pCtable = new ColorTable();
	// contour flow
	m_contour->SetInput(m_ImageData);
	m_contour_mapper->SetInputConnection(m_contour->GetOutputPort());
	m_contour_actor->SetMapper(m_contour_mapper);
	m_outlineMapper->SetLookupTable(m_lut);
	// outline flow
	m_outline->SetInput(m_ImageData);
	m_outlineMapper->SetInputConnection(m_outline->GetOutputPort());
	m_outlineActor->SetMapper( m_outlineMapper);
	// Vertex flow
	m_VertexFilter->SetInput(m_polydata);
	m_vertex_mapper->SetInputConnection(m_VertexFilter->GetOutputPort());
	m_vertex_actor->SetMapper(m_vertex_mapper);

	m_vertex_mapper->SetLookupTable(m_lut);
	m_contour_mapper->SetLookupTable(m_lut);

	m_Renderer->AddActor(m_outlineActor);
	m_Renderer->AddActor(m_vertex_actor);
	m_Renderer->AddActor(m_contour_actor);
	m_Renderer->SetBackground(.1, .2, .3);
	m_RenderWindow->AddRenderer(m_Renderer);
	m_iren->SetRenderWindow(m_RenderWindow);
	m_Renderer->SetActiveCamera(m_camera);

	m_axes_widget->SetOutlineColor( 0.9300, 0.5700, 0.1300 );
	m_axes_widget->SetOrientationMarker( m_axes );
	m_axes_widget->SetInteractor( m_iren );
	m_axes_widget->On();

	m_planeWidgetX->SetLeftButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_planeWidgetX->SetMiddleButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_planeWidgetX->SetRightButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_planeWidgetY->SetLeftButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_planeWidgetY->SetMiddleButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_planeWidgetY->SetRightButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_planeWidgetZ->SetLeftButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_planeWidgetZ->SetMiddleButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_planeWidgetZ->SetRightButtonAction(vtkImagePlaneWidget::VTK_CURSOR_ACTION);
	m_planeWidgetX->SetLookupTable(m_chiplut);
	m_planeWidgetY->SetLookupTable(m_chiplut);
	m_planeWidgetZ->SetLookupTable(m_chiplut);
	m_volumeProperty->SetScalarOpacity(m_CompositeOpacity);
	m_volumeProperty->SetColor(m_ColorTransferFun);
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
	m_points->Initialize();
	m_volcolors->Initialize();
	m_colors->Initialize();
	
	m_colors->SetNumberOfComponents(3);
	m_colors->SetName("Colors");
	uint i, j, k,  kOffset, jOffset, offset;
	const uint x_len = m_SJCScalarField3d->NumX(),
		y_len = m_SJCScalarField3d->NumY(),
		z_len = m_SJCScalarField3d->NumZ();
	m_camera->SetPosition(0, 0, (x_len+y_len+z_len)/3.0);
	m_camera->SetFocalPoint(x_len/2.0, y_len/2.0, z_len/2.0);
	for(k=0;k<z_len;k++)
	{
		kOffset = k*y_len*x_len;
		for(j=0; j<y_len; j++)
		{
			jOffset = j*x_len;
			for(i=0;i<x_len;i++)
			{
				offset = i + jOffset + kOffset;
				m_volcolors->InsertTuple1(offset, sf3d->Value(i, j, k));
				m_points->InsertNextPoint(i, j, k);
				double dcolor[3];
				m_lut->GetColor(sf3d->Value(i, j, k), dcolor);
				unsigned char color[3];
				for(unsigned int j = 0; j < 3; j++)
				{
					color[j] = static_cast<unsigned char>(255.0 * dcolor[j]);
				}
				//OutputDebugString( (ConvStr::GetWstr(color[0])+L" "+ConvStr::GetWstr(color[1])+L" "+ConvStr::GetWstr(color[2])+L"\n").c_str() );
				m_colors->InsertNextTupleValue(color);
			}
		}
	}
	m_vol->SetDimensions(x_len, y_len, z_len);
	m_vol->GetPointData()->SetScalars(m_volcolors);
	m_ImageData->SetDimensions(x_len, y_len, z_len);
	m_ImageData->GetPointData()->SetScalars(m_volcolors);
	m_polydata->SetPoints(m_points);
	m_polydata->GetPointData()->SetScalars(m_colors);

// 	m_pCtable->clear();
// 	m_pCtable->push_back(m_histogram.GetPersentValue(1),Color4(255, 0, 0,0));	// ¬õ
// 	m_pCtable->push_back(m_histogram.GetPersentValue(0.75),Color4(255, 128, 0,0));	// ¾í
// 	m_pCtable->push_back(m_histogram.GetPersentValue(0.625),Color4(255, 255, 0,0));	// ¶À
// 	m_pCtable->push_back(m_histogram.GetPersentValue(0.5),Color4(0, 255, 0,0));	// ºñ
// 	m_pCtable->push_back(m_histogram.GetPersentValue(0.375),Color4(0, 255, 255,0));	// «C
// 	m_pCtable->push_back(m_histogram.GetPersentValue(0.25),Color4(0, 0, 255,0));	// ÂÅ
// 	m_pCtable->push_back(m_histogram.GetPersentValue(0.125),Color4(102, 0, 255,0));	// ÀQ
// 	m_pCtable->push_back(m_histogram.GetPersentValue(0),Color4(167, 87, 168,0));	// µµ

	// plane
	m_planeWidgetX->SetInteractor(m_iren);
	m_planeWidgetX->RestrictPlaneToVolumeOn();
	m_planeWidgetX->SetInput(m_vol);
	m_planeWidgetX->SetPlaneOrientationToXAxes();
	
	m_planeWidgetY->SetInteractor(m_iren);
	m_planeWidgetY->RestrictPlaneToVolumeOn();
	m_planeWidgetY->SetInput(m_vol);
	m_planeWidgetY->SetPlaneOrientationToYAxes();
	
	m_planeWidgetZ->SetInteractor(m_iren);
	m_planeWidgetZ->RestrictPlaneToVolumeOn();
	m_planeWidgetZ->SetInput(m_vol);
	m_planeWidgetZ->SetPlaneOrientationToZAxes();
	
	m_CompositeOpacity->AddPoint(0, 0.0);
	m_CompositeOpacity->AddPoint(50.0, 0.5);
	m_CompositeOpacity->AddPoint(85.0, 0.8);
	m_CompositeOpacity->AddPoint(125, 1.0);

	m_ColorTransferFun->AddRGBPoint(0.0, 1.0,0.0,0.0);
	m_ColorTransferFun->AddRGBPoint(50.0, 0.0,1.0,0.0);
	m_ColorTransferFun->AddRGBPoint(85.0, 0.0,0.0,1.0);
	m_ColorTransferFun->AddRGBPoint(125.0, 1.0,1.0,1.0);

	m_ImageShiftScale->SetInput(m_vol);
	m_ImageShiftScale->SetScale(1);
	m_ImageShiftScale->SetOutputScalarTypeToUnsignedChar();
	m_ImageShiftScale->Update();
	m_volumeMapper->SetInputConnection(m_ImageShiftScale->GetOutputPort());
	m_volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);
	m_volume->SetMapper(m_volumeMapper);
	m_volume->SetProperty(m_volumeProperty);
	//m_Renderer->AddViewProp(m_volume);
	
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
	m_ImageShiftScale->SetInput(m_ImageData);
	m_ImageShiftScale->SetScale(1);
	m_ImageShiftScale->SetOutputScalarTypeToUnsignedChar();
	m_ImageShiftScale->Update();
	m_volumeMapper->SetInputConnection(m_ImageShiftScale->GetOutputPort());
}

void Solid::SetSlice( AXIS axes, double percent )
{
	const uint x_len = m_SJCScalarField3d->NumX(),
		y_len = m_SJCScalarField3d->NumY(),
		z_len = m_SJCScalarField3d->NumZ();
	switch (axes)
	{
	case USE_X:
		m_planeWidgetX->SetSlicePosition(percent*x_len/100.0);
		break;
	case USE_Y:
		m_planeWidgetY->SetSlicePosition(percent*y_len/100.0);
		break;
	case USE_Z:
		m_planeWidgetZ->SetSlicePosition(percent*z_len/100.0);
		break;
	}
}

void Solid::EnableSlice( AXIS axes )
{
	switch (axes)
	{
	case USE_X:
		m_planeWidgetX->On();
		break;
	case USE_Y:
		m_planeWidgetY->On();
		break;
	case USE_Z:
		m_planeWidgetZ->On();
		break;
	}
}

void Solid::DisableSlice( AXIS axes )
{
	switch (axes)
	{
	case USE_X:
		m_planeWidgetX->Off();
		break;
	case USE_Y:
		m_planeWidgetY->Off();
		break;
	case USE_Z:
		m_planeWidgetZ->Off();
		break;
	}
}