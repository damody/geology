#include "StdVtkWx.h"
#include "DrawView.h"

DrawView::DrawView()
{
	m_focus_height = 3;
	vtkSmartNew_Initialize(m_hs_points);
	vtkSmartNew_Initialize(m_de_points);
	vtkSmartNew_Initialize(m_hs_vertices);
	vtkSmartNew_Initialize(m_de_vertices);
	vtkSmartNew_Initialize(m_hs_colors);
	vtkSmartNew_Initialize(m_de_colors);
	vtkSmartNew_Initialize(m_hs_lines);
	vtkSmartNew_Initialize(m_de_lines);
	vtkSmartNew_Initialize(m_hs_poly);
	vtkSmartNew_Initialize(m_de_poly);
	vtkSmartNew_Initialize(m_imagedata);

	memset(m_ucHSColor, 0, sizeof(m_ucHSColor));
	memset(m_ucDEColor, 0, sizeof(m_ucDEColor));
	m_hs_colors->SetNumberOfComponents(3);
	m_hs_colors->SetName ("Colors");
	m_de_colors->SetNumberOfComponents(3);
	m_de_colors->SetName ("Colors");

	vtkSmartNew(m_Append_hs);
	vtkSmartNew(m_Append_de);
	vtkSmartNew(m_hs_Mapper);
	vtkSmartNew(m_de_Mapper);
	vtkSmartNew(m_hs_Actor);
	vtkSmartNew(m_de_Actor);
	vtkSmartNew(m_Renderer);
	vtkSmartNew(m_RenderWindow);
	vtkSmartNew(m_Camera);
	vtkSmartNew(m_Axes);
	vtkSmartNew(m_WindowInteractor);
	vtkSmartNew(m_Axes_widget);
	vtkSmartNew(m_style);

	m_RenderWindow->AddRenderer(m_Renderer);
	m_WindowInteractor->SetRenderWindow(m_RenderWindow);
	m_WindowInteractor->SetInteractorStyle( m_style );
	m_Renderer->SetActiveCamera(m_Camera);
	m_Renderer->SetBackground(.1, .2, .3);
	m_Axes_widget->SetOutlineColor( 0.8300, 0.6700, 0.5300 );
	m_Axes_widget->SetOrientationMarker( m_Axes );
	m_Axes_widget->SetInteractor( m_WindowInteractor );
	m_Axes_widget->On();
	// horizontal surface vertex
	m_hs_poly->SetPoints(m_hs_points);
	m_hs_poly->SetVerts(m_hs_vertices);
	m_hs_poly->SetLines(m_hs_lines);
	m_Append_hs->SetInput(m_hs_poly);
	m_hs_Mapper->SetInputConnection(m_Append_hs->GetOutputPort());
	m_hs_Actor->SetMapper(m_hs_Mapper);
	m_Renderer->AddActor(m_hs_Actor);
	// depth surface vertex
	m_de_poly->SetPoints(m_de_points);
	m_de_poly->SetVerts(m_de_vertices);
	m_de_poly->SetLines(m_de_lines);
	m_Append_de->SetInput(m_de_poly);
	m_de_Mapper->SetInputConnection(m_Append_de->GetOutputPort());
	m_de_Actor->SetMapper(m_de_Mapper);
	m_Renderer->AddActor(m_de_Actor);
}

void DrawView::Clear()
{
	m_hs_points->Initialize();
	m_de_points->Initialize();
	m_hs_vertices->Initialize();
	m_de_vertices->Initialize();
	m_hs_colors->Initialize();
	m_de_colors->Initialize();
	m_hs_lines->Initialize();
	m_de_lines->Initialize();
	m_raw_points.clear();
	m_Append_hs->RemoveAllInputs();
 	m_Append_hs->AddInput(m_hs_poly);
	m_Append_de->RemoveAllInputs();
 	m_Append_de->AddInput(m_de_poly);
}

void DrawView::AddDataList( const nmeaINFOs& infos )
{
	for (nmeaINFOs::const_iterator it = infos.begin();
		it != infos.end(); it++)
	{
		AddData(*it);
	}
}

void DrawView::AddData( const nmeaINFO& info )
{
	DataPoint data;
	data.E = info.lon;
	data.N = info.lat;
	data.depth = info.depthinfo.depth_M;
	if (data.E == 0 && data.N == 0)
		return;

	m_raw_points.push_back(data);
	int len = m_raw_points.size();
	const float phs[3] = {data.E, data.N, 0};
	const float pdep[3] = {data.E, data.N, -data.depth};
	vtkIdType pid[1];
	pid[0] = m_hs_points->InsertNextPoint(phs);
	m_hs_vertices->InsertNextCell(1,pid);
	m_hs_colors->InsertNextTupleValue(m_ucHSColor);
	m_hs_poly->GetPointData()->SetScalars(m_hs_colors);
	m_Append_hs->RemoveAllInputs();
	m_Append_hs->AddInput(m_hs_poly);

	pid[0] = m_de_points->InsertNextPoint(pdep);
	m_de_vertices->InsertNextCell(1,pid);
	m_de_colors->InsertNextTupleValue(m_ucDEColor);
	m_de_poly->GetPointData()->SetScalars(m_de_colors);
	m_Append_de->RemoveAllInputs();
	m_Append_de->AddInput(m_de_poly);
	if (len>1)
	{
		vtkLine_Sptr line;
		vtkSmartNew(line);
		line->GetPointIds()->SetId(0, len-2);
		line->GetPointIds()->SetId(1, len-1);
		m_hs_lines->InsertNextCell(line);
		vtkLine_Sptr line2;
		vtkSmartNew(line2);
		line2->GetPointIds()->SetId(0, len-2);
		line2->GetPointIds()->SetId(1, len-1);
		m_de_lines->InsertNextCell(line2);
	}
}

void DrawView::Render()
{
	m_RenderWindow->Render();
}
void DrawView::SetHwnd( HWND hwnd )
{
	m_RenderWindow->SetParentId(hwnd);
}

void DrawView::ReSize( int w, int h )
{
	m_RenderWindow->SetSize(w, h);
}

void DrawView::AddTest()
{
// 	m_hs_points->Initialize();
// 	m_hs_vertices->Initialize();
// 	m_hs_lines->Initialize();
// 	int i, j, k,  kOffset, jOffset, offset;
// 	static int q=10;
// 	q--;
// 	const int x_len = 3+q,
// 		y_len = 3+q,
// 		z_len = 3+q;
// 	for(k=0;k<z_len;k++)
// 	{
// 		kOffset = k*y_len*x_len;
// 		for(j=0; j<y_len; j++)
// 		{
// 			jOffset = j*x_len;
// 			for(i=0;i<x_len;i++)
// 			{
// 				offset = i + jOffset + kOffset;
// 				const float p[3] = {i, j, k};
// 				vtkIdType pid[1];
// 				pid[0] = m_hs_points->InsertNextPoint(i, j, k);
// 				m_hs_vertices->InsertNextCell(1,pid);
// 				m_hs_colors->InsertNextTupleValue(m_ucHSColor);
// 				vtkLine_Sptr line;
// 				vtkSmartNew(line);
// 				line->GetPointIds()->SetId(0, offset);
// 				line->GetPointIds()->SetId(1, offset+1);
// 				m_hs_lines->InsertNextCell(line);
// 			}
// 		}
// 	}
// 	m_hs_poly->SetLines(m_hs_lines);
// 	m_hs_poly->GetPointData()->SetScalars(m_hs_colors);
// 	m_Append_hs->RemoveAllInputs();
// 	m_Append_hs->AddInput(m_hs_poly);
}

void DrawView::FocusLast()
{
	if (!m_raw_points.empty())
	{
		DataPoint &data = m_raw_points.back();
		m_Camera->SetPosition(data.E, data.N, m_focus_height/10.0);
		double pos[3];
		m_Camera->GetPosition(pos);
		m_Camera->SetFocalPoint(pos[0], pos[1], -data.depth);
		m_Camera->SetFocalDisk(data.depth+100);
		m_Camera->UpdateViewport(m_Renderer);
	}
}

void DrawView::NormalLook( double angle )
{
	double pos[3];
	m_Camera->SetViewUp(0, 1, 0);
	m_Camera->GetPosition(pos);
	m_Camera->SetPosition(pos[0], pos[1], m_focus_height/10.0);
	m_Camera->SetFocalPoint(pos[0], pos[1], pos[2]-2);
	m_Camera->SetFocalDisk(pos[2]+1);
}

void DrawView::SetHSColor( unsigned char r, unsigned char g, unsigned char b )
{
	m_ucHSColor[0] = r;
	m_ucHSColor[1] = g;
	m_ucHSColor[2] = b;
}

void DrawView::SetDEColor( unsigned char r, unsigned char g, unsigned char b )
{
	m_ucDEColor[0] = r;
	m_ucDEColor[1] = g;
	m_ucDEColor[2] = b;
}

void DrawView::SetPointSize( int size )
{
	assert(size>=0);
	m_hs_Actor->GetProperty()->SetPointSize(size);
	m_de_Actor->GetProperty()->SetPointSize(size);
}

DrawView::DataPoint DrawView::GetLastData()
{
	if (!m_raw_points.empty())
		return m_raw_points.back();
	return DataPoint();
}

int DrawView::GetTotal()
{
	return (int)m_raw_points.size();
}

