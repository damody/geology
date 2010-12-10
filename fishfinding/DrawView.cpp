#include "StdVtkWx.h"
#include "DrawView.h"

DrawView::DrawView()
{
	m_focus_height = 3;
	vtkSmartNew_Initialize(m_hs_points);
	vtkSmartNew_Initialize(m_depth_points);
	vtkSmartNew_Initialize(m_hs_vertices);
	vtkSmartNew_Initialize(m_depth_vertices);
	vtkSmartNew_Initialize(m_imagedata);
	vtkSmartNew_Initialize(m_hs_poly);
	vtkSmartNew_Initialize(m_depth_poly);
	vtkSmartNew(m_hs_Mapper);
	vtkSmartNew(m_depth_Mapper);
	vtkSmartNew(m_hs_Actor);
	vtkSmartNew(m_depth_Actor);
	vtkSmartNew(m_Renderer);
	vtkSmartNew(m_RenderWindow);
	vtkSmartNew(m_Camera);
	vtkSmartNew(m_Axes);
	vtkSmartNew(m_Append_hs);
	vtkSmartNew(m_Append_depth);
	vtkSmartNew(m_WindowInteractor);
	vtkSmartNew(m_Axes_widget);

	m_RenderWindow->AddRenderer(m_Renderer);
	m_WindowInteractor->SetRenderWindow(m_RenderWindow);
	m_Renderer->SetActiveCamera(m_Camera);
	m_Renderer->SetBackground(.1, .2, .3);
	m_Axes_widget->SetOutlineColor( 0.8300, 0.6700, 0.5300 );
	m_Axes_widget->SetOrientationMarker( m_Axes );
	m_Axes_widget->SetInteractor( m_WindowInteractor );
	m_Axes_widget->On();
	// horizontal surface vertex
	m_hs_poly->SetPoints(m_hs_points);
	m_hs_poly->SetVerts(m_hs_vertices);
	m_Append_hs->SetInput(m_hs_poly);
	m_hs_Mapper->SetInputConnection(m_Append_hs->GetOutputPort());
	m_hs_Actor->SetMapper(m_hs_Mapper);
	m_Renderer->AddActor(m_hs_Actor);
	// depth surface vertex
	m_depth_poly->SetPoints(m_depth_points);
	m_depth_poly->SetVerts(m_depth_vertices);
	m_Append_depth->SetInput(m_depth_poly);
	m_depth_Mapper->SetInputConnection(m_Append_depth->GetOutputPort());
	m_depth_Actor->SetMapper(m_depth_Mapper);
	//m_depth_Actor->GetProperty()->SetPointSize(10);
	m_Renderer->AddActor(m_depth_Actor);

	
}

void DrawView::AddDataList( const nmeaINFOs& infos )
{
	for (nmeaINFOs::const_iterator it = infos.begin();
		it != infos.end(); it++)
	{
		DataPoint data;
		data.E = it->lon;
		data.N = it->lat;
		data.depth = it->depthinfo.depth_M;
		m_raw_points.push_back(data);
		const float phs[3] = {data.E, data.N, 0};
		const float pdep[3] = {data.E, data.N, -data.depth};
		vtkIdType pid[1];
		pid[0] = m_hs_points->InsertNextPoint(phs);
		m_hs_vertices->InsertNextCell(1,pid);
		m_Append_hs->SetInput(m_hs_poly);
		pid[0] = m_depth_points->InsertNextPoint(pdep);
		m_depth_vertices->InsertNextCell(1,pid);
		m_Append_depth->SetInput(m_depth_poly);
	}
}

void DrawView::AddData( const nmeaINFO& info )
{
	DataPoint data;
	data.E = info.lon;
	data.N = info.lat;
	data.depth = info.depthinfo.depth_M;
	m_raw_points.push_back(data);
	const float phs[3] = {data.E, data.N, 0};
	const float pdep[3] = {data.E, data.N, -data.depth};
	vtkIdType pid[1];
	pid[0] = m_hs_points->InsertNextPoint(phs);
	m_hs_vertices->InsertNextCell(1,pid);
	m_Append_hs->AddInput(m_hs_poly);
	pid[0] = m_depth_points->InsertNextPoint(pdep);
	m_depth_vertices->InsertNextCell(1,pid);
	m_Append_depth->AddInput(m_depth_poly);
}

void DrawView::Render()
{
	m_Append_hs->Update();
	m_Append_depth->Update();
	m_RenderWindow->Render();
}
void DrawView::Clear()
{
	m_raw_points.clear();
	vtkSmartNew_Initialize(m_hs_points);
	vtkSmartNew_Initialize(m_depth_points);
	vtkSmartNew_Initialize(m_hs_vertices);
	vtkSmartNew_Initialize(m_depth_vertices);
	m_hs_poly->SetPoints(m_hs_points);
	m_depth_poly->SetPoints(m_depth_points);
	m_hs_poly->SetVerts(m_hs_vertices);
	m_depth_poly->SetVerts(m_depth_vertices);
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
	//vtkSmartNew_Initialize(m_hs_vertices);
	//vtkSmartNew_Initialize(m_hs_points);
	int i, j, k,  kOffset, jOffset, offset;
	const int x_len = 5,
		y_len = 5,
		z_len = 5;
	for(k=0;k<z_len;k++)
	{
		kOffset = k*y_len*x_len;
		for(j=0; j<y_len; j++)
		{
			jOffset = j*x_len;
			for(i=0;i<x_len;i++)
			{
				offset = i + jOffset + kOffset;
				const float p[3] = {i, j, k};
				vtkIdType pid[1];
				pid[0] = m_hs_points->InsertNextPoint(i, j, k);
				m_hs_vertices->InsertNextCell(1,pid);
			}
		}
	}
	m_Append_hs->AddInput(m_hs_poly);
}

void DrawView::FocusLast()
{
	if (!m_raw_points.empty())
	{
		DataPoint &data = m_raw_points.back();
		m_Camera->SetPosition(data.E, data.N, m_focus_height/10.0);
		double pos[3];
		m_Camera->GetPosition(pos);
		m_Camera->SetFocalPoint(pos[0], pos[1], pos[2]-1);
		m_Camera->SetFocalDisk(data.depth+1);
		m_Camera->UpdateViewport(m_Renderer);
	}
}

void DrawView::NormalLook( double angle )
{
	double pos[3];
	m_Camera->SetViewUp(0, 1, 0);
	m_Camera->GetPosition(pos);
	m_Camera->SetPosition(pos[0], pos[1], m_focus_height/10.0);
	m_Camera->SetFocalPoint(pos[0], pos[1], pos[2]-1);
	m_Camera->SetFocalDisk(pos[2]+1);
}

