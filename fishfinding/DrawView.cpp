#include "DrawView.h"

DrawView::DrawView()
{
}

void DrawView::AddDataList( const nmeaINFOs& infos )
{
	for (nmeaINFOs::const_iterator it = infos.begin();
		it != infos.end(); it++)
	{
		DataPoint data;
		data.E = it->lon;
		data.N = it->lat;
		data.Depth = it->depthinfo.depth_M;
		m_points.push_back(data);
	}
}

void DrawView::AddData( const nmeaINFO& info )
{
	DataPoint data;
	data.E = info.lon;
	data.N = info.lat;
	data.Depth = info.depthinfo.depth_M;
	m_points.push_back(data);
}

void DrawView::Render()
{

}

void DrawView::Clear()
{
	m_points.clear();
}

