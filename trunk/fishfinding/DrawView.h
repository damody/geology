#pragma once
#include "NmeaCell.h"
#include "TShape.hpp"

class DrawView
{
public:
	struct DataPoint
	{
		double N, E, Depth;
	};
	typedef std::vector<DataPoint> DataPoints;
	DrawView();
	void AddDataList(const nmeaINFOs& infos);
	void AddData(const nmeaINFO& info);
	void Clear();
	void Render();
	void SetRect(const Rectf& rect) {m_area = rect;}
private:
	DataPoints m_points;
	Rectf m_area;
};
