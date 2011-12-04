#pragma once
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkPolyDataAlgorithm.h"
#include <vector>


class vtkGridHeightFilter
{
public:
	void SetSeaLineHeight(double v){m_SeaLineHeight = v;}
	double GetSeaLineHeight(){return m_SeaLineHeight;}
	void IgnoreSeaData(bool use){m_IgrSeaLinePoint = use;}
	void IgnoreNullValuePoint(bool use){m_IgrNullValuePoint = use;}
protected:

	double m_SeaLineHeight;	//sea line height, the height is this value will set null value
	bool m_IgrSeaLinePoint;	//ignore the data in sea
	bool m_IgrNullValuePoint;	//ignore the point which value is null value
	std::vector<double> v_PlaneSampleValue;

	double GetHeight(vtkPolyData* heightdata, double px, double pz);
	double GetError(double p1x, double p2x, double p1y, double p2y);
	void BuildPlaneSample(vtkPolyData* TPpolydata, vtkDoubleArray* valuedata, double height);
	void BuildPlaneSample(double* dataset, int n, double height);
};
