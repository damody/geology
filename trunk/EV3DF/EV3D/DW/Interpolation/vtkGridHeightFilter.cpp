// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
#include "vtkGridHeightFilter.h"

double vtkGridHeightFilter::GetHeight( vtkPolyData* heightdata, double px, double pz )
{
	int MinIndex = 0;
	double p3[3];
	heightdata->GetPoint(0, p3);
	double minerr = GetError(p3[0], px, p3[2], pz);
	double err;
	for (int i=1; i<heightdata->GetNumberOfPoints(); i++)
	{
		heightdata->GetPoint(i, p3);
		err = GetError(p3[0], px, p3[2], pz);
		if (err < minerr)
		{
			MinIndex = i;
			minerr = err;
		}
	}
	heightdata->GetPoint(MinIndex, p3);
	return p3[1];
}


double vtkGridHeightFilter::GetError( double p1x, double p2x, double p1y, double p2y )
{
	return fabs(p1x-p2x) + fabs(p1y-p2y);
}

void vtkGridHeightFilter::BuildPlaneSample( vtkPolyData* Temptdata, vtkDoubleArray* valuedata , double height)
{
	double pos[3];
	double newTemperature;
	double t, deltaT;
	int s = Temptdata->GetNumberOfPoints();
	v_PlaneSampleValue.resize(Temptdata->GetNumberOfPoints()/2);
	for (int i=0; i<(int)Temptdata->GetNumberOfPoints(); i+=2)
	{
		Temptdata->GetPoint(i, pos);
		t = valuedata->GetValue(i);
		if (pos[1] == -3500)
			continue;
		deltaT = (valuedata->GetValue(i+1)-t) / (pos[1] + 3500) * 100;
		if (pos[1] < height)		//plane height > sample point height
		{
			v_PlaneSampleValue[i/2] = 0;
		}
		else
		{
			newTemperature = (pos[1] - height)/100 * deltaT + t;
			v_PlaneSampleValue[i/2] = newTemperature;
		}
	}
}



void vtkGridHeightFilter::BuildPlaneSample( double* dataset, int n, double height )
{
	double pos[3];
	double newTemperature;
	double t, deltaT;
	v_PlaneSampleValue.resize(n/2);
	for (int i=0; i<n; i+=2, dataset+=4)
	{
		pos[0] = dataset[0];
		pos[1] = dataset[1];
		pos[2] = dataset[2];
		t = dataset[3];
		if (pos[1] == -3500)
			continue;
		deltaT = (*(dataset+7)-t) / (pos[1] + 3500) * 100;
		if (pos[1] < height)		//plane height > sample point height
		{
			v_PlaneSampleValue[i/2] = 0;
		}
		else
		{
			newTemperature = (pos[1] - height)/100 * deltaT + t;
			v_PlaneSampleValue[i/2] = newTemperature;
		}
	}

}
// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
