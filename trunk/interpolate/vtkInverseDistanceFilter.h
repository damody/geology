// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include <vector>

//Limited Inverse Distance Filter
//ignore terrain height
class vtkInverseDistanceFilter : public vtkInterpolationGridingPolyDataFilter 
{
public:
	vtkTypeMacro(vtkInverseDistanceFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkInverseDistanceFilter *New();

	//set and get power of weight
	void SetPowerValue(double v);
	double GetPowerValue();
protected:
	vtkInverseDistanceFilter();
	~vtkInverseDistanceFilter();

	//real deal function
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
protected:
	typedef std::vector<double> doubles;
	//power of weight
	double	m_PowerValue;

private:
	vtkInverseDistanceFilter(const vtkInverseDistanceFilter&);  // Not implemented.
	void operator=(const vtkInverseDistanceFilter&);  // Not implemented.

};

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)