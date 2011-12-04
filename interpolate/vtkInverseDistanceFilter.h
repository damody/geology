// athour: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a, ³¯¥ú«³
// In academic purposes only
#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include <vector>
//v
class vtkInverseDistanceFilter : public vtkInterpolationGridingPolyDataFilter 
{
public:
	vtkTypeMacro(vtkInverseDistanceFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkInverseDistanceFilter *New();
	void SetPowerValue(double v);
	double GetPowerValue();
protected:
	vtkInverseDistanceFilter();
	~vtkInverseDistanceFilter();
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
protected:
	typedef std::vector<double> doubles;
	double	m_PowerValue;
private:
	vtkInverseDistanceFilter(const vtkInverseDistanceFilter&);  // Not implemented.
	void operator=(const vtkInverseDistanceFilter&);  // Not implemented.

};
// athour: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a, ³¯¥ú«³
// In academic purposes only