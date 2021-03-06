// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
// In academic purposes only(2012/1/12)
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
// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
// In academic purposes only(2012/1/12)
