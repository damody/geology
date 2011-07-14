#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include <vector>
#include <cassert>
#include "vtkKdTreePointLocator.h"
#include "vtkSmartPointer.h"
//v
class vtkLimitedInverseDistanceFilter : public vtkInterpolationGridingPolyDataFilter
{
public:
	vtkTypeMacro(vtkLimitedInverseDistanceFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkLimitedInverseDistanceFilter *New();
	void SetLimitRadius(double r);
	void SetNumOfLimitPoints(int n);
	double GetLimitRadius();
	int GetNumOfLimitPoints();
	void SetPowerValue(double v);
	double GetPowerValue();
protected:
	vtkLimitedInverseDistanceFilter();
	~vtkLimitedInverseDistanceFilter();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
	vtkSmartPointer<vtkKdTreePointLocator> m_kDTree;
protected:
	typedef std::vector<double> doubles;
	double	m_Radius;
	double	m_PowerValue;
	int	m_LimitNum;
	enum 
	{
		LIMIT_RADIUS,
		LIMIT_NUMBER
	};
	int	m_limitMethod;
private:
	vtkLimitedInverseDistanceFilter(const vtkLimitedInverseDistanceFilter&);  // Not implemented.
	void operator=(const vtkLimitedInverseDistanceFilter&);  // Not implemented.

	
};
