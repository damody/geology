#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include <vector>
#include <cassert>
#include "vtkSmartPointer.h"
//v
class vtkLimitedInverseDistanceFilterCuda : public vtkInterpolationGridingPolyDataFilter
{
public:
	vtkTypeMacro(vtkLimitedInverseDistanceFilterCuda,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkLimitedInverseDistanceFilterCuda *New();
	void SetLimitRadius(double r);
	void SetNumOfLimitPoints(int n);
	double GetLimitRadius();
	int GetNumOfLimitPoints();
	void SetPowerValue(double v);
	double GetPowerValue();
protected:
	vtkLimitedInverseDistanceFilterCuda();
	~vtkLimitedInverseDistanceFilterCuda();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
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
	vtkLimitedInverseDistanceFilterCuda(const vtkLimitedInverseDistanceFilterCuda&);  // Not implemented.
	void operator=(const vtkLimitedInverseDistanceFilterCuda&);  // Not implemented.


};