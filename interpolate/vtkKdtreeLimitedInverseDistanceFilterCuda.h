#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include <vector>
#include <cassert>
#include "vtkSmartPointer.h"
//v
class vtkKdtreeLimitedInverseDistanceFilterCuda : public vtkInterpolationGridingPolyDataFilter
{
public:
	vtkTypeMacro(vtkKdtreeLimitedInverseDistanceFilterCuda,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkKdtreeLimitedInverseDistanceFilterCuda *New();
	void SetLimitRadius(double r);
	void SetNumOfLimitPoints(int n);
	double GetLimitRadius();
	int GetNumOfLimitPoints();
	void SetPowerValue(double v);
	double GetPowerValue();
protected:
	vtkKdtreeLimitedInverseDistanceFilterCuda();
	~vtkKdtreeLimitedInverseDistanceFilterCuda();

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
	vtkKdtreeLimitedInverseDistanceFilterCuda(const vtkKdtreeLimitedInverseDistanceFilterCuda&);  // Not implemented.
	void operator=(const vtkKdtreeLimitedInverseDistanceFilterCuda&);  // Not implemented.


};
