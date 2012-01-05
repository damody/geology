// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
//  In academic purposes only(2012/1/12)

#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include <vector>
#include <cassert>
#include "vtkSmartPointer.h"
//Limited Inverse Distance Filter
//ignore terrain height
//only add call Cuda interpolation function in RequestData and use k-d tree
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

// author: t1238142000@gmail.com Liang-Shiuan Huang ���G�a
// author: a910000@gmail.com Kuang-Yi Chen ������
//  In academic purposes only(2012/1/12)