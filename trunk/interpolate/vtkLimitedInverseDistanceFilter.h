// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include <vector>
#include <cassert>
#include "vtkKdTreePointLocator.h"
#include "vtkSmartPointer.h"
#include "kd_tree.h"
//Limited Inverse Distance Filter
//ignore terrain height
class vtkLimitedInverseDistanceFilter : public vtkInterpolationGridingPolyDataFilter
{
public:
	vtkTypeMacro(vtkLimitedInverseDistanceFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkLimitedInverseDistanceFilter *New();

	//set basic parameter
	void SetLimitRadius(double r);
	void SetNumOfLimitPoints(int n);
	void SetPowerValue(double v);

	//get basic parameter
	double GetLimitRadius();
	int GetNumOfLimitPoints();
	double GetPowerValue();
protected:
	vtkLimitedInverseDistanceFilter();
	~vtkLimitedInverseDistanceFilter();

	//real deal function
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
	vtkSmartPointer<vtkIdList>	m_ids;
	vtkSmartPointer<vtkPoints>	m_outpoints;
	vtkSmartPointer<vtkDoubleArray>	m_outScalars;
	kd_tree		m_kd_tree;
protected:
	typedef std::vector<double> doubles;
	double	m_Radius;			//k-d tree find rang radius
	double	m_PowerValue;		//power of weight
	int	m_LimitNum;				//k-d tree limit number of capture
	enum 
	{
		LIMIT_RADIUS,
		LIMIT_NUMBER
	};
	int	m_limitMethod;			//use which kink of k-d tree method
private:
	vtkLimitedInverseDistanceFilter(const vtkLimitedInverseDistanceFilter&);  // Not implemented.
	void operator=(const vtkLimitedInverseDistanceFilter&);  // Not implemented.


};

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)