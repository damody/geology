// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include "vtkKrigingFilter.h"

//Kriging Filter for Cuda
//ignore terrain height
//only add call Cuda interpolation function in RequestData
class vtkKrigingFilterCuda : public vtkKrigingFilter 
{
public:
	vtkTypeMacro(vtkKrigingFilterCuda,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkKrigingFilterCuda *New();
	void SetPowerValue(double v);
	double GetPowerValue();
protected:
	vtkKrigingFilterCuda();
	~vtkKrigingFilterCuda();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
	double	m_PowerValue;
};

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)