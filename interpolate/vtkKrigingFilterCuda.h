#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include "vtkKrigingFilter.h"

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
