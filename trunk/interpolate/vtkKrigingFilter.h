#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"

class vtkKrigingFilter : public vtkInterpolationGridingPolyDataFilter 
{
public:
	vtkTypeMacro(vtkKrigingFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkKrigingFilter *New();

	void SetDistStep(double step){m_DistStep = step;}
	void SetStepAutomatic(bool setting){m_AutoGetStep = setting;}
	double GetAutoDistStep();

protected:
	vtkKrigingFilter();
	~vtkKrigingFilter();
	double m_DistStep;
	bool m_AutoGetStep;

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
	vtkKrigingFilter(const vtkKrigingFilter&);  // Not implemented.
	void operator=(const vtkKrigingFilter&);  // Not implemented.

};
