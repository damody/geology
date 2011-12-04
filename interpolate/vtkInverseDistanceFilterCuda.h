#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"

class vtkInverseDistanceFilterCuda : public vtkInterpolationGridingPolyDataFilter 
{
public:
	vtkTypeMacro(vtkInverseDistanceFilterCuda,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkInverseDistanceFilterCuda *New();
	void SetPowerValue(double v);
	double GetPowerValue();
protected:
	vtkInverseDistanceFilterCuda();
	~vtkInverseDistanceFilterCuda();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
	double	m_PowerValue;
private:
	vtkInverseDistanceFilterCuda(const vtkInverseDistanceFilterCuda&);  // Not implemented.
	void operator=(const vtkInverseDistanceFilterCuda&);  // Not implemented.

};
