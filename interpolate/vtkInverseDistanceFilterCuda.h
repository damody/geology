// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"

//Inverse Distance Filter
//ignore terrain height
//only add call Cuda interpolation function in RequestData
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

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)