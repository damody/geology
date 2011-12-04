#pragma once
#include "vtkInverseDistanceFilter.h"
#include "vtkGridHeightFilter.h"

class vtkInverseDistanceHeightFilter :
	public vtkInverseDistanceFilter, public vtkGridHeightFilter
{
public:
	vtkTypeMacro(vtkInverseDistanceHeightFilter,vtkPolyDataAlgorithm);
	static vtkInverseDistanceHeightFilter *New();

protected:
	vtkInverseDistanceHeightFilter();
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
};
