#pragma once
#include "vtkLimitedInverseDistanceFilter.h"
#include "vtkGridHeightFilter.h"
class vtkLimitedInverseDistanceHeightFilter :
	public vtkLimitedInverseDistanceFilter,
	public vtkGridHeightFilter
{
public:
	vtkTypeMacro(vtkLimitedInverseDistanceHeightFilter, vtkPolyDataAlgorithm);
	static vtkLimitedInverseDistanceHeightFilter	*New();
protected:
	vtkLimitedInverseDistanceHeightFilter(void);
	int	RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
};
