// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)

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
// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
