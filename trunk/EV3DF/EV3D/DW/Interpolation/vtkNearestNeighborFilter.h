// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
//v
class vtkNearestNeighborFilter : public vtkInterpolationGridingPolyDataFilter 
{
public:
	vtkTypeMacro(vtkNearestNeighborFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkNearestNeighborFilter *New();

protected:
	vtkNearestNeighborFilter();
	~vtkNearestNeighborFilter();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
	vtkNearestNeighborFilter(const vtkNearestNeighborFilter&);  // Not implemented.
	void operator=(const vtkNearestNeighborFilter&);  // Not implemented.

};
// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
