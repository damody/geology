// athour: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a, ³¯¥ú«³
// In academic purposes only
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
// athour: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a, ³¯¥ú«³
// In academic purposes only