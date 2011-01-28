#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"

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
