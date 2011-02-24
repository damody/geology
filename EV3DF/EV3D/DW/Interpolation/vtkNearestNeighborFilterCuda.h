#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"

class vtkNearestNeighborFilterCuda : public vtkInterpolationGridingPolyDataFilter 
{
public:
	vtkTypeMacro(vtkNearestNeighborFilterCuda,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkNearestNeighborFilterCuda *New();

protected:
	vtkNearestNeighborFilterCuda();
	~vtkNearestNeighborFilterCuda();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
	vtkNearestNeighborFilterCuda(const vtkNearestNeighborFilterCuda&);  // Not implemented.
	void operator=(const vtkNearestNeighborFilterCuda&);  // Not implemented.

};
