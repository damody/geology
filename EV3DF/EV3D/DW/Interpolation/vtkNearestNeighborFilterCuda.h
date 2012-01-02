// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
//v
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
// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
