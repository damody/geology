#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"
#include <vector>

class vtkInverseDistanceFilter : public vtkInterpolationGridingPolyDataFilter 
{
public:
	vtkTypeMacro(vtkInverseDistanceFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkInverseDistanceFilter *New();

protected:
	typedef std::vector<double> doubles;

	vtkInverseDistanceFilter();
	~vtkInverseDistanceFilter();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
	vtkInverseDistanceFilter(const vtkInverseDistanceFilter&);  // Not implemented.
	void operator=(const vtkInverseDistanceFilter&);  // Not implemented.

};
