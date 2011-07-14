#pragma once
#include "vtknearestneighborfilter.h"
#include "vtkGridHeightFilter.h"

class vtkNearestNeighborHeightFilter :
	public vtkNearestNeighborFilter, public vtkGridHeightFilter
{
public:
	vtkTypeMacro(vtkNearestNeighborFilter,vtkPolyDataAlgorithm);
	static vtkNearestNeighborHeightFilter *New();

protected:
	vtkNearestNeighborHeightFilter(void);
	~vtkNearestNeighborHeightFilter(void);

	typedef std::vector<double> doubles;
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
};
