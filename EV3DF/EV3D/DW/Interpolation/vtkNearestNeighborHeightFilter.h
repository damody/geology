// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
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
// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
