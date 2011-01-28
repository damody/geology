#pragma once
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkPolyDataAlgorithm.h"
#include "vtkBounds.h"


class vtkInterpolationGridingPolyDataFilter : public vtkPolyDataAlgorithm 
{
public:
	vtkTypeMacro(vtkInterpolationGridingPolyDataFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkInterpolationGridingPolyDataFilter *New();
	// Customer function
	void SetBounds(const double bounds[]);
	void GetBounds(double bounds[]);
	void SetInterval(double inter);
	int  NumOfInterpolationPoints();
	int  NumOfXPoints();
	int  NumOfYPoints();
	int  NumOfZPoints();
protected:
	vtkBounds m_bounds;	// vtk's format bounds
	double m_interval;	// griding's interval
	double PointsDistanceSquare(double pos1[], double pos2[]);
	vtkInterpolationGridingPolyDataFilter();
	~vtkInterpolationGridingPolyDataFilter();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
	vtkInterpolationGridingPolyDataFilter(const vtkInterpolationGridingPolyDataFilter&);  // Not implemented.
	void operator=(const vtkInterpolationGridingPolyDataFilter&);  // Not implemented.

};
