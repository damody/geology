#pragma once
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkPolyDataAlgorithm.h"
#include "vtkBounds.h"
#include <vtkMath.h>
#define VTK_CREATE(var, type) vtkSmartPointer<type> var = vtkSmartPointer<type>::New();

class vtkInterpolationGridingPolyDataFilter : public vtkPolyDataAlgorithm 
{
public:
	vtkTypeMacro(vtkInterpolationGridingPolyDataFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkInterpolationGridingPolyDataFilter *New();
	// Customer function
	void SetBounds(const double bounds[]);
	void GetBounds(double bounds[]);
	void SetInterval(double x, double y, double z);
	void SetInterval(double inter[]);
	int  NumOfInterpolationPoints();
	int  NumOfXPoints();
	int  NumOfYPoints();
	int  NumOfZPoints();
	// if don't have any value set this value, default is VTK_FLOAT_MIN
	void SetNullValue( double v )
	{
		m_NullValue = v;
	}
	double GetNullValue()
	{
		return m_NullValue;
	}
protected:
	double	m_NullValue;
	vtkBounds m_Bounds;	// vtk's format bounds
	double m_Interval[3];	// griding's interval
	vtkInterpolationGridingPolyDataFilter();
	~vtkInterpolationGridingPolyDataFilter();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
	vtkInterpolationGridingPolyDataFilter(const vtkInterpolationGridingPolyDataFilter&);  // Not implemented.
	void operator=(const vtkInterpolationGridingPolyDataFilter&);  // Not implemented.

};
