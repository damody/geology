// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
//  In academic purposes only(2012/1/12)

#pragma once
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkPolyDataAlgorithm.h"
#include "vtkBounds.h"
#include <vtkMath.h>
#define VTK_CREATE(var, type) vtkSmartPointer<type> var = vtkSmartPointer<type>::New();

//base class for VTK interpolation griding filter
class vtkInterpolationGridingPolyDataFilter : public vtkPolyDataAlgorithm 
{
public:
	vtkTypeMacro(vtkInterpolationGridingPolyDataFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkInterpolationGridingPolyDataFilter *New();
	// set basic information
	void SetBounds(const double bounds[]);
	void GetBounds(double bounds[]);
	void SetInterval(double x, double y, double z);
	void SetInterval(double inter[]);
	void SetCudaThreadNum(int num){m_CudaThreadNum = num;}
	void SetUseSharedMem(bool setting){m_UseSharedMem = setting;}

	int  NumOfInterpolationPoints();	//get the number of interpolation point
	int  NumOfXPoints();				//get number of points at x-axle
	int  NumOfYPoints();				//get number of points at y-axle
	int  NumOfZPoints();				//get number of points at z-axle

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
	float	m_NullValue;		// the number for null value
	vtkBounds m_Bounds;			// vtk's format bounds
	double m_Interval[3];		// griding's interval
	int		m_CudaThreadNum;	// the number of threads
	bool	m_UseSharedMem;		// use shared memory
	vtkInterpolationGridingPolyDataFilter();
	~vtkInterpolationGridingPolyDataFilter();

	//real deal function
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);	//compute function

private:
	vtkInterpolationGridingPolyDataFilter(const vtkInterpolationGridingPolyDataFilter&);  // Not implemented.
	void operator=(const vtkInterpolationGridingPolyDataFilter&);  // Not implemented.

};

// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
//  In academic purposes only(2012/1/12)