// athour: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒, 陳光奕
// In academic purposes only
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
	void SetCudaThreadNum(int num){m_CudaThreadNum = num;}
	void SetUseSharedMem(bool setting){m_UseSharedMem = setting;}
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
	float	m_NullValue;
	vtkBounds m_Bounds;	// vtk's format bounds
	double m_Interval[3];	// griding's interval
	int		m_CudaThreadNum;	//the number of threads
	bool	m_UseSharedMem;		//use shared memory
	vtkInterpolationGridingPolyDataFilter();
	~vtkInterpolationGridingPolyDataFilter();

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
	vtkInterpolationGridingPolyDataFilter(const vtkInterpolationGridingPolyDataFilter&);  // Not implemented.
	void operator=(const vtkInterpolationGridingPolyDataFilter&);  // Not implemented.

};
// athour: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒, 陳光奕
// In academic purposes only