// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)

#pragma once
#include "vtkInterpolationGridingPolyDataFilter.h"

//Kriging Filter
//ignore terrain height
class vtkKrigingFilter : public vtkInterpolationGridingPolyDataFilter 
{
public:
	vtkTypeMacro(vtkKrigingFilter,vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent);
	static vtkKrigingFilter *New();

	//set basic parameter
	void SetDistStep(double step){m_DistStep = step;}
	void SetStepAutomatic(bool setting){m_AutoGetStep = setting;}

protected:
	vtkKrigingFilter();
	~vtkKrigingFilter();

	//! The variogram distance step
	double m_DistStep;
	//!  The setting for whether use getting distance step automatically or not
	bool m_AutoGetStep;

	//real deal function
	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
	vtkKrigingFilter(const vtkKrigingFilter&);  // Not implemented.
	void operator=(const vtkKrigingFilter&);  // Not implemented.

};

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
//  In academic purposes only(2012/1/12)