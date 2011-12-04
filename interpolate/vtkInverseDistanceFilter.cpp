#include "vtkInverseDistanceFilter.h"

#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"
#include <cassert>
#include <numeric>
#include <algorithm>
#include <vtkMath.h>

vtkStandardNewMacro(vtkInverseDistanceFilter);

template<class _Ty>
struct power
	: public std::binary_function<_Ty, _Ty, _Ty>
{	// functor for operator%
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const
	{	// apply operator% to operands
		return pow(_Left, _Right);
	}
};


vtkInverseDistanceFilter::vtkInverseDistanceFilter()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
	m_PowerValue = 2.0;
}

vtkInverseDistanceFilter::~vtkInverseDistanceFilter()
{

}

int vtkInverseDistanceFilter::RequestData(vtkInformation *vtkNotUsed(request),
			       vtkInformationVector **inputVector,
			       vtkInformationVector *outputVector)
{
	// get the info objects
	vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	// get the input and ouptut
	vtkPolyData *input = vtkPolyData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

	VTK_CREATE(outpoints, vtkPoints);
	VTK_CREATE(outScalars, vtkDoubleArray);
	vtkDoubleArray* inScalars = (vtkDoubleArray*)(input->GetPointData()->GetScalars());
	const int MAX_POINTS = input->GetNumberOfPoints();
	double *raw_points = (double*)malloc(sizeof(double) * 4 * input->GetNumberOfPoints()),
		*save_pos = raw_points;
	
	for(vtkIdType i = 0; i < input->GetNumberOfPoints(); i++, save_pos += 4)
	{
		input->GetPoint(i, save_pos);
		save_pos[3] = inScalars->GetValue(i);
	}
	doubles vals;
	double dim[3];
	{ // fix Numerical error
		m_Bounds.xmax += m_Interval[0]*0.5;
		m_Bounds.ymax += m_Interval[1]*0.5;
		m_Bounds.zmax += m_Interval[2]*0.5;
	}
	for (dim[2] = m_Bounds.zmin;dim[2] <= m_Bounds.zmax;dim[2]+=m_Interval[2])
	{
		for (dim[1] = m_Bounds.ymin;dim[1] <= m_Bounds.ymax;dim[1]+=m_Interval[1])
		{
			for (dim[0] = m_Bounds.xmin;dim[0] <= m_Bounds.xmax;dim[0]+=m_Interval[0])
			{
				outpoints->InsertNextPoint(dim);
				double sum = 0, sum2 = 0, dis;
				for (int i = 0;i < MAX_POINTS;i++)
				{
					dis = sqrt(vtkMath::Distance2BetweenPoints(raw_points+i*4, dim));
					if (dis<0.001)
					{
						outScalars->InsertNextTuple1(*(raw_points+i*4+3));
						break;
					}
					double tmp = pow(dis, -m_PowerValue);
					sum += tmp;
					sum2 += tmp*(*(raw_points+i*4+3)); // *(raw_points+i*4+3) is as same as inScalars->GetValue(i);
				}
				if (dis>=0.001)
				{
					sum2 /= sum;
					outScalars->InsertNextTuple1(sum2);
				}
			}
		}
	}
	{ // fix Numerical error
		m_Bounds.xmax -= m_Interval[0]*0.5;
		m_Bounds.ymax -= m_Interval[1]*0.5;
		m_Bounds.zmax -= m_Interval[2]*0.5;
	}
	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	free(raw_points);
	return 1;
}


//----------------------------------------------------------------------------
void vtkInverseDistanceFilter::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}

void vtkInverseDistanceFilter::SetPowerValue( double v )
{
	m_PowerValue = v;
}

double vtkInverseDistanceFilter::GetPowerValue()
{
	return m_PowerValue;
}

