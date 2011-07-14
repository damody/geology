#include "vtkLimitedInverseDistanceFilter.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkIdList.h"
#include <cassert>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <vtkMath.h>
template<class _Ty> struct power :
	public std::binary_function<_Ty, _Ty, _Ty>
{		// functor for operator%
	_Ty operator () (const _Ty &_Left, const _Ty &_Right) const
	{	// apply operator% to operands
		return pow(_Left, _Right);
	}
};
vtkStandardNewMacro(vtkLimitedInverseDistanceFilter);
vtkLimitedInverseDistanceFilter::vtkLimitedInverseDistanceFilter()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
	m_LimitNum = VTK_INT_MAX;
	m_Radius = VTK_FLOAT_MAX;
	m_limitMethod = LIMIT_NUMBER;
	m_PowerValue = 2.0;
	m_kDTree = vtkSmartPointer<vtkKdTreePointLocator>::New();
}

vtkLimitedInverseDistanceFilter::~vtkLimitedInverseDistanceFilter()
{ }
int vtkLimitedInverseDistanceFilter::RequestData
(
	vtkInformation		*vtkNotUsed(request),
	vtkInformationVector	**inputVector,
	vtkInformationVector	*outputVector
)
{

	// get the info objects
	vtkInformation	*inInfo = inputVector[0]->GetInformationObject(0);
	vtkInformation	*outInfo = outputVector->GetInformationObject(0);

	// get the input and ouptut
	vtkPolyData	*input = vtkPolyData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkPolyData	*output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

	// set kDtree and build
	m_kDTree->SetDataSet(input);
	m_kDTree->BuildLocator();

	// set safe number of points
	if (m_LimitNum > input->GetNumberOfPoints())
		m_LimitNum = input->GetNumberOfPoints();

	// new id list to get points
	VTK_CREATE(ids, vtkIdList);
	VTK_CREATE(outpoints, vtkPoints);
	VTK_CREATE(outScalars, vtkDoubleArray);
	vtkDoubleArray	*inScalars = (vtkDoubleArray *) (input->GetPointData()->GetScalars());
	double		*raw_points = (double *) malloc(sizeof(double) * 4 * input->GetNumberOfPoints()), *save_pos = raw_points;
	for (vtkIdType i = 0; i < input->GetNumberOfPoints(); i++, save_pos += 4)
	{
		input->GetPoint(i, save_pos);
		save_pos[3] = inScalars->GetValue(i);
	}
	save_pos = raw_points;
	doubles vals;
	double	dim[3];
	for (dim[2] = m_Bounds.zmin; dim[2] <= m_Bounds.zmax; dim[2] += m_Interval[2])
	{
		for (dim[1] = m_Bounds.ymin; dim[1] <= m_Bounds.ymax; dim[1] += m_Interval[1])
		{
			for (dim[0] = m_Bounds.xmin; dim[0] <= m_Bounds.xmax; dim[0] += m_Interval[0])
			{
				outpoints->InsertNextPoint(dim);
				doubles weights;
				int	zero_pos_index = -1;
				save_pos = raw_points;
				if (LIMIT_NUMBER == m_limitMethod)
					m_kDTree->FindClosestNPoints(m_LimitNum, dim, ids);
				else if (LIMIT_RADIUS == m_limitMethod)
					m_kDTree->FindPointsWithinRadius(m_Radius, dim, ids);
				const int MAX_POINTS = ids->GetNumberOfIds();
				double sum = 0, sum2 = 0, dis;
				for (int i = 0; i < ids->GetNumberOfIds(); i++)
				{
					const int index = ids->GetId(i);
					dis = sqrt(vtkMath::Distance2BetweenPoints(raw_points+index*4, dim));
					if (dis<0.001)
					{
						outScalars->InsertNextTuple1(*(raw_points+index*4+3));
						break;
					}
					double tmp = pow(dis, m_PowerValue);
					sum += tmp;
					sum2 += tmp*(*(raw_points+index*4+3)); // *(raw_points+i*4+3) is as same as inScalars->GetValue(i);
				}
				if (dis>=0.001)
				{
					sum2 /= sum;
					outScalars->InsertNextTuple1(sum2);
				}
			}
		}
	}
	// set out points
	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	free(raw_points);
	return 1;
}

//----------------------------------------------------------------------------
void vtkLimitedInverseDistanceFilter::PrintSelf(ostream &os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os, indent);
}

void vtkLimitedInverseDistanceFilter::SetLimitRadius(double r)
{
	assert(r > 0);
	m_Radius = r;
	m_limitMethod = LIMIT_RADIUS;
}

void vtkLimitedInverseDistanceFilter::SetNumOfLimitPoints(int n)
{
	assert(n > 0);
	m_LimitNum = n;
	m_limitMethod = LIMIT_NUMBER;
}

double vtkLimitedInverseDistanceFilter::GetLimitRadius()
{
	if (LIMIT_RADIUS == m_limitMethod)
		return m_Radius;
	else
		return -1;
}

int vtkLimitedInverseDistanceFilter::GetNumOfLimitPoints()
{
	if (LIMIT_NUMBER == m_limitMethod)
		return m_LimitNum;
	else
		return -1;
}

void vtkLimitedInverseDistanceFilter::SetPowerValue(double v)
{
	m_PowerValue = v;
}

double vtkLimitedInverseDistanceFilter::GetPowerValue()
{
	return m_PowerValue;
}
