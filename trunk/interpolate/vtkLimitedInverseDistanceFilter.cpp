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
#include <vtkKdTree.h>


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
	m_LimitNum = 20;
	m_Radius = VTK_FLOAT_MAX;
	m_limitMethod = LIMIT_NUMBER;
	m_PowerValue = 2.0;
}

vtkLimitedInverseDistanceFilter::~vtkLimitedInverseDistanceFilter()
{ }
int vtkLimitedInverseDistanceFilter::RequestData( vtkInformation *vtkNotUsed(request),
						 vtkInformationVector	**inputVector,
						 vtkInformationVector	*outputVector )
{
	// get the info objects
	vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	// get the input and ouptut
	vtkPolyData *input = vtkPolyData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));
	
	// new id list to get points
	VTK_CREATE(ids, vtkIdList);
	VTK_CREATE(kDTree, vtkKdTreePointLocator);
	VTK_CREATE(outpoints, vtkPoints);
	VTK_CREATE(outScalars, vtkDoubleArray);
	vtkDoubleArray* inScalars = (vtkDoubleArray*)(input->GetPointData()->GetScalars());
	double *raw_points = (double*)malloc(sizeof(double) * 4 * input->GetNumberOfPoints()),
		*save_pos = raw_points;
	point* pts = (point*)malloc(sizeof(point) * input->GetNumberOfPoints());
	// set kDtree and build
	kDTree->SetDataSet(input);
	kDTree->BuildLocator();

	// set safe number of points
	if (m_LimitNum > input->GetNumberOfPoints())
		m_LimitNum = input->GetNumberOfPoints();

	for (vtkIdType i = 0; i < input->GetNumberOfPoints(); i++, save_pos += 4)
	{
		input->GetPoint(i, save_pos);
		save_pos[3] = inScalars->GetValue(i);
		pts[i].x = save_pos[0];
		pts[i].y = save_pos[1];
		pts[i].z = save_pos[2];
	}
	m_kd_tree.BuildKdtree(pts, input->GetNumberOfPoints());
	save_pos = raw_points;
	doubles vals;
	double	dim[3];
	{ // fix Numerical error
		m_Bounds.xmax += m_Interval[0]*0.5;
		m_Bounds.ymax += m_Interval[1]*0.5;
		m_Bounds.zmax += m_Interval[2]*0.5;
	}
	float sqr2 = m_Radius*m_Radius;
	struct myvalue
	{
		myvalue(){}
		myvalue(double d, int i)
			:distance(d), index(i)
		{
		}
		double distance;
		int index;
		bool operator < (const myvalue& rhs)
		{
			return distance < rhs.distance;
		}
	};
	std::vector<myvalue> myvalues;
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
				{
					kDTree->FindClosestNPoints(m_LimitNum, dim, ids);
					const int MAX_POINTS = ids->GetNumberOfIds();
					double sum = 0, sum2 = 0, dis;
					if (ids->GetNumberOfIds() > 0)
					{
						myvalues.clear();
						int max_ids = ids->GetNumberOfIds();
						for (int i = 0; i < max_ids; i++)
						{
							const int index = ids->GetId(i);
							dis = vtkMath::Distance2BetweenPoints(raw_points+index*4, dim);
							dis = sqrt(dis);
							myvalue mv;
							mv.distance = dis;
							mv.index = index;
							myvalues.push_back(mv);
							if (dis<0.001)
							{
								outScalars->InsertNextTuple1(*(raw_points+index*4+3));
								break;
							}
						}
						std::nth_element(myvalues.begin(), myvalues.begin()+m_LimitNum, myvalues.end());
						for (int i = 0;i < m_LimitNum;i++)
						{
							double tmp = pow(myvalues[i].distance, -m_PowerValue);
							sum += tmp;
							sum2 += tmp*(*(raw_points+myvalues[i].index*4+3)); // *(raw_points+i*4+3) is as same as inScalars->GetValue(i);
						}
						if (dis>=0.001)
						{
							sum2 /= sum;
							outScalars->InsertNextTuple1(sum2);
						}
					}
					else
					{
						outScalars->InsertNextTuple1(m_NullValue);
					}
				}
				else if (LIMIT_RADIUS == m_limitMethod)
				{
					/*
					int buffer[1000];
					float bound[6];
					bound[0] = dim[0]-m_Radius;
					bound[1] = dim[0]+m_Radius;
					bound[2] = dim[1]-m_Radius;
					bound[3] = dim[1]+m_Radius;
					bound[4] = dim[2]-m_Radius;
					bound[5] = dim[2]+m_Radius;
					m_kd_tree.FSearchKdtreeByBounding(pts, bound, buffer, 1000);
					double sum = 0, sum2 = 0, dis=999;
					if (buffer[0] != -1)
					{
						int count = 0;
						for (int i=0;i<1000;++i,++count)
						{
							const int index = buffer[i];
							if (buffer[i] == -1)
								break;
							dis = vtkMath::Distance2BetweenPoints(raw_points+index*4, dim);
							dis = sqrt(dis);
							if (dis<0.001)
							{
								outScalars->InsertNextTuple1(*(raw_points+index*4+3));
								break;
							}
							double tmp = pow(dis, -m_PowerValue);
							sum += tmp;
							sum2 += tmp*(*(raw_points+index*4+3));
						}
						//printf("%d\n", count);
						if (dis>=0.001)
						{
							sum2 /= sum;
							outScalars->InsertNextTuple1(sum2);
						}
					}
					else
						outScalars->InsertNextTuple1(m_NullValue);
					*/
					double sum = 0, sum2 = 0, dis=999;
					kDTree->FindPointsWithinRadius(m_Radius, dim, ids);
					if (ids->GetNumberOfIds() > 0)
					{
						int max_ids = ids->GetNumberOfIds();
						//if (LIMIT_NUMBER == m_limitMethod) max_ids = m_LimitNum;
						for (int i = 0; i < max_ids; i++)
						{
							const int index = ids->GetId(i);
							dis = vtkMath::Distance2BetweenPoints(raw_points+index*4, dim);
							dis = sqrt(dis);
							if (dis<0.001)
							{
								outScalars->InsertNextTuple1(*(raw_points+index*4+3));
								break;
							}
							double tmp = pow(dis, -m_PowerValue);
							sum += tmp;
							sum2 += tmp*(*(raw_points+index*4+3)); // *(raw_points+i*4+3) is as same as inScalars->GetValue(i);
						}
						if (dis>=0.001)
						{
							sum2 /= sum;
							outScalars->InsertNextTuple1(sum2);
						}
					}
					else
					{
						outScalars->InsertNextTuple1(m_NullValue);
					}
				}
			}
		}
	}
	{ // fix Numerical error
		m_Bounds.xmax -= m_Interval[0]*0.5;
		m_Bounds.ymax -= m_Interval[1]*0.5;
		m_Bounds.zmax -= m_Interval[2]*0.5;
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
