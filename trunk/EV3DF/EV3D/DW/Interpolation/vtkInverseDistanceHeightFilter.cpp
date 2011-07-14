#include "vtkInverseDistanceHeightFilter.h"

#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"
#include <cassert>
#include <numeric>
#include <algorithm>

vtkStandardNewMacro(vtkInverseDistanceHeightFilter);
template<class _Ty>
struct power
	: public std::binary_function<_Ty, _Ty, _Ty>
{	// functor for operator%
	_Ty operator()(const _Ty& _Left, const _Ty& _Right) const
	{	// apply operator% to operands
		return pow(_Left, _Right);
	}
};

int vtkInverseDistanceHeightFilter::RequestData(vtkInformation *vtkNotUsed(request),
												vtkInformationVector **inputVector,
												vtkInformationVector *outputVector)
{
	// get the info objects
	vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
	vtkInformation *inInfoForH = inputVector[1]->GetInformationObject(0);
	vtkInformation *outInfo = outputVector->GetInformationObject(0);
	// get the input and ouptut
	vtkPolyData *input = vtkPolyData::SafeDownCast(
		inInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkPolyData *inputForH = vtkPolyData::SafeDownCast(
		inInfoForH->Get(vtkDataObject::DATA_OBJECT()));
	vtkPolyData *output = vtkPolyData::SafeDownCast(
		outInfo->Get(vtkDataObject::DATA_OBJECT()));
	
	vtkPoints * outpoints = vtkPoints::New();
	vtkDoubleArray* outScalars = vtkDoubleArray::New();
	vtkDoubleArray* inScalars = (vtkDoubleArray*)(input->GetPointData()->GetScalars());
	double *raw_points = (double*)malloc(sizeof(double) * 4 * input->GetNumberOfPoints()),
		*save_pos = raw_points;
	for(vtkIdType i = 0; i < input->GetNumberOfPoints(); i++, save_pos += 4)
	{
		input->GetPoint(i, save_pos);
		save_pos[3] = inScalars->GetValue(i);
#ifdef _DEBUG
		std::cout << "Point " << i << " : (" << save_pos[0] << " " << save_pos[1] << " " << save_pos[2] << ")" << std::endl;
#endif // _DEBUG
	}
	save_pos = raw_points;
	doubles vals;
	double dim[3], newpos[3];
	doubles heighttable;
	doubles weights;
	int hindex;
	int nx = (m_Bounds.xmax-m_Bounds.xmin)/m_Interval[0]+1;
	int ny = (m_Bounds.ymax-m_Bounds.ymin)/m_Interval[1]+1;
	int nz = (m_Bounds.zmax-m_Bounds.zmin)/m_Interval[2]+1;
	int ox=0, oy=0, oz=0;
	doubles tres;
	tres.resize(nx*ny*nz);

	for(vtkIdType i = 0; i < input->GetNumberOfPoints(); i++, save_pos += 4)
	{
		vals.push_back(inScalars->GetValue(i));
	}
	heighttable.clear();
	for (dim[0] = m_Bounds.xmin;dim[0] <= m_Bounds.xmax;dim[0]+=m_Interval[0])
	{
		for (dim[2] = m_Bounds.zmin;dim[2] <= m_Bounds.zmax;dim[2]+=m_Interval[2])
		{
			heighttable.push_back(GetHeight(inputForH, dim[0], dim[2]));
		}
	}

	for (dim[1] = m_Bounds.ymin;dim[1] <= m_Bounds.ymax;dim[1]+=m_Interval[1], oy++)
	{
		BuildPlaneSample(input, inScalars, dim[1]);
		hindex = 0;
		ox=0;
		for (dim[0] = m_Bounds.xmin;dim[0] <= m_Bounds.xmax;dim[0]+=m_Interval[0], ox++)
		{
			oz=0;
			for (dim[2] = m_Bounds.zmin;dim[2] <= m_Bounds.zmax;dim[2]+=m_Interval[2], hindex++, oz++)
			{
				if (m_IgrSeaLinePoint)
					if (GetError(heighttable[hindex], m_SeaLineHeight, 0, 0) < 1 || heighttable[hindex] < dim[1])	//the point's height is close sea line
					{
						tres[ox*ny*nz+oy*nz+oz] = m_NullValue;
						continue;
					}

				int zero_pos_index = -1;
				weights.clear();
				vals.clear();

				save_pos = raw_points;
				for(vtkIdType i = 0; i < (int)v_PlaneSampleValue.size(); i++, save_pos += 8)
				{
					if (v_PlaneSampleValue[i] <= 0)
						continue;
					newpos[0] = save_pos[0];
					newpos[1] = dim[1];
					newpos[2] = save_pos[2];
					double dis = sqrt(vtkMath::Distance2BetweenPoints(newpos, dim));
					assert(dis>=0);
					if (dis == 0.0)
					{
						zero_pos_index = i;
						break;
					}
					weights.push_back(dis);
					vals.push_back(v_PlaneSampleValue[i]);
				}
				if (zero_pos_index != -1)
				{
					tres[ox*ny*nz+oy*nz+oz] = m_NullValue;
				}
				else
				{
					std::transform(weights.begin(), weights.end(), weights.begin(),
						std::bind2nd(power<double>(), -2*m_PowerValue));
					double sum = accumulate(weights.begin(), weights.end(), 0.0, std::plus<double>());
					double multi_num;
					if (sum > 0)
					{
						multi_num = 1.0/sum;
						// each weight multiplies multi_num, let sum = 1
						std::transform(weights.begin(), weights.end(), weights.begin(),
							std::bind2nd(std::multiplies<double>(), multi_num));
						// get average use (weight1 * value1) + (weight2 * value2) + ...
						double val = std::inner_product(weights.begin(), weights.end(),
							vals.begin(), 0.0, std::plus<double>(), std::multiplies<double>());
						tres[ox*ny*nz+oy*nz+oz] = val;
					}
					else
					{
						tres[ox*ny*nz+oy*nz+oz] = m_NullValue;
					}
				}

			}
		}
	}

	ox=oy=oz=0;
	for (dim[2] = m_Bounds.zmin;dim[2] <= m_Bounds.zmax;dim[2]+=m_Interval[2],oz++)
	{
		oy = 0;
		for (dim[1] = m_Bounds.ymin;dim[1] <= m_Bounds.ymax;dim[1]+=m_Interval[1],oy++)
		{
			ox = 0;
			for (dim[0] = m_Bounds.xmin;dim[0] <= m_Bounds.xmax;dim[0]+=m_Interval[0],ox++)
			{
				outpoints->InsertNextPoint(dim);
				outScalars->InsertNextTuple1(tres[ox*ny*nz+oy*nz+oz]);
			}
		}
	}
	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	return 1;
}

vtkInverseDistanceHeightFilter::vtkInverseDistanceHeightFilter()
{
	this->SetNumberOfInputPorts(2);
	this->SetNumberOfOutputPorts(1);
	m_NullValue = VTK_FLOAT_MIN;
	m_PowerValue = 2;
	m_SeaLineHeight = 0;
	m_IgrSeaLinePoint = true;
	m_IgrNullValuePoint = false;
}
