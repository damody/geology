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

vtkStandardNewMacro(vtkInverseDistanceFilter);

vtkInverseDistanceFilter::vtkInverseDistanceFilter()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
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
	vtkPolyData *input = vtkPolyData::SafeDownCast(
		inInfo->Get(vtkDataObject::DATA_OBJECT()));
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
	if (input->GetNumberOfPoints()<2) //頂點數小於2直接離開
	{
		output->ShallowCopy(input);
		return 1;
	}
	save_pos = raw_points;
	doubles vals;
	for(vtkIdType i = 0; i < input->GetNumberOfPoints(); i++, save_pos += 4)
	{
		vals.push_back(inScalars->GetValue(i));
	}
	double dim[3];
	for (dim[2] = m_bounds.zmin;dim[2] <= m_bounds.zmax;dim[2]+=m_interval)
	{
		for (dim[1] = m_bounds.ymin;dim[1] <= m_bounds.ymax;dim[1]+=m_interval)
		{
			for (dim[0] = m_bounds.xmin;dim[0] <= m_bounds.xmax;dim[0]+=m_interval)
			{
				
				doubles weights;
				int zero_pos_index = -1;
				save_pos = raw_points;
				for(vtkIdType i = 0; i < input->GetNumberOfPoints(); i++, save_pos += 4)
				{
					double dis = PointsDistanceSquare(save_pos, dim);
					assert(dis>=0);
					if (dis == 0.0)
					{
						zero_pos_index = i;
						break;
					}
					weights.push_back(dis);
				}
				if (zero_pos_index != -1)
				{
					outpoints->InsertNextPoint(dim);
					outScalars->InsertNextTuple1(inScalars->GetValue(zero_pos_index));
				}
				else
				{
					double sum = accumulate(weights.begin(), weights.end(), 0.0, std::plus<double>());
					assert(sum>0);
					double multi_num = 1.0/sum;
					// each weight multiplies multi_num, let sum = 1
					std::transform(weights.begin(), weights.end(), weights.begin(),
						std::bind2nd(std::multiplies<double>(), multi_num));
					// get average use (weight1 * value1) + (weight2 * value2) + ...
					double val = std::inner_product(weights.begin(), weights.end(),
						vals.begin(), 0.0, std::plus<double>(), std::multiplies<double>());
					outpoints->InsertNextPoint(dim);
					outScalars->InsertNextTuple1(val);
				}
			}
		}
	}
	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	return 1;
}


//----------------------------------------------------------------------------
void vtkInverseDistanceFilter::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}

