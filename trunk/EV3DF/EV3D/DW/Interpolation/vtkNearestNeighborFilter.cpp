#include "vtkNearestNeighborFilter.h"

#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"

vtkStandardNewMacro(vtkNearestNeighborFilter);

vtkNearestNeighborFilter::vtkNearestNeighborFilter()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
}

vtkNearestNeighborFilter::~vtkNearestNeighborFilter()
{

}

int vtkNearestNeighborFilter::RequestData(vtkInformation *vtkNotUsed(request),
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
	VTK_CREATE(outpoints, vtkPoints);
	VTK_CREATE(outScalars, vtkDoubleArray);
	vtkDoubleArray* inScalars = (vtkDoubleArray*)(input->GetPointData()->GetScalars());
	double *raw_points = (double*)malloc(sizeof(double) * 4 * input->GetNumberOfPoints()),
		*save_pos = raw_points;
	for(vtkIdType i = 0;i < input->GetNumberOfPoints();i++, save_pos += 4)
	{
		input->GetPoint(i, save_pos);
		save_pos[3] = inScalars->GetValue(i);
	}
	double dim[3];
	
	for (dim[2] = m_Bounds.zmin;dim[2] <= m_Bounds.zmax;dim[2]+=m_Interval[2])
	{
		for (dim[1] = m_Bounds.ymin;dim[1] <= m_Bounds.ymax;dim[1]+=m_Interval[1])
		{
			for (dim[0] = m_Bounds.xmin;dim[0] <= m_Bounds.xmax;dim[0]+=m_Interval[0])
			{
				double min_dis = VTK_FLOAT_MAX;
				double val = VTK_FLOAT_MAX;
				save_pos = raw_points;
				for(vtkIdType i = 0; i < input->GetNumberOfPoints(); i++)
				{
					double dis = sqrt(vtkMath::Distance2BetweenPoints(save_pos, dim));
					if (min_dis > dis)
					{
						min_dis = dis;
						val = save_pos[3];
					}
					save_pos += 4;
				}
				if (val == VTK_FLOAT_MAX)
					val = m_NullValue;
				outpoints->InsertNextPoint(dim);
				outScalars->InsertNextTuple1(val);
			}
		}
	}
	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	return 1;
}


//----------------------------------------------------------------------------
void vtkNearestNeighborFilter::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}

