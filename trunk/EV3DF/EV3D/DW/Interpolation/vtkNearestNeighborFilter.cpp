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
	vtkPoints * outpoints = vtkPoints::New();
	vtkDoubleArray* outScalars = vtkDoubleArray::New();
	vtkDoubleArray* inScalars = (vtkDoubleArray*)(input->GetPointData()->GetScalars());
	double *raw_points = (double*)malloc(sizeof(double) * 4 * input->GetNumberOfPoints()),
		*save_pos = raw_points;
	for(vtkIdType i = 0;i < input->GetNumberOfPoints();i++, save_pos += 4)
	{
		input->GetPoint(i, save_pos);
		save_pos[3] = inScalars->GetValue(i);
#ifdef _DEBUG
		//std::cout << "Point " << i << " : (" << save_pos[0] << " " << save_pos[1] << " " << save_pos[2] << ")" << std::endl;
#endif // _DEBUG
	}
	double dim[3];
	int count[3] = {0,0,0};
	for (dim[2] = m_bounds.zmin;dim[2] <= m_bounds.zmax;dim[2]+=m_interval[2])
	{
		count[2]++;
		for (dim[1] = m_bounds.ymin;dim[1] <= m_bounds.ymax;dim[1]+=m_interval[1])
		{
			if (count[2] == 1)
				count[1]++;
			for (dim[0] = m_bounds.xmin;dim[0] <= m_bounds.xmax;dim[0]+=m_interval[0])
			{
				if (count[1] == 1)
					count[0]++;
				double min_dis = VTK_DOUBLE_MAX;
				double val = VTK_DOUBLE_MAX;
				save_pos = raw_points;
				for(vtkIdType i = 0; i < input->GetNumberOfPoints(); i++)
				{
					double dis = PointsDistanceSquare(save_pos, dim);
					if (min_dis > dis)
					{
						min_dis = dis;
						val = save_pos[3];
					}
					save_pos += 4;
				}
				outpoints->InsertNextPoint(dim);
				outScalars->InsertNextTuple1(val);
			}
		}
	}
	printf("x:%d y:%d z:%d\n", count[0], count[1], count[2]);
	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	return 1;
}


//----------------------------------------------------------------------------
void vtkNearestNeighborFilter::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}

