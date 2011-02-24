#include "vtkNearestNeighborFilterCuda.h"

#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"
#include "NearestNeighbor.h"

vtkStandardNewMacro(vtkNearestNeighborFilterCuda);

vtkNearestNeighborFilterCuda::vtkNearestNeighborFilterCuda()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
}

vtkNearestNeighborFilterCuda::~vtkNearestNeighborFilterCuda()
{

}

int vtkNearestNeighborFilterCuda::RequestData(vtkInformation *vtkNotUsed(request),
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
		std::cout << "Point " << i << " : (" << save_pos[0] << " " << save_pos[1] << " " << save_pos[2] << ")" << std::endl;
#endif // _DEBUG
	}
	int h_total = input->GetNumberOfPoints();
	
	InterpolationInfo info(h_total);
	info.GetPosFromXYZArray(raw_points);
	free(raw_points);
	info.interval[0] = m_interval;
	info.interval[1] = m_interval;
	info.interval[2] = m_interval;
	info.min[0] = m_bounds.xmin;
	info.min[1] = m_bounds.ymin;
	info.min[2] = m_bounds.zmin;
	info.max[0] = m_bounds.xmax;
	info.max[1] = m_bounds.ymax;
	info.max[2] = m_bounds.zmax;
	int size = NearestNeighbor_SetData(&info);
	float *outdata = new float[size];
	NearestNeighbor_ComputeData(outdata);
	double dim[3];
	int outindex = 0;
	for (dim[2] = m_bounds.zmin;dim[2] <= m_bounds.zmax;dim[2]+=m_interval)
	{
		for (dim[1] = m_bounds.ymin;dim[1] <= m_bounds.ymax;dim[1]+=m_interval)
		{
			for (dim[0] = m_bounds.xmin;dim[0] <= m_bounds.xmax;dim[0]+=m_interval)
			{
				outpoints->InsertNextPoint(dim);
				outScalars->InsertNextTuple1(outdata[outindex++]);
			}
		}
	}
	printf("size: %d, outindex: %d\n", size, outindex);
	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	return 1;
}


//----------------------------------------------------------------------------
void vtkNearestNeighborFilterCuda::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}

