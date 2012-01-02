// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
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
	int h_total = input->GetNumberOfPoints();
	
	InterpolationInfo info(h_total);
	info.GetPosFromXYZArray(raw_points, h_total);
	free(raw_points);
	info.interval[0] = m_Interval[0];
	info.interval[1] = m_Interval[1];
	info.interval[2] = m_Interval[2];
	info.SetBounds(m_Bounds);
	int size = NearestNeighbor_SetData(&info);
	float *outdata = new float[size];
	NearestNeighbor_ComputeData(outdata);
	double dim[3];
	int outindex = 0;
	for (dim[2] = m_Bounds.zmin;dim[2] <= m_Bounds.zmax;dim[2]+=m_Interval[2])
	{
		for (dim[1] = m_Bounds.ymin;dim[1] <= m_Bounds.ymax;dim[1]+=m_Interval[1])
		{
			for (dim[0] = m_Bounds.xmin;dim[0] <= m_Bounds.xmax;dim[0]+=m_Interval[0])
			{
				outpoints->InsertNextPoint(dim);
				outScalars->InsertNextTuple1(outdata[outindex++]);
			}
		}
	}
	delete outdata;
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

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
