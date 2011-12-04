#include "vtkLimitedInverseDistanceFilterCuda.h"

#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"
#include "LimitedInverseDistance.h"

vtkStandardNewMacro(vtkLimitedInverseDistanceFilterCuda);

vtkLimitedInverseDistanceFilterCuda::vtkLimitedInverseDistanceFilterCuda()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
	m_LimitNum = VTK_INT_MAX;
	m_Radius = VTK_FLOAT_MAX;
	m_limitMethod = LIMIT_NUMBER;
	m_PowerValue = 2.0;
}

vtkLimitedInverseDistanceFilterCuda::~vtkLimitedInverseDistanceFilterCuda()
{

}

int vtkLimitedInverseDistanceFilterCuda::RequestData(vtkInformation *vtkNotUsed(request),
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
	info.GetPosFromXYZArray(raw_points);
	free(raw_points);
	info.interval[0] = m_Interval[0];
	info.interval[1] = m_Interval[1];
	info.interval[2] = m_Interval[2];
	info.SetBounds(m_Bounds);
	int size = LimitedInverseDistance_SetData(&info, m_PowerValue, m_Radius, m_LimitNum, m_NullValue);
	float *outdata = new float[size];
	LimitedInverseDistance_ComputeData(outdata, m_CudaThreadNum);
	double dim[3];
	int outindex = 0;
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
				outScalars->InsertNextTuple1(outdata[outindex++]);
			}
		}
	}
	{ // fix Numerical error
		m_Bounds.xmax -= m_Interval[0]*0.5;
		m_Bounds.ymax -= m_Interval[1]*0.5;
		m_Bounds.zmax -= m_Interval[2]*0.5;
	}
	delete outdata;
	printf("size: %d, outindex: %d\n", size, outindex);
	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	return 1;
}


//----------------------------------------------------------------------------
void vtkLimitedInverseDistanceFilterCuda::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}

void vtkLimitedInverseDistanceFilterCuda::SetPowerValue( double v )
{
	m_PowerValue = v;
}

double vtkLimitedInverseDistanceFilterCuda::GetPowerValue()
{
	return m_PowerValue;
}

void vtkLimitedInverseDistanceFilterCuda::SetLimitRadius(double r)
{
	assert(r > 0);
	m_Radius = r;
	m_limitMethod = LIMIT_RADIUS;
}

void vtkLimitedInverseDistanceFilterCuda::SetNumOfLimitPoints(int n)
{
	assert(n > 0);
	m_LimitNum = n;
	m_limitMethod = LIMIT_NUMBER;
}

double vtkLimitedInverseDistanceFilterCuda::GetLimitRadius()
{
	if (LIMIT_RADIUS == m_limitMethod)
		return m_Radius;
	else
		return -1;
}

int vtkLimitedInverseDistanceFilterCuda::GetNumOfLimitPoints()
{
	if (LIMIT_NUMBER == m_limitMethod)
		return m_LimitNum;
	else
		return -1;
}
