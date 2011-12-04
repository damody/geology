#include "vtkKdtreeLimitedInverseDistanceFilterCuda.h"

#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"
#include "KdtreeLimitedInverseDistance.h"
#include "kd_tree.h"

vtkStandardNewMacro(vtkKdtreeLimitedInverseDistanceFilterCuda);

vtkKdtreeLimitedInverseDistanceFilterCuda::vtkKdtreeLimitedInverseDistanceFilterCuda()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
	m_PowerValue = 2.0;
	m_Radius = 15000;
	m_LimitNum = 100;
}

vtkKdtreeLimitedInverseDistanceFilterCuda::~vtkKdtreeLimitedInverseDistanceFilterCuda()
{

}

int vtkKdtreeLimitedInverseDistanceFilterCuda::RequestData(vtkInformation *vtkNotUsed(request),
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
	int h_total = input->GetNumberOfPoints();
	vtkDoubleArray* inScalars = (vtkDoubleArray*)(input->GetPointData()->GetScalars());
	double *raw_points = (double*)malloc(sizeof(double) * 4 * h_total),
		*save_pos = raw_points;
	point *pts = (point*)malloc(sizeof(point) * h_total),
		*save_pts = pts;
	kd_tree mykd;
	for(vtkIdType i = 0;i < h_total;i++, save_pos += 4)
	{
		input->GetPoint(i, save_pos);
		save_pos[3] = inScalars->GetValue(i);
		save_pts->x = save_pos[0];
		save_pts->y = save_pos[1];
		save_pts->z = save_pos[2];
		++save_pts;
	}
	mykd.BuildKdtree(pts, h_total);
	kdtree_setdata(&mykd, pts, h_total, m_LimitNum);
	
	InterpolationInfo info(h_total);
	info.GetPosFromXYZArray(raw_points);
	free(raw_points);
	info.interval[0] = m_Interval[0];
	info.interval[1] = m_Interval[1];
	info.interval[2] = m_Interval[2];
	info.SetBounds(m_Bounds);
	int size = KdtreeLimitedInverseDistance_SetData(&info, m_PowerValue, m_Radius, m_LimitNum, m_NullValue);
	float *outdata = new float[size];
	KdtreeLimitedInverseDistance_ComputeData(outdata, m_CudaThreadNum);
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
	kdtree_freedata();
	free(pts);
	return 1;
}


//----------------------------------------------------------------------------
void vtkKdtreeLimitedInverseDistanceFilterCuda::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}

void vtkKdtreeLimitedInverseDistanceFilterCuda::SetPowerValue( double v )
{
	m_PowerValue = v;
}

double vtkKdtreeLimitedInverseDistanceFilterCuda::GetPowerValue()
{
	return m_PowerValue;
}


void vtkKdtreeLimitedInverseDistanceFilterCuda::SetLimitRadius(double r)
{
	assert(r > 0);
	m_Radius = r;
	m_limitMethod = LIMIT_RADIUS;
}

void vtkKdtreeLimitedInverseDistanceFilterCuda::SetNumOfLimitPoints(int n)
{
	assert(n > 0);
	m_LimitNum = n;
	m_limitMethod = LIMIT_NUMBER;
}

double vtkKdtreeLimitedInverseDistanceFilterCuda::GetLimitRadius()
{
	if (LIMIT_RADIUS == m_limitMethod)
		return m_Radius;
	else
		return -1;
}

int vtkKdtreeLimitedInverseDistanceFilterCuda::GetNumOfLimitPoints()
{
	if (LIMIT_NUMBER == m_limitMethod)
		return m_LimitNum;
	else
		return -1;
}
