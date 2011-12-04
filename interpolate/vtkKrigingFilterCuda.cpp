#include "vtkKrigingFilterCuda.h"

#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"
#include "kriging interpolation/kriging.h"
#include "SimpleOrdinaryKriging.h"

vtkStandardNewMacro(vtkKrigingFilterCuda);

vtkKrigingFilterCuda::vtkKrigingFilterCuda()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
	m_PowerValue = 2.0;
}

vtkKrigingFilterCuda::~vtkKrigingFilterCuda()
{

}

int vtkKrigingFilterCuda::RequestData(vtkInformation *vtkNotUsed(request),
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
	const int MAX_POINTS = input->GetNumberOfPoints();
	double *raw_points = (double*)malloc(sizeof(double) * 4 * input->GetNumberOfPoints()),
		*save_pos = raw_points;

	int smpnum = input->GetNumberOfPoints();
	std::vector<Record3> smpset(smpnum);
	double smppos[3];
	for(vtkIdType i = 0; i < smpnum; i++, save_pos += 4)
	{
		input->GetPoint(i, smppos);
		dPos3 newpos(smppos);
		smpset[i].pos = newpos;
		smpset[i].val = inScalars->GetValue(i);

		input->GetPoint(i, save_pos);
		save_pos[3] = inScalars->GetValue(i);
	}
	kriging kf;
	kf.SetInterval(m_Interval);
	kf.SetSample(smpset);
	kf.Initial(smpnum, 1);
	kf.SetAutoGetDistStep(m_AutoGetStep);
	kf.SetPreCompute(true);
	kf.Estimate(Variogram::VARIO_SPERICAL, 2, m_DistStep);
	printf("DistStep:%f, \n", kf.GetDistStep());

	double dim[3];
	InterpolationInfo info(smpnum);
	info.GetPosFromXYZArray(raw_points);
	info.interval[0] = m_Interval[0];
	info.interval[1] = m_Interval[1];
	info.interval[2] = m_Interval[2];
	info.SetBounds(m_Bounds);
	float* matvec = new float[(smpnum+1)*(smpnum+1)];
	kf.GetFloatMatrix(matvec);
	int size = OK_SetData(&info, matvec, kf.GetNugget(), kf.GetSill(), kf.GetRange());

	float *h_outdata;
	h_outdata = new float[size];
	OK_ComputeData(h_outdata, m_CudaThreadNum, m_UseSharedMem);
	{ // fix Numerical error
		m_Bounds.xmax += m_Interval[0]*0.5;
		m_Bounds.ymax += m_Interval[1]*0.5;
		m_Bounds.zmax += m_Interval[2]*0.5;
	}
	int outindex=0;
	for (dim[2] = m_Bounds.zmin; dim[2] <= m_Bounds.zmax; dim[2]+=m_Interval[2])
	{
		for (dim[1] = m_Bounds.ymin; dim[1] <= m_Bounds.ymax; dim[1]+=m_Interval[1])
		{
			for (dim[0] = m_Bounds.xmin; dim[0] <= m_Bounds.xmax; dim[0]+=m_Interval[0])
			{
				outpoints->InsertNextPoint(dim);
				outScalars->InsertNextTuple1(h_outdata[outindex++]);
			}
		}
	}
	{ // fix Numerical error
		m_Bounds.xmax -= m_Interval[0]*0.5;
		m_Bounds.ymax -= m_Interval[1]*0.5;
		m_Bounds.zmax -= m_Interval[2]*0.5;
	}
	delete h_outdata;
	delete matvec;

	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	free(raw_points);
	return 1;
}


//----------------------------------------------------------------------------
void vtkKrigingFilterCuda::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}