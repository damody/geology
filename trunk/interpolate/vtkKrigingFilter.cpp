#include "vtkKrigingFilter.h"

#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"

#include "SimpleOrdinaryKriging.h"

#include "kriging interpolation/kriging.h"

vtkStandardNewMacro(vtkKrigingFilter);

vtkKrigingFilter::vtkKrigingFilter()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
	m_AutoGetStep = true;
	m_DistStep = 100;
}

vtkKrigingFilter::~vtkKrigingFilter()
{
}

int vtkKrigingFilter::RequestData(vtkInformation *vtkNotUsed(request),
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
	kf.Estimate(Variogram::VARIO_SPERICAL, 2, m_DistStep);
	printf("DistStep:%f, \n", kf.GetDistStep());

	double val, var;
	double dim[3];
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
				kf.GetPredictData(dim, val, var);
				outScalars->InsertNextTuple1(val);
			}
		}
	}
	{ // fix Numerical error
		m_Bounds.xmax -= m_Interval[0]*0.5;
		m_Bounds.ymax -= m_Interval[1]*0.5;
		m_Bounds.zmax -= m_Interval[2]*0.5;
	}
	output->SetPoints(outpoints);
	output->GetPointData()->SetScalars(outScalars);
	free(raw_points);
	return 1;
}


//----------------------------------------------------------------------------
void vtkKrigingFilter::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}


