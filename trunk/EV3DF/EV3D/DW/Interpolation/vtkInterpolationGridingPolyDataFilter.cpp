#include "vtkInterpolationGridingPolyDataFilter.h"

#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"
#include <cassert>
vtkStandardNewMacro(vtkInterpolationGridingPolyDataFilter);

vtkInterpolationGridingPolyDataFilter::vtkInterpolationGridingPolyDataFilter()
{
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);
}

vtkInterpolationGridingPolyDataFilter::~vtkInterpolationGridingPolyDataFilter()
{

}
void vtkInterpolationGridingPolyDataFilter::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
// Customer function
void vtkInterpolationGridingPolyDataFilter::SetBounds( const double bounds[] )
{
	m_bounds.SetBounds(bounds);
}

void vtkInterpolationGridingPolyDataFilter::GetBounds( double bounds[] )
{
	m_bounds.GetBounds(bounds);
}

void vtkInterpolationGridingPolyDataFilter::SetInterval( double inter )
{
	assert(inter>0);
	m_interval = inter;
}

int vtkInterpolationGridingPolyDataFilter::NumOfInterpolationPoints()
{
	return NumOfXPoints()*NumOfYPoints()*NumOfZPoints();
}

int vtkInterpolationGridingPolyDataFilter::NumOfXPoints()
{
	return floor((m_bounds.xmax-m_bounds.xmin)/m_interval);
}

int vtkInterpolationGridingPolyDataFilter::NumOfYPoints()
{
	return floor((m_bounds.ymax-m_bounds.ymin)/m_interval);
}

int vtkInterpolationGridingPolyDataFilter::NumOfZPoints()
{
	return floor((m_bounds.zmax-m_bounds.zmin)/m_interval);
}

int vtkInterpolationGridingPolyDataFilter::RequestData( vtkInformation *vtkNotUsed(request),
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
	input->GetPoints()->InsertNextPoint(1.0, 1.0, 1.0);
	output->ShallowCopy(input);
	return 1;
}

double vtkInterpolationGridingPolyDataFilter::PointsDistanceSquare( double pos1[], double pos2[] )
{
	double dis[3];
	for (int i=0;i<3;i++)
		dis[i] = pos1[i]-pos2[i];
	return  dis[0]*dis[0] +
		dis[1]*dis[1] +
		dis[2]*dis[2];
}
