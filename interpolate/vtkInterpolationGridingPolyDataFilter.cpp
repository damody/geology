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
	m_NullValue = VTK_FLOAT_MIN;;
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
	m_Bounds.SetBounds(bounds);
}

void vtkInterpolationGridingPolyDataFilter::GetBounds( double bounds[] )
{
	m_Bounds.GetBounds(bounds);
}

void vtkInterpolationGridingPolyDataFilter::SetInterval( double x, double y, double z )
{
	assert(x > 0);
	assert(y > 0);
	assert(z > 0);
	m_Interval[0] = x;
	m_Interval[1] = y;
	m_Interval[2] = z;
}

void vtkInterpolationGridingPolyDataFilter::SetInterval( double inter[] )
{
	memcpy(m_Interval, inter, sizeof(double)*3);
}

int vtkInterpolationGridingPolyDataFilter::NumOfInterpolationPoints()
{
	return NumOfXPoints()*NumOfYPoints()*NumOfZPoints();
}

int vtkInterpolationGridingPolyDataFilter::NumOfXPoints()
{
	return floor((m_Bounds.xmax-m_Bounds.xmin)/m_Interval[0]+1);
}

int vtkInterpolationGridingPolyDataFilter::NumOfYPoints()
{
	return floor((m_Bounds.ymax-m_Bounds.ymin)/m_Interval[1]+1);
}

int vtkInterpolationGridingPolyDataFilter::NumOfZPoints()
{
	return floor((m_Bounds.zmax-m_Bounds.zmin)/m_Interval[2]+1);
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
