// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
#include <vtkSmartPointer.h>
#include <cassert>

#include "vtkHeatTranslationFilter.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"

#include "vtkInterpolationGridingPolyDataFilter.h"
#include "vtkInverseDistanceHeightFilter.h"
#include "vtkLimitedInverseDistanceHeightFilter.h"
#include "vtkNearestNeighborHeightFilter.h"

#define VTKSMART_PTR(x) \
	typedef vtkSmartPointer< x >	x##_Sptr; \
	typedef std::vector< x##_Sptr >	x##_Sptrs;

VTKSMART_PTR(vtkInterpolationGridingPolyDataFilter)
VTKSMART_PTR(vtkInverseDistanceFilter)
VTKSMART_PTR(vtkLimitedInverseDistanceFilter)
VTKSMART_PTR(vtkNearestNeighborFilter)

template <class T>
void vtkSmartNew(vtkSmartPointer<T>& Ptr)
{
	Ptr = vtkSmartPointer<T>::New();
	assert(Ptr.GetPointer() != 0);
}

vtkStandardNewMacro(vtkHeatTranslationFilter);
vtkHeatTranslationFilter::vtkHeatTranslationFilter(void)
{
	m_Volume = 0;
	m_NullValue = 0;
	m_Etotal = 0;
	m_Ejh = 0;
	m_Emw = 0;
	m_FilterType = INVERSE_FILTER;
	m_DoInterpolate = false;
}

vtkHeatTranslationFilter::~vtkHeatTranslationFilter(void)
{
}

int vtkHeatTranslationFilter::RequestData(vtkInformation *vtkNotUsed(request),
					  vtkInformationVector **inputVector,
					  vtkInformationVector *outputVector)
{
	// get the info objects
	vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
	// get the input and ouptut
	vtkPolyData *input = vtkPolyData::SafeDownCast(
		inInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkDoubleArray* inScalars = (vtkDoubleArray*)(input->GetPointData()->GetScalars());

	int ix = NumOfXPoints(), iy = NumOfYPoints(), iz = NumOfZPoints();
	double avgT = 0, heat = 0, tn;

	m_Volume = m_Interval[0]*m_Interval[1]*m_Interval[2];

	if (!m_DoInterpolate)
	{
		double pos[3];
		for (int i=0; i<input->GetNumberOfPoints(); i++)
		{
			input->GetPoint(i, pos);
			if ( PosInBound(pos) )
			{
				tn = inScalars->GetValue(i);
				if (tn >= m_HeatParmeter.LimitTemperature)
					heat+=CmpE(tn);
			}
		}
	}
	else
	{
		vtkInterpolationGridingPolyDataFilter_Sptr filter;
		switch(m_FilterType)
		{
		case INVERSE_FILTER:
			filter = vtkSmartPointer<vtkInverseDistanceFilter>::New();
			break;
		case LIMITED_FILTER:
			filter = vtkSmartPointer<vtkLimitedInverseDistanceFilter>::New();
			break;
		case NEARESTNEIGHBOR_FILTER:
			filter = vtkSmartPointer<vtkNearestNeighborFilter>::New();
			break;
		}
		double boun[6];
		input->GetBounds(boun);
		for (int i=0; i<6; i+=2)
			if (boun[i]>m_Bounds[i])
				m_Bounds[i] = boun[i];

		for (int i=1; i<6; i+=2)
			if (boun[i]<m_Bounds[i])
				m_Bounds[i] = boun[i];

		filter->SetInput(input);
		filter->SetBounds(m_Bounds);
		filter->SetInterval(m_Interval);
		filter->Update();

		vtkPolyData* newdata = filter->GetOutput();
		vtkDoubleArray* newScalars = (vtkDoubleArray*)newdata->GetPointData()->GetScalars();


		for (int i=0; i<newdata->GetNumberOfPoints(); i++)
		{
			tn = newScalars->GetValue(i);
			if (tn >= m_HeatParmeter.LimitTemperature)
				heat+=CmpE(tn);
		}
	}

	
	
	m_Etotal = heat;
	m_Ejh = heat*1000000000/WHJ/356/24/m_HeatParmeter.Life;
	m_Emw = m_Ejh*WHJ/1000000;

	return 1;
}



double vtkHeatTranslationFilter::CmpE( double Tn )
{
	double res = m_Volume/1000000000*m_HeatParmeter.Hv*m_HeatParmeter.Rt/m_HeatParmeter.Fppc*(Tn-m_HeatParmeter.Tzero)*WHJ;
	m_Etotal = res;
	m_Ejh = m_Volume/24/365*m_HeatParmeter.Hv*m_HeatParmeter.Rt/m_HeatParmeter.Fppc*(Tn-m_HeatParmeter.Tzero)/m_HeatParmeter.Life;
	m_Emw = m_Ejh*WHJ/1000000;
	return res;
}

bool vtkHeatTranslationFilter::PosInBound( double pos[] )
{
	if (pos[0] < m_Bounds.xmin || pos[0] > m_Bounds.xmax)
		return false;
	if (pos[1] < m_Bounds.ymin || pos[1] > m_Bounds.ymax)
		return false;
	if (pos[2] < m_Bounds.zmin || pos[2] > m_Bounds.zmax)
		return false;
	return true;
}

void vtkHeatTranslationFilter::SetNumberOfXYZ( double x, double y, double z )
{
	m_Interval[0] = (m_Bounds.xmax-m_Bounds.xmin)/x;
	m_Interval[1] = (m_Bounds.ymax-m_Bounds.ymin)/y;
	m_Interval[2] = (m_Bounds.zmax-m_Bounds.zmin)/z;
}

void vtkHeatTranslationFilter::SetNumberOfXYZ( double xyz[3] )
{
	m_Interval[0] = (m_Bounds.xmax-m_Bounds.xmin)/xyz[0];
	m_Interval[1] = (m_Bounds.ymax-m_Bounds.ymin)/xyz[1];
	m_Interval[2] = (m_Bounds.zmax-m_Bounds.zmin)/xyz[2];
}
// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
