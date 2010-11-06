#include "StdVtk.h"
#include "SolidDoc.h"
#include "SolidCtrl.h"


SolidDoc::SolidDoc(BoxArea_Sptr area):
m_area(area)
{
}

int SolidDoc::SetPolyData( vtkPolyData_Sptr polyData )
{
	m_PolyData = polyData;
	return SET_OK;
}

int SolidDoc::SetImageData( vtkImageData_Sptr imageData )
{
	m_ImageData = imageData;
	return SET_OK;
}

