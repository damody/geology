#include "StdVtk.h"
#include "SolidDoc.h"
#include "SolidCtrl.h"


SolidDoc::SolidDoc(BoxArea_Sptr area):
m_area(area)
{
}

void SolidDoc::SetPolyData( vtkPolyData_Sptr polyData )
{
	m_PolyData = polyData;
}

void SolidDoc::SetImageData( vtkImageData_Sptr imageData )
{
	m_ImageData = imageData;
}

