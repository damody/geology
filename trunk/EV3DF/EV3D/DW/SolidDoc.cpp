#include "StdVtk.h"
#include "SolidDoc.h"

SolidDoc::SolidDoc()
{
}

SolidDoc::~SolidDoc(void)
{
}

int SolidDoc::SetPolyData( vtkPolyData_Sptr& polyData )
{
	m_PolyData = polyData;
	return SET_OK;
}

int SolidDoc::SetImageData( vtkImageData_Sptr& imageData )
{
	m_ImageData = imageData;
	return SET_OK;
}
