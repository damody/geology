#include "StdVtk.h"
#include "SolidDoc.h"
#include "SolidCtrl.h"
#include "SolidView.h"


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

void SolidDoc::RmAllView()
{
	for (SolidView_Sptrs::iterator it = m_SolidViews.begin();
		it != m_SolidViews.end();
		it++)
	{
		(*it)->SetVisable(false);
		SolidView_Sptr tmp;
		tmp.swap(*it);
	}
	m_SolidViews.clear();
}

