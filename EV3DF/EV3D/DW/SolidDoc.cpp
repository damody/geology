// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)

#include "StdWxVtk.h"
#include "SolidDoc.h"
#include "SolidCtrl.h"
#include "SolidView.h"

SolidDoc::SolidDoc(vtkBounds bound) :
	m_bounds(bound)
{
}

void SolidDoc::SetPolyData(vtkPolyData_Sptr polyData)
{
	m_PolyData = polyData;
}

void SolidDoc::SetImageData(vtkImageData_Sptr imageData)
{
	m_ImageData = imageData;
}

void SolidDoc::RmAllView()
{
	for (SolidView_Sptrs::iterator it = m_SolidViews.begin(); it != m_SolidViews.end(); it++)
	{
		(*it)->SetVisable(false);
		SolidView_Sptr	tmp;
		tmp.swap(*it);
	}
	m_SolidViews.clear();
}
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
