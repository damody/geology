#pragma once
#include "SolidDefine.h"
/**

*/
class SolidDoc
{
public:
	SolidDoc_Sptr	GetParentDoc() const {return m_ParentDoc;}
	SolidCtrl*	GetParentCtrl() const {return m_ParentCtrl;}
	bool		HasPolyData() const;
	bool		HasImageData() const;
	void		RmAllView();
	SolidView_Sptrs	GetViewList() const;
	void		SetPolyData(vtkPolyData_Sptr polyData);
	void		SetImageData(vtkImageData_Sptr imageData);
	int		GridingData(const vtkBounds area);
private:
	SolidDoc(vtkBounds bound);
	Histogramd		m_histogram;
	SolidDoc_Sptr		m_ParentDoc;
	SolidCtrl		*m_ParentCtrl;
	SolidView_Sptrs		m_SolidViews;	/// 所有相關的views
	vtkPolyData_Sptr	m_PolyData;	/// 離散 data
	vtkImageData_Sptr	m_ImageData;	/// griding data
	vtkBounds		m_bounds;		/// 範圍
	
	friend SolidCtrl;
	friend SolidView;
};
