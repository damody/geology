#pragma once
#include "SolidDefine.h"
/**

*/
class SolidDoc
{
public:
	virtual ~SolidDoc(void);
	SolidDoc*	GetParentDoc() const;
	SolidCtrl*	GetParentCtrl() const;
	bool		HasPolyData() const;
	bool		HasImageData() const;
	void		RmAllView();
	SolidView_Sptrs	GetViewList() const;
	int		SetPolyData(vtkPolyData_Sptr& polyData);
	int		SetImageData(vtkImageData_Sptr& imageData);
	int		GridingData(const BoxArea_Sptr area);
private:
	SolidDoc();
	Histogramd		m_histogram;
	SolidDoc_Sptr		m_ParentDoc;
	SolidCtrl_Sptr		m_ParentCtrl;
	SolidView_Sptrs		m_SolidViews;	/// 所有相關的views
	vtkPolyData_Sptr	m_PolyData;	/// 離散 data
	vtkImageData_Sptr	m_ImageData;	/// griding data
	BoxArea_Sptr		m_area;		/// 範圍
	friend SolidCtrl;
	friend SolidView;
};
