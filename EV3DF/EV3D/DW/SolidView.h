#pragma once

#include <vtkActor.h>
#include <vtkLookupTable.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include "SolidDefine.h"
/**
顯示單元
*/
class SolidView
{
public:
	enum {
		CHANGE_OK,	///< 改變渲染對象成功
		CHANGE_FAIL	///< 改變渲染對象失敗
	};
	vtkSmartPointer<vtkActor>		m_actor;
	vtkSmartPointer<vtkLookupTable>		m_ltable;
	vtkSmartPointer<vtkPolyDataMapper>	m_polydataMapper;
	bool	m_visable;
	
	virtual ~SolidView(void);
	void SetDoc()
	{
		
	}
	SolidDoc*	GetParentDoc();
	SolidCtrl*	GetParentCtrl();
	void SetVisable(bool show);
	void SetRenderTarget(vtkSmartPointer<vtkRenderer> renderer);
private:
	vtkSmartPointer<vtkRenderer>		m_Renderer;
	SolidView(SolidDoc* Doc);
	friend SolidCtrl;
	friend SolidDoc;
};
