#pragma once
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
	
public:	
	SolidView()
	{
		vtkSmartNew(m_actor);
		vtkSmartNew(m_Renderer);
		vtkSmartNew(m_actor);
		vtkSmartNew(m_actor);
		
	}
	virtual ~SolidView(void);
	int		SetDoc(SolidDoc* doc);
	void		SetType(int type) {m_Type = type;}
	int		GetType() {return m_Type;}
	void		SetVisable(bool show);
	int		GetVisable() {return m_Type;}
	SolidDoc_Sptr	GetParentDoc(){return m_ParentDoc;}
	SolidCtrl_Sptr	GetParentCtrl(){return m_ParentCtrl;}
	void		SetSetting(SEffect_Setting_Sptr setting);
	void		SetRenderTarget(vtkRenderer_Sptr renderer);
	void		Update();
	void		SetColorTable();
private:
	SolidView(SolidDoc_Sptr& Doc);
private:
	SolidDoc_Sptr		m_ParentDoc;
	SolidCtrl_Sptr		m_ParentCtrl;
	bool			m_visable;	///< 能見度
	int			m_Type;		///< effect種類
	vtkRenderer_Sptr	m_Renderer;
	vtkActor_Sptr		m_actor;
	vtkLookupTable_Sptr	m_ltable;
	vtkPolyDataMapper_Sptr	m_polydataMapper;
	SEffect_Setting_Sptr	m_SEffect_Setting;
private:	
	friend SolidCtrl;
	friend SolidDoc;
};
