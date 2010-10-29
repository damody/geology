#pragma once
#include <vtkActor.h>
#include <vtkLookupTable.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
/**
顯示單元
*/
class SolidView
{
public:
	vtkSmartPointer<vtkActor>		m_actor;
	vtkSmartPointer<vtkLookupTable>		m_ltable;
	vtkSmartPointer<vtkPolyDataMapper>	m_polydataMapper;
	bool		m_visable;
	SolidView(void);
	virtual ~SolidView(void);
	void SetDoc()
	{
		
	}
	void SetVisable(bool show)
	{
		m_visable = show;
		if (show)
			m_Renderer->AddActor(m_actor);
		else
			m_Renderer->RemoveActor(m_actor);
	}
	void SetRenderTarget(vtkSmartPointer<vtkRenderer> renderer)
	{
		m_Renderer = renderer;
	}
private:
	vtkSmartPointer<vtkRenderer>		m_Renderer;
};
