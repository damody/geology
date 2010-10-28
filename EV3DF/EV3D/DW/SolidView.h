#pragma once
#include <vtkActor.h>
#include <vtkLookupTable.h>
#include <vtkPolyDataMapper.h>
/**
顯示單元
*/
class SolidView
{
public:
	vtkActor		*m_actor;
	vtkLookupTable		*m_ltable;
	vtkPolyDataMapper	*m_polydataMapper;
	SolidView(void);
	virtual ~SolidView(void);
};
