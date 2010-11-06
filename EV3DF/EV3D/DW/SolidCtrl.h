#pragma once
#include <vector>
#include <boost/shared_ptr.hpp>
#include "BoxArea.h"
#include "SolidDefine.h"
#include "SJCScalarField3.h"
#include "DWHistogram.h"
/**
control unit
*/
class SolidCtrl
{
public:
	SolidCtrl(vtkRenderer_Sptr renderer):m_Renderer(renderer){}
	SolidView_Sptrs		m_SolidViewPtrs;
	SolidDoc_Sptrs		m_SolidDocPtrs;
	vtkRenderer_Sptr	m_Renderer;
	vtkRenderWindow_Sptr	m_RenderWindow;
	SJCScalarField3d	*m_sf3d;
public:
	int SetData(SJCScalarField3d* sf3d);
	int RmView(SolidView_Sptr& view);
	int RmDoc(SolidDoc_Sptr& doc);
	SolidDoc_Sptr	NewDoc();
	SolidView_Sptr	NewView(SEffect_Sptr& area, SolidDoc_Sptr& doc);
	~SolidCtrl(void)
	{
	}

};
