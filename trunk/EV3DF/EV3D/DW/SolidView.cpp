#include "StdVtk.h"
#include "SolidView.h"

SolidView::SolidView(SolidDoc* Doc)
:m_visable(false)
{
}

SolidView::~SolidView(void)
{
}

void SolidView::SetVisable( bool show )
{
	m_visable = show;
	if (show)
		m_Renderer->AddActor(m_actor);
	else
		m_Renderer->RemoveActor(m_actor);
}

void SolidView::SetRenderTarget( vtkSmartPointer<vtkRenderer> renderer )
{
	m_Renderer = renderer;
}
