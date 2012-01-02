// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
#include "SelctionBounding.h"

SelctionBounding::SelctionBounding(vtkRenderWindowInteractor* iren, double pos[3])
{

	m_sphereSource = vtkSmartPointer<vtkSphereSource>::New();
	m_sphereSource->SetCenter(pos[0], pos[1], pos[2]);
	m_sphereSource->SetRadius(10000);

	m_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_mapper->SetInputConnection(m_sphereSource->GetOutputPort());
	m_actor = vtkSmartPointer<vtkActor>::New();
	m_actor->SetMapper(m_mapper);
	m_actor->SetVisibility(false);

	m_boxWidget = vtkSmartPointer<vtkBoxWidget>::New();
	m_boxWidget->SetInteractor(iren);
	m_boxWidget->SetPlaceFactor(1.25);
	m_boxWidget->SetProp3D(m_actor);
	m_boxWidget->PlaceWidget();

	m_boxWidget->RotationEnabledOff();
	m_boxWidget->On();
}

SelctionBounding::~SelctionBounding(void)
{
}

void SelctionBounding::SetVisible( bool setting )
{
	if (setting)
		m_boxWidget->On();
	else
		m_boxWidget->Off();
}

void SelctionBounding::GetBounds( double bound[] )
{
	vtkSmartPointer<vtkPolyData> pd = vtkSmartPointer<vtkPolyData>::New();
	m_boxWidget->GetPolyData(pd);
	pd->GetBounds(bound);
}

double* SelctionBounding::GetBounds()
{
	vtkSmartPointer<vtkPolyData> pd = vtkSmartPointer<vtkPolyData>::New();
	m_boxWidget->GetPolyData(pd);
	pd->GetBounds(m_bounds);
	return m_bounds;
}
// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
