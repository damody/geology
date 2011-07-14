#include "SelectionSphere.h"



SelectionSphere::SelectionSphere( double pos[3], double radius )
{
	m_sphereSource = vtkSmartPointer<vtkSphereSource>::New();
	m_sphereSource->SetCenter(pos[0], pos[1], pos[2]);
	m_sphereSource->SetRadius(radius);
	
	m_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_mapper->SetInputConnection(m_sphereSource->GetOutputPort());
	 
	m_actor = vtkSmartPointer<vtkActor>::New();
	m_actor->SetMapper(m_mapper);

	m_IsSelect = false;
}

SelectionSphere::~SelectionSphere(void)
{
	//m_render->RemoveActor(m_actor);
}

void SelectionSphere::SetSelect()
{
	if (m_IsSelect)
		return;
	m_actor->GetProperty()->SetColor(1, 0, 0);
	m_IsSelect = true;
}

void SelectionSphere::SetUnselect()
{
	if (!m_IsSelect)
		return;
	m_actor->GetProperty()->SetColor(1, 1, 1);
	m_IsSelect = false;
}
