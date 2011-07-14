#pragma once

#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>

class SelectionSphere
{
protected:
	vtkSmartPointer<vtkSphereSource> m_sphereSource;
	vtkSmartPointer<vtkPolyDataMapper> m_mapper;
	vtkSmartPointer<vtkActor> m_actor;
	bool m_IsSelect;
public:
	SelectionSphere( double pos[3], double radius);
	~SelectionSphere(void);

	void SetSelect();
	void SetUnselect();
	vtkActor* GetActor(){return m_actor;}
	
};
