#pragma once

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkBoxWidget.h>
#include <vtkSphereSource.h>
//#include <vtkCamera.h>
//#include <vtkCommand.h>
//#include <vtkConeSource.h>
//#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>

class SelctionBounding
{
private:
	vtkSmartPointer<vtkSphereSource> m_sphereSource;
	vtkSmartPointer<vtkPolyDataMapper> m_mapper;
	vtkSmartPointer<vtkActor> m_actor;
	vtkSmartPointer<vtkBoxWidget> m_boxWidget;
	double m_bounds[6];
public:
	SelctionBounding::SelctionBounding(vtkRenderWindowInteractor* iren, double pos[3]);
	~SelctionBounding(void);

	double* GetBounds();
	void GetBounds(double bound[]);
	void SetVisible(bool setting);
};
