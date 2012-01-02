// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
#pragma once

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkBoxWidget.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>

// Select Bounding in vtk
class SelctionBounding
{
private:
	vtkSmartPointer<vtkSphereSource> m_sphereSource;
	vtkSmartPointer<vtkPolyDataMapper> m_mapper;
	vtkSmartPointer<vtkActor> m_actor;
	vtkSmartPointer<vtkBoxWidget> m_boxWidget;
	double m_bounds[6];
public:
	// send vtkRenderWindowInteractor to this class to create bounding
	SelctionBounding::SelctionBounding(vtkRenderWindowInteractor* iren, double pos[3]);
	~SelctionBounding(void);

	// for get bounding
	double* GetBounds();
	void GetBounds(double bound[]);
	void SetVisible(bool setting);
};

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)