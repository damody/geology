// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
#include "vtkBounds.h"
#include <cstdlib>
#include <memory>
#include <cassert>

vtkBounds::vtkBounds()
{
	memset(m_bounds, 0, sizeof(m_bounds));
}

vtkBounds::vtkBounds( double data[] )
{
	memcpy(m_bounds, data, sizeof(m_bounds));
}

vtkBounds::vtkBounds( double xmin, double xmax, double ymin, double ymax, double zmin, double zmax )
{
	m_bounds[0] = xmin;
	m_bounds[1] = xmax;
	m_bounds[2] = ymin;
	m_bounds[3] = ymax;
	m_bounds[4] = zmin;
	m_bounds[5] = zmax;
}

void vtkBounds::SetBounds( const double bounds[] )
{
	assert(bounds!=NULL);
	memcpy(m_bounds, bounds, sizeof(double)*6);
}

void vtkBounds::GetBounds( double bounds[] )
{
	assert(bounds!=NULL);
	memcpy(bounds, m_bounds, sizeof(double)*6);
}
// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
