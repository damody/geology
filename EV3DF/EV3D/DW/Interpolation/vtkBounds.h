#pragma once
#pragma warning(disable:4201)
class vtkBounds
{
public:
	// xmin: bounds[0]
	// xmax: bounds[1]
	// ymin: bounds[2]
	// ymax: bounds[3]
	// zmin: bounds[4]
	// zmax: bounds[5]
	union
	{
		double m_bounds[6];
		struct 
		{
			double  xmin,
				xmax,
				ymin,
				ymax,
				zmin,
				zmax;
		};
	};
	vtkBounds();
	vtkBounds(double data[]);
	vtkBounds(	double xmin,
			double xmax,
			double ymin,
			double ymax,
			double zmin,
			double zmax);
	void SetBounds(const double bounds[]);
	void GetBounds(double bounds[]);
};
