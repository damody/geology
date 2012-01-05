#pragma once
#pragma warning(disable:4201)

//data bound, it include max and min point
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
	operator double*() {return m_bounds;}
	void SetBounds(const vtkBounds& b){*this = b;}
	void SetBounds(const double bounds[]);
	void GetBounds(double bounds[]);
	double Xlen(){return xmax-xmin;}			//get x length
	double Ylen(){return ymax-ymin;}			//get y length
	double Zlen(){return zmax-zmin;}			//get z length
	double Xmid(){return (xmax+xmin)/2.0;}		//get x center
	double Ymid(){return (ymax+ymin)/2.0;}		//get y center
	double Zmid(){return (zmax+zmin)/2.0;}		//get z center
};
