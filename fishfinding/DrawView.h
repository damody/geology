#pragma once
#include "StdVtkWx.h"
#include "NmeaCell.h"
#include "TShape.hpp"
#include <boost/shared_ptr.hpp>

class DrawView
{
public:
	struct DataPoint
	{
		union
		{
			struct
			{
				double N, E;
			};
			struct
			{
				double x, y;
			};
		};
		double depth;
	};
	typedef std::vector<DataPoint> DataPoints;
public:
	DrawView();
	void SetHwnd(HWND hwnd);
	void ReSize(int w, int h);
	void AddDataList(const nmeaINFOs& infos);
	void AddData(const nmeaINFO& info);
	void Clear();
	void Render();
	void SetRect(const Rectf& rect) {m_area = rect;}
	void AddTest();
	void FocusLast();
	void SetFocusHeight(double height){m_focus_height = height;}
	void NormalLook(double angle = 0);
 	void SetHSColor(unsigned char r, unsigned char g, unsigned char b);
 	void SetDEColor(unsigned char r, unsigned char g, unsigned char b);
private:
	DataPoints	m_raw_points;
	Rectf		m_area;
	double		m_focus_height;
	unsigned char	m_ucHSColor[3],
			m_ucDEColor[3];
	/// hs = horizontal surface
	/// de = depth surface
	vtkPoints_Sptr		m_hs_points,
				m_de_points;
	vtkCellArray_Sptr	m_hs_vertices,
				m_de_vertices;
	vtkImageData_Sptr	m_imagedata;	// not use now
	vtkPolyData_Sptr	m_hs_poly,
				m_de_poly;
	vtkCellArray_Sptr	m_hs_lines,
				m_de_lines;
	vtkPolyDataMapper_Sptr	m_hs_Mapper,
				m_de_Mapper;
	vtkActor_Sptr		m_hs_Actor,
				m_de_Actor;
	vtkRenderer_Sptr	m_Renderer;
	vtkRenderWindow_Sptr	m_RenderWindow;
	vtkCamera_Sptr		m_Camera;
	vtkAxesActor_Sptr	m_Axes;
	vtkAppendPolyData_Sptr	m_Append_hs,
				m_Append_depth;
	vtkLookupTable_Sptr	m_hs_lut,
				m_depth_lut;
	vtkUnsignedCharArray_Sptr	m_hs_colors,
					m_de_colors;
	vtkRenderWindowInteractor_Sptr	m_WindowInteractor;
	vtkOrientationMarkerWidget_Sptr	m_Axes_widget;
	vtkInteractorStyleTrackballCamera_Sptr	m_style;
};
