/////////////////////////////////////////////////////////////////////////////
// Name:        mygrid.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     04/10/2010 14:37:18
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////
#include "StdWxVtk.h"
// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif
#include "DW/SolidDefine.h"
#include "DW/SEffect.h"
#include "DW/SolidView.h"
#include "firstmain.h"
////@begin includes
////@end includes

#include "mygrid.h"
////@begin XPM images
////@end XPM images


/*
 * MyGrid type definition
 */

IMPLEMENT_DYNAMIC_CLASS( MyGrid, wxGrid )


/*
 * MyGrid event table definition
 */

BEGIN_EVENT_TABLE( MyGrid, wxGrid )

////@begin MyGrid event table entries
    EVT_GRID_CELL_CHANGE( MyGrid::OnCellChange )
    EVT_GRID_CMD_CELL_CHANGE( ID_GRID, MyGrid::OnGridCellChange )
    EVT_KEY_DOWN( MyGrid::OnKeyDown )

////@end MyGrid event table entries

END_EVENT_TABLE()


/*
 * MyGrid constructors
 */

MyGrid::MyGrid()
{
    Init();
}

MyGrid::MyGrid(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
    Init();
    Create(parent, id, pos, size, style);
}


/*
 * MyGrid creator
 */

bool MyGrid::Create(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
////@begin MyGrid creation
    wxGrid::Create(parent, id, pos, size, style);
    CreateControls();
////@end MyGrid creation
    return true;
}


/*
 * MyGrid destructor
 */

MyGrid::~MyGrid()
{
////@begin MyGrid destruction
////@end MyGrid destruction
}


/*
 * Member initialisation
 */

void MyGrid::Init()
{
////@begin MyGrid member initialisation
////@end MyGrid member initialisation
}


/*
 * Control creation for MyGrid
 */

void MyGrid::CreateControls()
{
}


/*
 * Should we show tooltips?
 */

bool MyGrid::ShowToolTips()
{
    return true;
}

/*
 * Get bitmap resources
 */

wxBitmap MyGrid::GetBitmapResource( const wxString& name )
{
    // Bitmap retrieval
////@begin MyGrid bitmap retrieval
    wxUnusedVar(name);
    return wxNullBitmap;
////@end MyGrid bitmap retrieval
}

/*
 * Get icon resources
 */

wxIcon MyGrid::GetIconResource( const wxString& name )
{
    // Icon retrieval
////@begin MyGrid icon retrieval
    wxUnusedVar(name);
    return wxNullIcon;
////@end MyGrid icon retrieval
}





void MyGrid::ConvertTo_BoundingBox()
{
	ReCreateGrid(1,1);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("Color"));		// 框的顏色
	SetRowLabelValue(i++, wxT("ThickDegree"));	// 框的粗細
}

void MyGrid::ConvertTo_Vertex()
{
	ReCreateGrid(3,1);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("MaxValue"));		// 大於這個值才顯示
	SetRowLabelValue(i++, wxT("MinValue"));		// 小於這個值才顯示
	SetRowLabelValue(i++, wxT("Size"));		// 點的顯示大小
}

void MyGrid::ConvertTo_Contour()
{
	ReCreateGrid(2,1);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("ContourValue"));	// 要做出切面的值
	SetRowLabelValue(i++, wxT("Alpha"));		// 切面透明度的值
}

void MyGrid::ConvertTo_Axes()
{
	ReCreateGrid(4,1);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("ThickDegree"));	// 軸向的粗細
	SetRowLabelValue(i++, wxT("XColor"));		// X軸向的顏色
	SetRowLabelValue(i++, wxT("YColor"));		// Z軸向的顏色
	SetRowLabelValue(i++, wxT("ZColor"));		// Z軸向的顏色
}

void MyGrid::ConvertTo_Ruler()
{
	ReCreateGrid(7,1);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("Target"));		// 量尺的對象
	SetRowLabelValue(i++, wxT("TargetAxes"));	// 量尺的軸向
	SetRowLabelValue(i++, wxT("StartPoint"));	// 起始點，調整後對象與軸向選項失效
	SetRowLabelValue(i++, wxT("EndPoint"));		// 結束點，調整後對象與軸向選項失效
	SetRowLabelValue(i++, wxT("Scalar"));		// 量尺的突出程度
	SetRowLabelValue(i++, wxT("ThickDegree"));	// 量尺的粗細
	SetRowLabelValue(i++, wxT("Color"));		// 量尺的顏色
}

void MyGrid::ConvertTo_ClipPlane()
{
	ReCreateGrid(2,1);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i, wxT("Axes"));
	const wxString choices[] =
	{
		_T("X Axes"),
		_T("Y Axes"),
		_T("Z Axes"),
	};
	SetCellEditor(i, 0, new wxGridCellChoiceEditor(WXSIZEOF(choices), choices));
	SetCellValue(i++, 0, choices[0]);
	SetRowLabelValue(i, wxT("Percent"));
	SetCellValue(i++, 0, wxT("0"));
}

void MyGrid::ConvertTo_ClipContour()
{
	ReCreateGrid(3,1);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("Axes"));
	SetRowLabelValue(i++, wxT("Percent"));
	SetRowLabelValue(i++, wxT("ContourValue"));
}

void MyGrid::ConvertTo_VolumeRender()
{
	ReCreateGrid(1,1);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("Color"));
}

void MyGrid::DeleteGrid()
{
	while (GetNumberRows())
	{
		DeleteRows();
	}
	while (GetNumberCols())
	{
		DeleteCols();
	}
}

void MyGrid::AppendGrid(int Rows, int Cols)
{
	while (Cols--)
		AppendCols();
	while (Rows--)
		AppendRows();
}

void MyGrid::ReCreateGrid(int Rows, int Cols)
{
	DeleteGrid();
	AppendGrid( Rows, Cols );
}


/*
 * wxEVT_GRID_CELL_CHANGED event handler for ID_GRID
 */

void MyGrid::OnCellChange( wxGridEvent& event )
{
	//MESSAGE("OnCellChange");
	int col = event.GetCol();
	int row = event.GetRow();
	wxString Value = GetCellValue(row, col);
	ChangeToView(col, row, Value);
}


/*
 * wxEVT_KEY_DOWN event handler for ID_GRID
 */

void MyGrid::OnKeyDown( wxKeyEvent& event )
{
	//int key = event.GetKeyCode();
}


/*
 * wxEVT_GRID_CMD_CELL_CHANGE event handler for ID_GRID
 */

void MyGrid::OnGridCellChange( wxGridEvent& event )
{
	//MESSAGE("OnGridCellChange");
	int col = event.GetCol();
	int row = event.GetRow();
	wxString Value = GetCellValue(row, col);
	ChangeToView(col, row, Value);
}

void MyGrid::ChangeToView( int col, int row, const wxString& data )
{
	SolidView_Sptr view = ((FirstMain*)GetParent())->m_ActiveView;
	SEffect_Sptr SuperSetting = view->GetEffect();
	double number = 0;
	data.ToDouble(&number);
	switch (view->GetType())
	{
	case SEffect::BOUNDING_BOX:
		break;
	case SEffect::VERTEX:
		break;
	case SEffect::CONTOUR:
		{
			Contour_Setting* setting = (Contour_Setting*)SuperSetting.get();
			if (0 == col && 0 == row)
			{
				setting->m_ContourValue = number;
			}
			else if (1 == row && 0 == col)
			{
				setting->m_alpha = number;
			}
		}
		break;
	case SEffect::AXES:
		break;
	case SEffect::CLIP_PLANE:
		{
			ClipPlane_Setting* setting = (ClipPlane_Setting*)SuperSetting.get();
			if (0 == col && 0 == row)
			{
				if (data == _T("X Axes"))
					setting->m_Axes = 0;
				else if (data == _T("Y Axes"))
					setting->m_Axes = 1;
				else if (data == _T("Z Axes"))
					setting->m_Axes = 2;
				setting->m_Percent = 0;
			}
			else if (1 == row && 0 == col)
			{
				setting->m_Percent = number;
			}
		}
		break;
	case SEffect::RULER:
		break;
	case SEffect::CLIP_CONTOUR:
		break;
	case SEffect::VOLUME_RENDERING:
		break;
	}
	view->Update();
}

bool MyGrid::ChangeGrid( SolidView_Sptr& view )
{
	if (view.get() == 0) // 點錯選項
		return false;
	switch (view->GetType())
	{
	case SEffect::BOUNDING_BOX:
		{
			ConvertTo_BoundingBox();
			return true;
		}
		break;
	case SEffect::VERTEX:
		{
			ConvertTo_Vertex();
			return true;
		}
		break;
	case SEffect::CONTOUR:
		{
			ConvertTo_Contour();
			const Contour_Setting* seffect = (Contour_Setting*)(view->GetEffect().get());
			wxString setvalue;
			setvalue << seffect->m_ContourValue;
			SetCellValue(0, 0, setvalue);
			setvalue.Clear();
			setvalue << seffect->m_alpha;
			SetCellValue(1, 0, setvalue);
			return true;
		}
		break;
	case SEffect::AXES:
		{
			ConvertTo_Axes();
			return true;
		}
		break;
	case SEffect::CLIP_PLANE:
		{
			ConvertTo_ClipPlane();
			const ClipPlane_Setting* seffect = (ClipPlane_Setting*)(view->GetEffect().get());
			switch (seffect->m_Axes)
			{
			case 0:
				SetCellValue(0, 0, _T("X Axes"));
				break;
			case 1:
				SetCellValue(0, 0, _T("Y Axes"));
				break;
			case 2:
				SetCellValue(0, 0, _T("Z Axes"));
				break;
			}
			wxString setvalue;
			setvalue << seffect->m_Percent;
			SetCellValue(1, 0, setvalue);
			return true;
		}
	case SEffect::RULER:
		{
			ConvertTo_Ruler();
			return true;
		}
		break;
	case SEffect::CLIP_CONTOUR:
		{
			ConvertTo_ClipContour();
			return true;
		}
		break;
	case SEffect::VOLUME_RENDERING:
		{
			ConvertTo_VolumeRender();
			return true;
		}
		break;
	}
	return false;
}


