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
#include "stdwx.h"
// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

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
    EVT_GRID_CELL_LEFT_CLICK( MyGrid::OnCellLeftClick )
    EVT_GRID_CELL_RIGHT_CLICK( MyGrid::OnCellRightClick )

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


/*
 * wxEVT_GRID_CELL_LEFT_CLICK event handler for ID_GRID
 */

void MyGrid::OnCellLeftClick( wxGridEvent& event )
{
////@begin wxEVT_GRID_CELL_LEFT_CLICK event handler for ID_GRID in MyGrid.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_GRID_CELL_LEFT_CLICK event handler for ID_GRID in MyGrid. 
}


/*
 * wxEVT_GRID_CELL_RIGHT_CLICK event handler for ID_GRID
 */

void MyGrid::OnCellRightClick( wxGridEvent& event )
{
////@begin wxEVT_GRID_CELL_RIGHT_CLICK event handler for ID_GRID in MyGrid.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_GRID_CELL_RIGHT_CLICK event handler for ID_GRID in MyGrid. 
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
	ReCreateGrid(1,3);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("MaxValue"));		// 大於這個值才顯示
	SetRowLabelValue(i++, wxT("MinValue"));		// 小於這個值才顯示
	SetRowLabelValue(i++, wxT("Size"));		// 點的顯示大小
}

void MyGrid::ConvertTo_IsosurfaceContour()
{
	ReCreateGrid(1,2);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("ContourValue"));	// 要做出切面的值
	SetRowLabelValue(i++, wxT("Alpha"));		// 切面透明度的值
}

void MyGrid::ConvertTo_Axes()
{
	ReCreateGrid(1,4);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("ThickDegree"));	// 軸向的粗細
	SetRowLabelValue(i++, wxT("XColor"));		// X軸向的顏色
	SetRowLabelValue(i++, wxT("YColor"));		// Z軸向的顏色
	SetRowLabelValue(i++, wxT("ZColor"));		// Z軸向的顏色
}

void MyGrid::ConvertTo_Ruler()
{
	ReCreateGrid(1,7);
	SetColLabelValue(0, wxT("Value"));
	int i=0;
	SetRowLabelValue(i++, wxT("Target"));		// 量尺的對象
	SetRowLabelValue(i++, wxT("TargetAxes"));		// 量尺的軸向
	SetRowLabelValue(i++, wxT("StartPoint"));		// 起始點，調整後對象與軸向選項失效
	SetRowLabelValue(i++, wxT("EndPoint"));		// 結束點，調整後對象與軸向選項失效
	SetRowLabelValue(i++, wxT("Scalar"));		// 量尺的突出程度
	SetRowLabelValue(i++, wxT("ThickDegree"));	// 量尺的粗細
	SetRowLabelValue(i++, wxT("Color"));		// 量尺的顏色
}

void MyGrid::ConvertTo_PlaneChip()
{
	ReCreateGrid(1,2);
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
	SetCellValue(i, 0, choices[0]);
	SetCellValue(i++, 0, wxT("0"));
	SetRowLabelValue(i++, wxT("Percent"));
}

void MyGrid::ConvertTo_ContourChip()
{
	ReCreateGrid(1,3);
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

void MyGrid::AppendGrid( int Cols, int Rows )
{
	while (Cols--)
		AppendCols();
	while (Rows--)
		AppendRows();
}

void MyGrid::ReCreateGrid( int Cols, int Rows )
{
	DeleteGrid();
	AppendGrid( Cols, Rows );
}
