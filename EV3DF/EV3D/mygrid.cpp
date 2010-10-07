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
	SetColLabelValue(0, wxT("None"));
	SetRowLabelValue(0, wxT("None"));
}

void MyGrid::ConvertTo_Vetex()
{
	ReCreateGrid(1,2);
	SetColLabelValue(0, wxT("Value"));
	SetRowLabelValue(0, wxT("Start"));
	SetRowLabelValue(1, wxT("End"));
}

void MyGrid::ConvertTo_IsosurfaceContour()
{
	ReCreateGrid(1,1);
	SetColLabelValue(0, wxT("Value"));
	SetRowLabelValue(0, wxT("ContourValue"));
}

void MyGrid::ConvertTo_Axes()
{
	ReCreateGrid(1,1);
	SetColLabelValue(0, wxT("None"));
	SetRowLabelValue(0, wxT("None"));
}

void MyGrid::ConvertTo_PlaneChip()
{
	ReCreateGrid(1,1);
	SetColLabelValue(0, wxT("None"));
	SetRowLabelValue(0, wxT("None"));
}

void MyGrid::ConvertTo_ContourChip()
{
	ReCreateGrid(1,1);
	SetColLabelValue(0, wxT("None"));
	SetRowLabelValue(0, wxT("None"));
}

void MyGrid::ConvertTo_VolumeRender()
{
// 	ClearGrid();
// 	CreateGrid(1, 1, wxGrid::wxGridSelectCells);
	SetColLabelValue(0, wxT("None"));
	SetRowLabelValue(0, wxT("None"));
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
