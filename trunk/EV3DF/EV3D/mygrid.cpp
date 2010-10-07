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
////@begin MyGrid content construction
////@end MyGrid content construction
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



