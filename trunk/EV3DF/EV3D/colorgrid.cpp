/////////////////////////////////////////////////////////////////////////////
// Name:        colorgrid.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     19/09/2011 17:09:25
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

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

#include "colorgrid.h"

////@begin XPM images
////@end XPM images


/*
 * ColorGrid type definition
 */

IMPLEMENT_DYNAMIC_CLASS( ColorGrid, wxGrid )


/*
 * ColorGrid event table definition
 */

BEGIN_EVENT_TABLE( ColorGrid, wxGrid )

////@begin ColorGrid event table entries
////@end ColorGrid event table entries

END_EVENT_TABLE()


/*
 * ColorGrid constructors
 */

ColorGrid::ColorGrid()
{
    Init();
}

ColorGrid::ColorGrid(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
    Init();
    Create(parent, id, pos, size, style);
}


/*
 * ColorGrid creator
 */

bool ColorGrid::Create(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
////@begin ColorGrid creation
    wxGrid::Create(parent, id, pos, size, style);
    CreateControls();
////@end ColorGrid creation
    return true;
}


/*
 * ColorGrid destructor
 */

ColorGrid::~ColorGrid()
{
////@begin ColorGrid destruction
////@end ColorGrid destruction
}


/*
 * Member initialisation
 */

void ColorGrid::Init()
{
////@begin ColorGrid member initialisation
////@end ColorGrid member initialisation
}


/*
 * Control creation for ColorGrid
 */

void ColorGrid::CreateControls()
{    
////@begin ColorGrid content construction
////@end ColorGrid content construction
}


/*
 * Should we show tooltips?
 */

bool ColorGrid::ShowToolTips()
{
    return true;
}

/*
 * Get bitmap resources
 */

wxBitmap ColorGrid::GetBitmapResource( const wxString& name )
{
    // Bitmap retrieval
////@begin ColorGrid bitmap retrieval
    wxUnusedVar(name);
    return wxNullBitmap;
////@end ColorGrid bitmap retrieval
}

/*
 * Get icon resources
 */

wxIcon ColorGrid::GetIconResource( const wxString& name )
{
    // Icon retrieval
////@begin ColorGrid icon retrieval
    wxUnusedVar(name);
    return wxNullIcon;
////@end ColorGrid icon retrieval
}
