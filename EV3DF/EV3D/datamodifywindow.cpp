/////////////////////////////////////////////////////////////////////////////
// Name:        datamodifywindow.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     20/04/2011 03:26:56
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

#include "datamodifywindow.h"

////@begin XPM images
////@end XPM images


/*
 * DataModifyWindow type definition
 */

IMPLEMENT_CLASS( DataModifyWindow, wxFrame )


/*
 * DataModifyWindow event table definition
 */

BEGIN_EVENT_TABLE( DataModifyWindow, wxFrame )

////@begin DataModifyWindow event table entries
////@end DataModifyWindow event table entries

END_EVENT_TABLE()


/*
 * DataModifyWindow constructors
 */

DataModifyWindow::DataModifyWindow()
{
    Init();
}

DataModifyWindow::DataModifyWindow( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
    Init();
    Create( parent, id, caption, pos, size, style );
}


/*
 * DataModifyWindow creator
 */

bool DataModifyWindow::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
////@begin DataModifyWindow creation
    wxFrame::Create( parent, id, caption, pos, size, style );

    CreateControls();
    Centre();
////@end DataModifyWindow creation
    return true;
}


/*
 * DataModifyWindow destructor
 */

DataModifyWindow::~DataModifyWindow()
{
////@begin DataModifyWindow destruction
////@end DataModifyWindow destruction
}


/*
 * Member initialisation
 */

void DataModifyWindow::Init()
{
////@begin DataModifyWindow member initialisation
////@end DataModifyWindow member initialisation
}


/*
 * Control creation for DataModifyWindow
 */

void DataModifyWindow::CreateControls()
{    
////@begin DataModifyWindow content construction
    DataModifyWindow* itemFrame1 = this;

    wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxHORIZONTAL);
    itemFrame1->SetSizer(itemBoxSizer2);

    wxGridBagSizer* itemGridBagSizer3 = new wxGridBagSizer(0, 0);
    itemGridBagSizer3->SetEmptyCellSize(wxSize(10, 20));
    itemBoxSizer2->Add(itemGridBagSizer3, 0, wxGROW|wxALL, 5);

    wxButton* itemButton4 = new wxButton( itemFrame1, ID_BUTTON4, wxGetTranslation(wxString() + (wxChar) 0x78BA + (wxChar) 0x5B9A), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton4, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton5 = new wxButton( itemFrame1, ID_BUTTON11, wxGetTranslation(wxString() + (wxChar) 0x53D6 + (wxChar) 0x6D88), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton5, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxGrid* itemGrid6 = new wxGrid( itemFrame1, ID_GRID1, wxDefaultPosition, wxSize(200, 150), wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL );
    itemGrid6->SetDefaultColSize(50);
    itemGrid6->SetDefaultRowSize(25);
    itemGrid6->SetColLabelSize(25);
    itemGrid6->SetRowLabelSize(50);
    itemGrid6->CreateGrid(5, 5, wxGrid::wxGridSelectCells);
    itemBoxSizer2->Add(itemGrid6, 1, wxGROW|wxALL, 5);

////@end DataModifyWindow content construction
}


/*
 * Should we show tooltips?
 */

bool DataModifyWindow::ShowToolTips()
{
    return true;
}

/*
 * Get bitmap resources
 */

wxBitmap DataModifyWindow::GetBitmapResource( const wxString& name )
{
    // Bitmap retrieval
////@begin DataModifyWindow bitmap retrieval
    wxUnusedVar(name);
    return wxNullBitmap;
////@end DataModifyWindow bitmap retrieval
}

/*
 * Get icon resources
 */

wxIcon DataModifyWindow::GetIconResource( const wxString& name )
{
    // Icon retrieval
////@begin DataModifyWindow icon retrieval
    wxUnusedVar(name);
    return wxNullIcon;
////@end DataModifyWindow icon retrieval
}
