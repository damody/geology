/////////////////////////////////////////////////////////////////////////////
// Name:        seetaiwan.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     11/04/2011 16:17:51
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

#include "seetaiwan.h"

////@begin XPM images
////@end XPM images


/*
 * SeeTaiwan type definition
 */

IMPLEMENT_CLASS( SeeTaiwan, wxFrame )


/*
 * SeeTaiwan event table definition
 */

BEGIN_EVENT_TABLE( SeeTaiwan, wxFrame )

////@begin SeeTaiwan event table entries
////@end SeeTaiwan event table entries

END_EVENT_TABLE()


/*
 * SeeTaiwan constructors
 */

SeeTaiwan::SeeTaiwan()
{
    Init();
}

SeeTaiwan::SeeTaiwan( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
    Init();
    Create( parent, id, caption, pos, size, style );
}


/*
 * SeeTaiwan creator
 */

bool SeeTaiwan::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
////@begin SeeTaiwan creation
    wxFrame::Create( parent, id, caption, pos, size, style );

    CreateControls();
    Centre();
////@end SeeTaiwan creation
    return true;
}


/*
 * SeeTaiwan destructor
 */

SeeTaiwan::~SeeTaiwan()
{
////@begin SeeTaiwan destruction
////@end SeeTaiwan destruction
}


/*
 * Member initialisation
 */

void SeeTaiwan::Init()
{
////@begin SeeTaiwan member initialisation
////@end SeeTaiwan member initialisation
}


/*
 * Control creation for SeeTaiwan
 */

void SeeTaiwan::CreateControls()
{    
////@begin SeeTaiwan content construction
    SeeTaiwan* itemFrame1 = this;

    wxGridBagSizer* itemGridBagSizer2 = new wxGridBagSizer(0, 0);
    itemGridBagSizer2->SetEmptyCellSize(wxSize(10, 20));
    itemFrame1->SetSizer(itemGridBagSizer2);

    wxButton* itemButton3 = new wxButton( itemFrame1, ID_BUTTON2, wxGetTranslation(wxString() + (wxChar) 0x8F38 + (wxChar) 0x51FA + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton3, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton4 = new wxButton( itemFrame1, ID_BUTTON1, wxGetTranslation(wxString() + (wxChar) 0x9078 + (wxChar) 0x53D6 + (wxChar) 0x7BC4 + (wxChar) 0x570D), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton4, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton5 = new wxButton( itemFrame1, ID_BUTTON3, wxGetTranslation(wxString() + (wxChar) 0x653E + (wxChar) 0x5927 + (wxChar) 0x93E1), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton5, wxGBPosition(2, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton6 = new wxButton( itemFrame1, ID_BUTTON4, wxGetTranslation(wxString() + (wxChar) 0x8F38 + (wxChar) 0x5165 + (wxChar) 0x7CBE + (wxChar) 0x5BC6 + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton6, wxGBPosition(3, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton7 = new wxButton( itemFrame1, ID_BUTTON5, wxGetTranslation(wxString() + (wxChar) 0x79FB + (wxChar) 0x9664 + (wxChar) 0x7CBE + (wxChar) 0x5BC6 + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton7, wxGBPosition(4, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

////@end SeeTaiwan content construction
}


/*
 * Should we show tooltips?
 */

bool SeeTaiwan::ShowToolTips()
{
    return true;
}

/*
 * Get bitmap resources
 */

wxBitmap SeeTaiwan::GetBitmapResource( const wxString& name )
{
    // Bitmap retrieval
////@begin SeeTaiwan bitmap retrieval
    wxUnusedVar(name);
    return wxNullBitmap;
////@end SeeTaiwan bitmap retrieval
}

/*
 * Get icon resources
 */

wxIcon SeeTaiwan::GetIconResource( const wxString& name )
{
    // Icon retrieval
////@begin SeeTaiwan icon retrieval
    wxUnusedVar(name);
    return wxNullIcon;
////@end SeeTaiwan icon retrieval
}
