/////////////////////////////////////////////////////////////////////////////
// Name:        convertdialog.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     24/02/2011 06:38:39
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

#include "convertdialog.h"

////@begin XPM images
////@end XPM images


/*
 * ConvertDialog type definition
 */

IMPLEMENT_DYNAMIC_CLASS( ConvertDialog, wxDialog )


/*
 * ConvertDialog event table definition
 */

BEGIN_EVENT_TABLE( ConvertDialog, wxDialog )

////@begin ConvertDialog event table entries
////@end ConvertDialog event table entries

END_EVENT_TABLE()


/*
 * ConvertDialog constructors
 */

ConvertDialog::ConvertDialog()
{
    Init();
}

ConvertDialog::ConvertDialog( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
    Init();
    Create(parent, id, caption, pos, size, style);
}


/*
 * ConvertDialog creator
 */

bool ConvertDialog::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
////@begin ConvertDialog creation
    SetExtraStyle(wxWS_EX_BLOCK_EVENTS);
    wxDialog::Create( parent, id, caption, pos, size, style );

    CreateControls();
    if (GetSizer())
    {
        GetSizer()->SetSizeHints(this);
    }
    Centre();
////@end ConvertDialog creation
    return true;
}


/*
 * ConvertDialog destructor
 */

ConvertDialog::~ConvertDialog()
{
////@begin ConvertDialog destruction
////@end ConvertDialog destruction
}


/*
 * Member initialisation
 */

void ConvertDialog::Init()
{
////@begin ConvertDialog member initialisation
////@end ConvertDialog member initialisation
}


/*
 * Control creation for ConvertDialog
 */

void ConvertDialog::CreateControls()
{    
////@begin ConvertDialog content construction
    ConvertDialog* itemDialog1 = this;

    wxGridBagSizer* itemGridBagSizer2 = new wxGridBagSizer(0, 0);
    itemGridBagSizer2->SetEmptyCellSize(wxSize(10, 20));
    itemDialog1->SetSizer(itemGridBagSizer2);

    wxStaticText* itemStaticText3 = new wxStaticText( itemDialog1, wxID_STATIC, _("xyz\\data field"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText3, wxGBPosition(0, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText4 = new wxStaticText( itemDialog1, wxID_STATIC, _("x"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText4, wxGBPosition(0, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText5 = new wxStaticText( itemDialog1, wxID_STATIC, _("y"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText5, wxGBPosition(0, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText6 = new wxStaticText( itemDialog1, wxID_STATIC, _("z"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText6, wxGBPosition(0, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText7 = new wxStaticText( itemDialog1, wxID_STATIC, _("min"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText7, wxGBPosition(1, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText8 = new wxStaticText( itemDialog1, wxID_STATIC, _("max"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText8, wxGBPosition(2, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText9 = new wxStaticText( itemDialog1, wxID_STATIC, _("input data info"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText9, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText10 = new wxStaticText( itemDialog1, wxID_STATIC, _("output data info"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText10, wxGBPosition(5, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl11 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemTextCtrl11->Enable(false);
    itemGridBagSizer2->Add(itemTextCtrl11, wxGBPosition(1, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl12 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL1, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemTextCtrl12->Enable(false);
    itemGridBagSizer2->Add(itemTextCtrl12, wxGBPosition(1, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl13 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL2, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemTextCtrl13->Enable(false);
    itemGridBagSizer2->Add(itemTextCtrl13, wxGBPosition(1, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl14 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL3, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemTextCtrl14->Enable(false);
    itemGridBagSizer2->Add(itemTextCtrl14, wxGBPosition(2, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl15 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL4, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemTextCtrl15->Enable(false);
    itemGridBagSizer2->Add(itemTextCtrl15, wxGBPosition(2, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl16 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL5, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemTextCtrl16->Enable(false);
    itemGridBagSizer2->Add(itemTextCtrl16, wxGBPosition(2, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl17 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL6, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemTextCtrl17, wxGBPosition(6, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText18 = new wxStaticText( itemDialog1, wxID_STATIC, _("x"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText18, wxGBPosition(5, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText19 = new wxStaticText( itemDialog1, wxID_STATIC, _("y"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText19, wxGBPosition(5, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText20 = new wxStaticText( itemDialog1, wxID_STATIC, _("z"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText20, wxGBPosition(5, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText21 = new wxStaticText( itemDialog1, wxID_STATIC, _("xyz\\data field"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText21, wxGBPosition(5, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText22 = new wxStaticText( itemDialog1, wxID_STATIC, _("min"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText22, wxGBPosition(6, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText23 = new wxStaticText( itemDialog1, wxID_STATIC, _("max"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText23, wxGBPosition(7, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText24 = new wxStaticText( itemDialog1, wxID_STATIC, _("interval"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText24, wxGBPosition(8, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl25 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL7, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemTextCtrl25, wxGBPosition(6, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl26 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL8, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemTextCtrl26, wxGBPosition(6, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl27 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL9, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemTextCtrl27, wxGBPosition(7, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl28 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL10, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemTextCtrl28, wxGBPosition(7, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl29 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL11, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemTextCtrl29, wxGBPosition(7, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl30 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL12, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemTextCtrl30, wxGBPosition(8, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl31 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL13, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemTextCtrl31, wxGBPosition(8, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl32 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL14, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemTextCtrl32, wxGBPosition(8, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton33 = new wxButton( itemDialog1, ID_BUTTON, _("Close"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton33, wxGBPosition(12, 5), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton34 = new wxButton( itemDialog1, ID_BUTTON1, _("Convert"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton34, wxGBPosition(12, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxFilePickerCtrl* itemFilePickerCtrl35 = new wxFilePickerCtrl( itemDialog1, ID_FILECTRL, wxEmptyString, wxEmptyString, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
    itemGridBagSizer2->Add(itemFilePickerCtrl35, wxGBPosition(4, 0), wxGBSpan(1, 5), wxALIGN_CENTER_HORIZONTAL|wxGROW|wxALL, 5);

    wxFilePickerCtrl* itemFilePickerCtrl36 = new wxFilePickerCtrl( itemDialog1, ID_FILEPICKERCTRL, wxEmptyString, wxEmptyString, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
    itemGridBagSizer2->Add(itemFilePickerCtrl36, wxGBPosition(10, 0), wxGBSpan(1, 5), wxALIGN_CENTER_HORIZONTAL|wxGROW|wxALL, 5);

    wxStaticText* itemStaticText37 = new wxStaticText( itemDialog1, wxID_STATIC, _("data total"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText37, wxGBPosition(3, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText38 = new wxStaticText( itemDialog1, wxID_STATIC, _("data total"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText38, wxGBPosition(9, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl39 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL15, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemTextCtrl39->Enable(false);
    itemGridBagSizer2->Add(itemTextCtrl39, wxGBPosition(3, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl40 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL16, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemTextCtrl40->Enable(false);
    itemGridBagSizer2->Add(itemTextCtrl40, wxGBPosition(9, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl41 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL17, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE );
    itemGridBagSizer2->Add(itemTextCtrl41, wxGBPosition(11, 0), wxGBSpan(2, 3), wxALIGN_CENTER_HORIZONTAL|wxGROW|wxALL, 5);

////@end ConvertDialog content construction
}


/*
 * Should we show tooltips?
 */

bool ConvertDialog::ShowToolTips()
{
    return true;
}

/*
 * Get bitmap resources
 */

wxBitmap ConvertDialog::GetBitmapResource( const wxString& name )
{
    // Bitmap retrieval
////@begin ConvertDialog bitmap retrieval
    wxUnusedVar(name);
    return wxNullBitmap;
////@end ConvertDialog bitmap retrieval
}

/*
 * Get icon resources
 */

wxIcon ConvertDialog::GetIconResource( const wxString& name )
{
    // Icon retrieval
////@begin ConvertDialog icon retrieval
    wxUnusedVar(name);
    return wxNullIcon;
////@end ConvertDialog icon retrieval
}
