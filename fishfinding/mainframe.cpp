/////////////////////////////////////////////////////////////////////////////
// Name:        mainframe.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     19/11/2010 20:44:33
// RCS-ID:      
// Copyright:   ntust
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

#include "mainframe.h"

////@begin XPM images
////@end XPM images


/*
 * mainframe type definition
 */

IMPLEMENT_CLASS( mainframe, wxFrame )


/*
 * mainframe event table definition
 */

BEGIN_EVENT_TABLE( mainframe, wxFrame )

////@begin mainframe event table entries
////@end mainframe event table entries

END_EVENT_TABLE()


/*
 * mainframe constructors
 */

mainframe::mainframe()
{
    Init();
}

mainframe::mainframe( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
    Init();
    Create( parent, id, caption, pos, size, style );
}


/*
 * mainframe creator
 */

bool mainframe::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
////@begin mainframe creation
    wxFrame::Create( parent, id, caption, pos, size, style );

    CreateControls();
    Centre();
////@end mainframe creation
    return true;
}


/*
 * mainframe destructor
 */

mainframe::~mainframe()
{
////@begin mainframe destruction
////@end mainframe destruction
}


/*
 * Member initialisation
 */

void mainframe::Init()
{
////@begin mainframe member initialisation
////@end mainframe member initialisation
}


/*
 * Control creation for mainframe
 */

void mainframe::CreateControls()
{    
////@begin mainframe content construction
    mainframe* itemFrame1 = this;

    wxMenuBar* menuBar = new wxMenuBar;
    wxMenu* itemMenu24 = new wxMenu;
    menuBar->Append(itemMenu24, _("File"));
    itemFrame1->SetMenuBar(menuBar);

    wxGridBagSizer* itemGridBagSizer2 = new wxGridBagSizer(0, 0);
    itemGridBagSizer2->SetEmptyCellSize(wxSize(10, 20));
    itemFrame1->SetSizer(itemGridBagSizer2);

    wxArrayString itemChoice3Strings;
    itemChoice3Strings.Add(_("110"));
    itemChoice3Strings.Add(_("300"));
    itemChoice3Strings.Add(_("600"));
    itemChoice3Strings.Add(_("1200"));
    itemChoice3Strings.Add(_("2400"));
    itemChoice3Strings.Add(_("4800"));
    itemChoice3Strings.Add(_("9600"));
    itemChoice3Strings.Add(_("19200"));
    itemChoice3Strings.Add(_("38400"));
    itemChoice3Strings.Add(_("57600"));
    itemChoice3Strings.Add(_("115200"));
    itemChoice3Strings.Add(_("128000"));
    itemChoice3Strings.Add(_("256000"));
    wxChoice* itemChoice3 = new wxChoice( itemFrame1, ID_CHOICE1, wxDefaultPosition, wxDefaultSize, itemChoice3Strings, 0 );
    itemChoice3->SetStringSelection(_("38400"));
    itemGridBagSizer2->Add(itemChoice3, wxGBPosition(0, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxArrayString itemChoice4Strings;
    itemChoice4Strings.Add(_("1"));
    itemChoice4Strings.Add(_("2"));
    itemChoice4Strings.Add(_("3"));
    itemChoice4Strings.Add(_("4"));
    itemChoice4Strings.Add(_("5"));
    itemChoice4Strings.Add(_("6"));
    itemChoice4Strings.Add(_("7"));
    itemChoice4Strings.Add(_("8"));
    itemChoice4Strings.Add(_("9"));
    itemChoice4Strings.Add(_("10"));
    itemChoice4Strings.Add(_("11"));
    itemChoice4Strings.Add(_("12"));
    itemChoice4Strings.Add(_("13"));
    itemChoice4Strings.Add(_("14"));
    itemChoice4Strings.Add(_("15"));
    wxChoice* itemChoice4 = new wxChoice( itemFrame1, ID_CHOICE, wxDefaultPosition, wxDefaultSize, itemChoice4Strings, 0 );
    itemChoice4->SetStringSelection(_("1"));
    itemGridBagSizer2->Add(itemChoice4, wxGBPosition(1, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText5 = new wxStaticText( itemFrame1, wxID_STATIC, _("BoundRate"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText5, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText6 = new wxStaticText( itemFrame1, wxID_STATIC, _("ComPort"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText6, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton7 = new wxButton( itemFrame1, ID_BUTTON, _("StartGet"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton7, wxGBPosition(2, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl8 = new wxTextCtrl( itemFrame1, ID_TEXTCTRL, wxEmptyString, wxDefaultPosition, wxSize(600, 100), wxTE_MULTILINE );
    itemGridBagSizer2->Add(itemTextCtrl8, wxGBPosition(14, 0), wxGBSpan(2, 3), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton9 = new wxButton( itemFrame1, ID_BUTTON1, _("StopGet"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton9, wxGBPosition(2, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxFilePickerCtrl* itemFilePickerCtrl10 = new wxFilePickerCtrl( itemFrame1, ID_FILECTRL, wxEmptyString, wxEmptyString, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
    itemGridBagSizer2->Add(itemFilePickerCtrl10, wxGBPosition(12, 0), wxGBSpan(1, 2), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText11 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x8CC7 + (wxChar) 0x6599 + (wxChar) 0x500B + (wxChar) 0x6578), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText11, wxGBPosition(3, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText12 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x7D93 + (wxChar) 0x5EA6 + wxT("E")), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText12, wxGBPosition(5, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText13 = new wxStaticText( itemFrame1, wxID_STATIC, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText13, wxGBPosition(4, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText14 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6C34 + (wxChar) 0x6DF1), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText14, wxGBPosition(4, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText15 = new wxStaticText( itemFrame1, wxID_STATIC, _("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText15, wxGBPosition(5, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText16 = new wxStaticText( itemFrame1, wxID_STATIC, _("0kb"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText16, wxGBPosition(10, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText17 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x7DEF + (wxChar) 0x5EA6 + wxT("N")), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText17, wxGBPosition(6, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText18 = new wxStaticText( itemFrame1, wxID_STATIC, _("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText18, wxGBPosition(6, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText19 = new wxStaticText( itemFrame1, wxID_STATIC, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText19, wxGBPosition(3, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText20 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6A94 + (wxChar) 0x6848 + (wxChar) 0x8DEF + (wxChar) 0x5F91), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText20, wxGBPosition(11, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText21 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6A94 + (wxChar) 0x6848 + (wxChar) 0x5927 + (wxChar) 0x5C0F), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText21, wxGBPosition(10, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxGLCanvas* itemGLCanvas22 = new wxGLCanvas( itemFrame1, ID_GLCANVAS, wxDefaultPosition, wxSize(400, 400), 0 );
    itemGridBagSizer2->Add(itemGLCanvas22, wxGBPosition(0, 2), wxGBSpan(14, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStatusBar* itemStatusBar25 = new wxStatusBar( itemFrame1, ID_STATUSBAR, wxST_SIZEGRIP|wxNO_BORDER );
    itemStatusBar25->SetFieldsCount(2);
    itemFrame1->SetStatusBar(itemStatusBar25);

////@end mainframe content construction
}


/*
 * Should we show tooltips?
 */

bool mainframe::ShowToolTips()
{
    return true;
}

/*
 * Get bitmap resources
 */

wxBitmap mainframe::GetBitmapResource( const wxString& name )
{
    // Bitmap retrieval
////@begin mainframe bitmap retrieval
    wxUnusedVar(name);
    return wxNullBitmap;
////@end mainframe bitmap retrieval
}

/*
 * Get icon resources
 */

wxIcon mainframe::GetIconResource( const wxString& name )
{
    // Icon retrieval
////@begin mainframe icon retrieval
    wxUnusedVar(name);
    return wxNullIcon;
////@end mainframe icon retrieval
}
