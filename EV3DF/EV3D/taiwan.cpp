/////////////////////////////////////////////////////////////////////////////
// Name:        taiwan.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     20/04/2011 02:44:01
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

#include "taiwan.h"

////@begin XPM images
////@end XPM images


/*
 * Taiwan type definition
 */

IMPLEMENT_CLASS( Taiwan, wxFrame )


/*
 * Taiwan event table definition
 */

BEGIN_EVENT_TABLE( Taiwan, wxFrame )

////@begin Taiwan event table entries
////@end Taiwan event table entries

END_EVENT_TABLE()


/*
 * Taiwan constructors
 */

Taiwan::Taiwan()
{
    Init();
}

Taiwan::Taiwan( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
    Init();
    Create( parent, id, caption, pos, size, style );
}


/*
 * Taiwan creator
 */

bool Taiwan::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
////@begin Taiwan creation
    wxFrame::Create( parent, id, caption, pos, size, style );

    CreateControls();
    Centre();
////@end Taiwan creation
    return true;
}


/*
 * Taiwan destructor
 */

Taiwan::~Taiwan()
{
////@begin Taiwan destruction
////@end Taiwan destruction
}


/*
 * Member initialisation
 */

void Taiwan::Init()
{
////@begin Taiwan member initialisation
////@end Taiwan member initialisation
}


/*
 * Control creation for Taiwan
 */

void Taiwan::CreateControls()
{    
////@begin Taiwan content construction
    Taiwan* itemFrame1 = this;

    wxMenuBar* menuBar = new wxMenuBar;
    wxMenu* itemMenu37 = new wxMenu;
    itemMenu37->Append(ID_MENUITEM, wxGetTranslation(wxString() + (wxChar) 0x958B + (wxChar) 0x59CB + (wxChar) 0x820A + (wxChar) 0x6A94), wxEmptyString, wxITEM_NORMAL);
    itemMenu37->Append(ID_MENUITEM1, wxGetTranslation(wxString() + (wxChar) 0x53E6 + (wxChar) 0x5B58 + (wxChar) 0x65B0 + (wxChar) 0x6A94), wxEmptyString, wxITEM_NORMAL);
    itemMenu37->Append(ID_MENUITEM2, wxGetTranslation(wxString() + (wxChar) 0x5132 + (wxChar) 0x5B58 + (wxChar) 0x6A94 + (wxChar) 0x6848), wxEmptyString, wxITEM_NORMAL);
    itemMenu37->Append(ID_MENUITEM3, wxGetTranslation(wxString() + (wxChar) 0x96E2 + (wxChar) 0x958B), wxEmptyString, wxITEM_NORMAL);
    menuBar->Append(itemMenu37, wxGetTranslation(wxString() + (wxChar) 0x6A94 + (wxChar) 0x6848));
    itemFrame1->SetMenuBar(menuBar);

    wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxHORIZONTAL);
    itemFrame1->SetSizer(itemBoxSizer2);

    wxGridBagSizer* itemGridBagSizer3 = new wxGridBagSizer(0, 0);
    itemGridBagSizer3->SetEmptyCellSize(wxSize(10, 20));
    itemBoxSizer2->Add(itemGridBagSizer3, 0, wxGROW|wxALL, 5);

    wxButton* itemButton4 = new wxButton( itemFrame1, ID_BUTTON, wxGetTranslation(wxString() + (wxChar) 0x9078 + (wxChar) 0x53D6 + (wxChar) 0x7BC4 + (wxChar) 0x570D), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton4, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton5 = new wxButton( itemFrame1, ID_BUTTON1, wxGetTranslation(wxString() + (wxChar) 0x8F38 + (wxChar) 0x51FA + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton5, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton6 = new wxButton( itemFrame1, ID_BUTTON3, wxGetTranslation(wxString() + (wxChar) 0x653E + (wxChar) 0x5927 + (wxChar) 0x93E1), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton6, wxGBPosition(2, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton7 = new wxButton( itemFrame1, ID_BUTTON5, wxGetTranslation(wxString() + (wxChar) 0x8A08 + (wxChar) 0x7B97 + (wxChar) 0x7BC4 + (wxChar) 0x570D + (wxChar) 0x5730 + (wxChar) 0x71B1), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton7, wxGBPosition(4, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText8 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x9078 + (wxChar) 0x64C7 + (wxChar) 0x7269 + (wxChar) 0x7406 + (wxChar) 0x91CF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText8, wxGBPosition(3, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton9 = new wxButton( itemFrame1, ID_BUTTON6, wxGetTranslation(wxString() + (wxChar) 0x8F38 + (wxChar) 0x5165 + (wxChar) 0x7CBE + (wxChar) 0x5BC6 + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton9, wxGBPosition(5, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton10 = new wxButton( itemFrame1, ID_BUTTON7, wxGetTranslation(wxString() + (wxChar) 0x79FB + (wxChar) 0x9664 + (wxChar) 0x7CBE + (wxChar) 0x5BC6 + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton10, wxGBPosition(6, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton11 = new wxButton( itemFrame1, ID_BUTTON8, wxGetTranslation(wxString() + (wxChar) 0x4FEE + (wxChar) 0x6539 + (wxChar) 0x5167 + (wxChar) 0x5EFA + (wxChar) 0x4E95 + (wxChar) 0x6E2C + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton11, wxGBPosition(18, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton12 = new wxButton( itemFrame1, ID_BUTTON9, wxGetTranslation(wxString() + (wxChar) 0x53E6 + (wxChar) 0x5B58 + (wxChar) 0x4E95 + (wxChar) 0x6E2C + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton12, wxGBPosition(7, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText13 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString(wxT("Maxmum EW,X,lon,")) + (wxChar) 0x7D93), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText13, wxGBPosition(10, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText14 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6ED1 + (wxChar) 0x9F20 + wxT(" EW,X,lon,") + (wxChar) 0x7D93), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText14, wxGBPosition(8, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl15 = new wxTextCtrl( itemFrame1, ID_TEXTCTRL18, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemTextCtrl15, wxGBPosition(10, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl16 = new wxTextCtrl( itemFrame1, ID_TEXTCTRL19, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemTextCtrl16, wxGBPosition(11, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl17 = new wxTextCtrl( itemFrame1, ID_TEXTCTRL20, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemTextCtrl17, wxGBPosition(12, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl18 = new wxTextCtrl( itemFrame1, ID_TEXTCTRL21, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemTextCtrl18, wxGBPosition(13, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl19 = new wxTextCtrl( itemFrame1, ID_TEXTCTRL22, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemTextCtrl19, wxGBPosition(14, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl20 = new wxTextCtrl( itemFrame1, ID_TEXTCTRL23, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemTextCtrl20, wxGBPosition(15, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText21 = new wxStaticText( itemFrame1, wxID_STATIC, _("Maxmum Y,Height"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText21, wxGBPosition(14, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText22 = new wxStaticText( itemFrame1, wxID_STATIC, _("Minmum Y,Height"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText22, wxGBPosition(15, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText23 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString(wxT("Maxmum NS,Y,lat,")) + (wxChar) 0x7DEF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText23, wxGBPosition(12, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText24 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString(wxT("Minmum NS,Y,lat,")) + (wxChar) 0x7DEF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText24, wxGBPosition(13, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText25 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString(wxT("Minmum EW,X,lon,")) + (wxChar) 0x7D93), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText25, wxGBPosition(11, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton26 = new wxButton( itemFrame1, ID_BUTTON10, wxGetTranslation(wxString() + (wxChar) 0x986F + (wxChar) 0x793A + (wxChar) 0x7BC4 + (wxChar) 0x570D + (wxChar) 0x5167 + (wxChar) 0x63D2 + (wxChar) 0x7D50 + (wxChar) 0x679C), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemButton26, wxGBPosition(19, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText27 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6BD4 + (wxChar) 0x4F8B + (wxChar) 0x5C3A), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText27, wxGBPosition(16, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxComboCtrl* itemComboCtrl28 = new wxComboCtrl( itemFrame1, ID_COMBOCTRL, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemComboCtrl28, wxGBPosition(3, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText29 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6ED1 + (wxChar) 0x9F20 + wxT(" NS,Y,lat,") + (wxChar) 0x7DEF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText29, wxGBPosition(9, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxSlider* itemSlider30 = new wxSlider( itemFrame1, ID_SLIDER, 0, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
    itemGridBagSizer3->Add(itemSlider30, wxGBPosition(16, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText31 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString(wxT("10 pixpel ")) + (wxChar) 0x6BD4 + wxT(" 300 ") + (wxChar) 0x516C + (wxChar) 0x5C3A), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer3->Add(itemStaticText31, wxGBPosition(17, 0), wxGBSpan(1, 2), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxFlexGridSizer* itemFlexGridSizer32 = new wxFlexGridSizer(0, 2, 0, 0);
    itemBoxSizer2->Add(itemFlexGridSizer32, 1, wxGROW|wxALL, 0);

    wxGLCanvas* itemGLCanvas33 = new wxGLCanvas( itemFrame1, ID_GLCANVAS1, wxDefaultPosition, wxSize(880, 720), 0 );
    itemFlexGridSizer32->Add(itemGLCanvas33, 1, wxGROW|wxGROW|wxALL, 0);

    wxScrollBar* itemScrollBar34 = new wxScrollBar( itemFrame1, ID_SCROLLBAR, wxDefaultPosition, wxDefaultSize, wxSB_VERTICAL );
    itemScrollBar34->SetScrollbar(0, 1, 100, 1);
    itemFlexGridSizer32->Add(itemScrollBar34, 1, wxGROW|wxGROW|wxALL, 0);

    wxScrollBar* itemScrollBar35 = new wxScrollBar( itemFrame1, ID_SCROLLBAR1, wxDefaultPosition, wxDefaultSize, wxSB_HORIZONTAL );
    itemScrollBar35->SetScrollbar(0, 1, 100, 1);
    itemFlexGridSizer32->Add(itemScrollBar35, 1, wxGROW|wxGROW|wxALL, 0);

////@end Taiwan content construction
}


/*
 * Should we show tooltips?
 */

bool Taiwan::ShowToolTips()
{
    return true;
}

/*
 * Get bitmap resources
 */

wxBitmap Taiwan::GetBitmapResource( const wxString& name )
{
    // Bitmap retrieval
////@begin Taiwan bitmap retrieval
    wxUnusedVar(name);
    return wxNullBitmap;
////@end Taiwan bitmap retrieval
}

/*
 * Get icon resources
 */

wxIcon Taiwan::GetIconResource( const wxString& name )
{
    // Icon retrieval
////@begin Taiwan icon retrieval
    wxUnusedVar(name);
    return wxNullIcon;
////@end Taiwan icon retrieval
}
