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
#include "stdwx.h"
#include "DW/SolidDefine.h"
#include "DW/SEffect.h"
// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include "firstmain.h"
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
#include "fileopen.xpm"
#include "filesave.xpm"
#include "find.xpm"
#include "findrepl.xpm"
#include "home.xpm"
#include "back.xpm"
#include "up.xpm"
#include "down.xpm"
#include "forward.xpm"
////@end XPM images
VOID CALLBACK TimerProc ( HWND hParent, UINT uMsg, UINT uEventID, DWORD dwTimer );

/*
* Taiwan type definition
*/

IMPLEMENT_CLASS( Taiwan, wxFrame )


/*
* Taiwan event table definition
*/

BEGIN_EVENT_TABLE( Taiwan, wxFrame )

////@begin Taiwan event table entries
    EVT_BUTTON( ID_BUTTON8, Taiwan::OnModifyData )

    EVT_BUTTON( ID_BUTTON10, Taiwan::OnShowResult )

    EVT_CHECKBOX( ID_CHECKBOX, Taiwan::OnShowInfo )

    EVT_CHECKBOX( ID_CHECKBOX3, Taiwan::OnCheckShowWellInfo )

    EVT_CHECKBOX( ID_CHECKBOX1, Taiwan::OnCheckboxAxis_Sync )

    EVT_CHECKBOX( ID_CHECKBOX2, Taiwan::OnCheckboxDepth_Sync )

    EVT_SCROLLBAR( ID_SCROLLBAR, Taiwan::OnScrollbarVLUpdated )

    EVT_SCROLLBAR( ID_SCROLLBAR5, Taiwan::OnScrollbarRLUpdated )

    EVT_SCROLLBAR( ID_SCROLLBAR3, Taiwan::OnScrollbarHL1Updated )

    EVT_SCROLLBAR( ID_SCROLLBAR4, Taiwan::OnScrollbarHR1Updated )

    EVT_SCROLLBAR( ID_SCROLLBAR1, Taiwan::OnScrollbarHL2Updated )

    EVT_SCROLLBAR( ID_SCROLLBAR2, Taiwan::OnScrollbarHR2Updated )

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
    GetAuiManager().UnInit();
	////@end Taiwan destruction
}


/*
* Member initialisation
*/

void Taiwan::Init()
{
	////@begin Taiwan member initialisation
    m_ruler_slider = NULL;
    m_ruler_spinctrl = NULL;
    m_Checkbox_Axis_Sync = NULL;
    m_Checkbox_Depth_Sync = NULL;
    m_CanvasL = NULL;
    m_ScrollbarVL = NULL;
    m_CanvasR = NULL;
    m_ScrollbarRL = NULL;
    m_ScrollbarHL1 = NULL;
    m_ScrollbarHR1 = NULL;
    m_ScrollbarHL2 = NULL;
    m_ScrollbarHR2 = NULL;
    m_area_infopanel = NULL;
    m_well_infopanel = NULL;
	////@end Taiwan member initialisation
}


/*
* Control creation for Taiwan
*/

void Taiwan::CreateControls()
{    
	////@begin Taiwan content construction
    Taiwan* itemFrame1 = this;

    GetAuiManager().SetManagedWindow(this);

    wxMenuBar* menuBar = new wxMenuBar;
    wxMenu* itemMenu3 = new wxMenu;
    itemMenu3->Append(ID_MENUITEM, wxGetTranslation(wxString() + (wxChar) 0x958B + (wxChar) 0x59CB + (wxChar) 0x820A + (wxChar) 0x6A94), wxEmptyString, wxITEM_NORMAL);
    itemMenu3->Append(ID_MENUITEM1, wxGetTranslation(wxString() + (wxChar) 0x53E6 + (wxChar) 0x5B58 + (wxChar) 0x65B0 + (wxChar) 0x6A94), wxEmptyString, wxITEM_NORMAL);
    itemMenu3->Append(ID_MENUITEM2, wxGetTranslation(wxString() + (wxChar) 0x5132 + (wxChar) 0x5B58 + (wxChar) 0x6A94 + (wxChar) 0x6848), wxEmptyString, wxITEM_NORMAL);
    itemMenu3->Append(ID_MENUITEM3, wxGetTranslation(wxString() + (wxChar) 0x96E2 + (wxChar) 0x958B), wxEmptyString, wxITEM_NORMAL);
    menuBar->Append(itemMenu3, wxGetTranslation(wxString() + (wxChar) 0x6A94 + (wxChar) 0x6848));
    wxMenu* itemMenu8 = new wxMenu;
    itemMenu8->Append(ID_MENUITEM4, wxGetTranslation(wxString() + (wxChar) 0x57FA + (wxChar) 0x672C + (wxChar) 0x64CD + (wxChar) 0x4F5C), wxEmptyString, wxITEM_NORMAL);
    itemMenu8->Append(ID_MENUITEM5, wxGetTranslation(wxString() + (wxChar) 0x8173 + (wxChar) 0x672C + (wxChar) 0x8A9E + (wxChar) 0x6CD5), wxEmptyString, wxITEM_NORMAL);
    menuBar->Append(itemMenu8, wxGetTranslation(wxString() + (wxChar) 0x8AAA + (wxChar) 0x660E));
    wxMenu* itemMenu11 = new wxMenu;
    itemMenu11->Append(ID_MENUITEM7, wxGetTranslation(wxString() + (wxChar) 0x570B + (wxChar) 0x7ACB + (wxChar) 0x6D77 + (wxChar) 0x6D0B + (wxChar) 0x5927 + (wxChar) 0x5B78), wxEmptyString, wxITEM_NORMAL);
    itemMenu11->Append(ID_MENUITEM6, wxGetTranslation(wxString() + (wxChar) 0x570B + (wxChar) 0x7ACB + (wxChar) 0x53F0 + (wxChar) 0x7063 + (wxChar) 0x79D1 + (wxChar) 0x6280 + (wxChar) 0x5927 + (wxChar) 0x5B78), wxEmptyString, wxITEM_NORMAL);
    itemMenu11->Append(ID_MENUITEM8, wxGetTranslation(wxString() + (wxChar) 0x4F5C + (wxChar) 0x8005), wxEmptyString, wxITEM_NORMAL);
    menuBar->Append(itemMenu11, wxGetTranslation(wxString() + (wxChar) 0x95DC + (wxChar) 0x65BC));
    itemFrame1->SetMenuBar(menuBar);

    wxPanel* itemPanel15 = new wxPanel( itemFrame1, ID_PANEL2, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxTAB_TRAVERSAL );
    itemFrame1->GetAuiManager().AddPane(itemPanel15, wxAuiPaneInfo()
        .Name(_T("ID_PANEL2")).Caption(wxGetTranslation(wxString() + (wxChar) 0x8CC7 + (wxChar) 0x6599 + (wxChar) 0x8A2D + (wxChar) 0x5B9A + (wxChar) 0x90E8 + (wxChar) 0x4EFD)).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingSize(wxSize(400, 800)).MaximizeButton(true));

    wxGridBagSizer* itemGridBagSizer16 = new wxGridBagSizer(0, 0);
    itemGridBagSizer16->SetEmptyCellSize(wxSize(10, 20));
    itemPanel15->SetSizer(itemGridBagSizer16);

    wxButton* itemButton17 = new wxButton( itemPanel15, ID_BUTTON, wxGetTranslation(wxString() + (wxChar) 0x9078 + (wxChar) 0x53D6 + (wxChar) 0x7BC4 + (wxChar) 0x570D), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemButton17, wxGBPosition(3, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton18 = new wxButton( itemPanel15, ID_BUTTON1, wxGetTranslation(wxString() + (wxChar) 0x8F38 + (wxChar) 0x51FA + (wxChar) 0x7BC4 + (wxChar) 0x570D + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemButton18, wxGBPosition(4, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton19 = new wxButton( itemPanel15, ID_BUTTON5, wxGetTranslation(wxString() + (wxChar) 0x8A08 + (wxChar) 0x7B97 + (wxChar) 0x7BC4 + (wxChar) 0x570D + (wxChar) 0x5730 + (wxChar) 0x71B1), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemButton19, wxGBPosition(5, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText20 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x9078 + (wxChar) 0x64C7 + (wxChar) 0x7269 + (wxChar) 0x7406 + (wxChar) 0x91CF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemStaticText20, wxGBPosition(7, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton21 = new wxButton( itemPanel15, ID_BUTTON6, wxGetTranslation(wxString() + (wxChar) 0x8F38 + (wxChar) 0x5165 + (wxChar) 0x7CBE + (wxChar) 0x5BC6 + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemButton21, wxGBPosition(6, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton22 = new wxButton( itemPanel15, ID_BUTTON7, wxGetTranslation(wxString() + (wxChar) 0x79FB + (wxChar) 0x9664 + (wxChar) 0x7CBE + (wxChar) 0x5BC6 + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemButton22, wxGBPosition(6, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton23 = new wxButton( itemPanel15, ID_BUTTON8, wxGetTranslation(wxString() + (wxChar) 0x8F09 + (wxChar) 0x5165 + (wxChar) 0x4E95 + (wxChar) 0x6E2C + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemButton23, wxGBPosition(12, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton24 = new wxButton( itemPanel15, ID_BUTTON9, wxGetTranslation(wxString() + (wxChar) 0x53E6 + (wxChar) 0x5B58 + (wxChar) 0x4E95 + (wxChar) 0x6E2C + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemButton24, wxGBPosition(13, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText25 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x542B + wxT("5000kw") + (wxChar) 0x71B1 + (wxChar) 0x91CF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemStaticText25, wxGBPosition(5, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText26 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6ED1 + (wxChar) 0x9F20 + wxT(" EW,X,lon,") + (wxChar) 0x7D93), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemStaticText26, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxButton* itemButton27 = new wxButton( itemPanel15, ID_BUTTON10, wxGetTranslation(wxString() + (wxChar) 0x986F + (wxChar) 0x793A + (wxChar) 0x7BC4 + (wxChar) 0x570D + (wxChar) 0x5167 + (wxChar) 0x63D2 + (wxChar) 0x7D50 + (wxChar) 0x679C), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemButton27, wxGBPosition(4, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText28 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6BD4 + (wxChar) 0x4F8B + (wxChar) 0x5C3A), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemStaticText28, wxGBPosition(10, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxArrayString itemChoice29Strings;
    itemChoice29Strings.Add(wxGetTranslation(wxString() + (wxChar) 0x6EAB + (wxChar) 0x5EA6));
    itemChoice29Strings.Add(wxGetTranslation(wxString() + (wxChar) 0x96FB + (wxChar) 0x963B + (wxChar) 0x503C));
    itemChoice29Strings.Add(wxGetTranslation(wxString() + (wxChar) 0x78C1 + (wxChar) 0x5834 + (wxChar) 0x5F37 + (wxChar) 0x5EA6));
    itemChoice29Strings.Add(wxGetTranslation(wxString() + (wxChar) 0x5929 + (wxChar) 0x7136 + (wxChar) 0x6C23));
    itemChoice29Strings.Add(wxGetTranslation(wxString() + (wxChar) 0x9435));
    wxChoice* itemChoice29 = new wxChoice( itemPanel15, ID_CHOICE, wxDefaultPosition, wxDefaultSize, itemChoice29Strings, 0 );
    itemChoice29->SetStringSelection(wxGetTranslation(wxString() + (wxChar) 0x6EAB + (wxChar) 0x5EA6));
    itemGridBagSizer16->Add(itemChoice29, wxGBPosition(7, 1), wxGBSpan(1, 1), wxGROW|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText30 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6ED1 + (wxChar) 0x9F20 + wxT(" NS,Y,lat,") + (wxChar) 0x7DEF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemStaticText30, wxGBPosition(2, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    m_ruler_slider = new wxSlider( itemPanel15, ID_SLIDER, 0, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
    itemGridBagSizer16->Add(m_ruler_slider, wxGBPosition(10, 1), wxGBSpan(1, 1), wxGROW|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxCheckBox* itemCheckBox32 = new wxCheckBox( itemPanel15, ID_CHECKBOX, wxGetTranslation(wxString() + (wxChar) 0x986F + (wxChar) 0x793A + (wxChar) 0x5167 + (wxChar) 0x63D2 + (wxChar) 0x7BC4 + (wxChar) 0x570D), wxDefaultPosition, wxDefaultSize, 0 );
    itemCheckBox32->SetValue(false);
    itemGridBagSizer16->Add(itemCheckBox32, wxGBPosition(3, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText33 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x5167 + (wxChar) 0x63D2 + (wxChar) 0x65B9 + (wxChar) 0x6CD5), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemStaticText33, wxGBPosition(15, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxArrayString itemChoice34Strings;
    itemChoice34Strings.Add(_("InverseDistence"));
    itemChoice34Strings.Add(_("NearestNeighbor"));
    itemChoice34Strings.Add(_("Kriging"));
    wxChoice* itemChoice34 = new wxChoice( itemPanel15, ID_CHOICE1, wxDefaultPosition, wxDefaultSize, itemChoice34Strings, 0 );
    itemChoice34->SetStringSelection(_("InverseDistence"));
    itemGridBagSizer16->Add(itemChoice34, wxGBPosition(15, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText35 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x5167 + (wxChar) 0x63D2 + (wxChar) 0x689D + (wxChar) 0x4EF6 + (wxChar) 0x8173 + (wxChar) 0x672C), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemStaticText35, wxGBPosition(18, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxBoxSizer* itemBoxSizer36 = new wxBoxSizer(wxHORIZONTAL);
    itemGridBagSizer16->Add(itemBoxSizer36, wxGBPosition(11, 0), wxGBSpan(1, 2), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText37 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString(wxT("10 pixpel ")) + (wxChar) 0x6BD4), wxDefaultPosition, wxDefaultSize, 0 );
    itemBoxSizer36->Add(itemStaticText37, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_ruler_spinctrl = new wxSpinCtrl( itemPanel15, ID_SPINCTRL, _T("100"), wxDefaultPosition, wxSize(80, -1), wxSP_ARROW_KEYS, 100, 10000, 100 );
    itemBoxSizer36->Add(m_ruler_spinctrl, 0, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText39 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x516C + (wxChar) 0x5C3A), wxDefaultPosition, wxDefaultSize, 0 );
    itemBoxSizer36->Add(itemStaticText39, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl40 = new wxTextCtrl( itemPanel15, ID_TEXTCTRL24, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE );
    itemGridBagSizer16->Add(itemTextCtrl40, wxGBPosition(19, 0), wxGBSpan(2, 2), wxGROW|wxGROW|wxALL, 0);

    wxCheckBox* itemCheckBox41 = new wxCheckBox( itemPanel15, ID_CHECKBOX5, wxGetTranslation(wxString() + (wxChar) 0x986F + (wxChar) 0x793A + (wxChar) 0x5167 + (wxChar) 0x63D2 + (wxChar) 0x53C3 + (wxChar) 0x6578), wxDefaultPosition, wxDefaultSize, 0 );
    itemCheckBox41->SetValue(false);
    itemGridBagSizer16->Add(itemCheckBox41, wxGBPosition(16, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton42 = new wxButton( itemPanel15, ID_BUTTON3, wxGetTranslation(wxString() + (wxChar) 0x9A57 + (wxChar) 0x8B49 + (wxChar) 0x662F + (wxChar) 0x5426 + (wxChar) 0x6B63 + (wxChar) 0x78BA), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemButton42, wxGBPosition(18, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxCheckBox* itemCheckBox43 = new wxCheckBox( itemPanel15, ID_CHECKBOX3, wxGetTranslation(wxString() + (wxChar) 0x986F + (wxChar) 0x793A + (wxChar) 0x4E95 + (wxChar) 0x6E2C + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
    itemCheckBox43->SetValue(false);
    itemGridBagSizer16->Add(itemCheckBox43, wxGBPosition(12, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText44 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString(wxT("12345.67 ")) + (wxChar) 0x7D93), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemStaticText44, wxGBPosition(1, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText45 = new wxStaticText( itemPanel15, wxID_STATIC, wxGetTranslation(wxString(wxT("12345.67 ")) + (wxChar) 0x7DEF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer16->Add(itemStaticText45, wxGBPosition(2, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    // Fit to content
    itemFrame1->GetAuiManager().GetPane(_T("ID_PANEL2")).BestSize(itemPanel15->GetSizer()->Fit(itemPanel15)).MinSize(itemPanel15->GetSizer()->GetMinSize());

    wxPanel* itemPanel46 = new wxPanel( itemFrame1, ID_PANEL3, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxTAB_TRAVERSAL );
    itemFrame1->GetAuiManager().AddPane(itemPanel46, wxAuiPaneInfo()
        .Name(_T("ID_PANEL3")).Caption(wxGetTranslation(wxString() + (wxChar) 0x8CC7 + (wxChar) 0x6599 + (wxChar) 0x986F + (wxChar) 0x793A + (wxChar) 0x90E8 + (wxChar) 0x4EFD)).Centre().BestSize(wxSize(800, 800)).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingSize(wxSize(800, 800)).MaximizeButton(true));

    wxBoxSizer* itemBoxSizer47 = new wxBoxSizer(wxVERTICAL);
    itemPanel46->SetSizer(itemBoxSizer47);

    wxBoxSizer* itemBoxSizer48 = new wxBoxSizer(wxHORIZONTAL);
    itemBoxSizer47->Add(itemBoxSizer48, 0, wxGROW|wxALL, 5);

    wxStaticText* itemStaticText49 = new wxStaticText( itemPanel46, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x5730 + (wxChar) 0x5F62 + (wxChar) 0x986F + (wxChar) 0x793A + (wxChar) 0x90E8 + (wxChar) 0x4EFD), wxDefaultPosition, wxDefaultSize, 0 );
    itemBoxSizer48->Add(itemStaticText49, 1, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    m_Checkbox_Axis_Sync = new wxCheckBox( itemPanel46, ID_CHECKBOX1, wxGetTranslation(wxString() + (wxChar) 0x5EA7 + (wxChar) 0x6A19 + (wxChar) 0x540C + (wxChar) 0x6B65), wxDefaultPosition, wxDefaultSize, 0 );
    m_Checkbox_Axis_Sync->SetValue(false);
    itemBoxSizer48->Add(m_Checkbox_Axis_Sync, 0, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText51 = new wxStaticText( itemPanel46, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x5730 + (wxChar) 0x71B1 + (wxChar) 0x5207 + (wxChar) 0x9762 + (wxChar) 0x90E8 + (wxChar) 0x4EFD), wxDefaultPosition, wxDefaultSize, 0 );
    itemBoxSizer48->Add(itemStaticText51, 1, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    m_Checkbox_Depth_Sync = new wxCheckBox( itemPanel46, ID_CHECKBOX2, wxGetTranslation(wxString() + (wxChar) 0x6DF1 + (wxChar) 0x5EA6 + (wxChar) 0x540C + (wxChar) 0x6B65), wxDefaultPosition, wxDefaultSize, 0 );
    m_Checkbox_Depth_Sync->SetValue(false);
    itemBoxSizer48->Add(m_Checkbox_Depth_Sync, 0, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxBoxSizer* itemBoxSizer53 = new wxBoxSizer(wxHORIZONTAL);
    itemBoxSizer47->Add(itemBoxSizer53, 1, wxGROW|wxALL, 5);

    m_CanvasL = new wxGLCanvas( itemPanel46, ID_GLCANVAS1, wxDefaultPosition, wxDefaultSize, 0 );
    itemBoxSizer53->Add(m_CanvasL, 1, wxGROW|wxALL, 2);

    m_ScrollbarVL = new wxScrollBar( itemPanel46, ID_SCROLLBAR, wxDefaultPosition, wxDefaultSize, wxSB_VERTICAL );
    m_ScrollbarVL->SetScrollbar(0, 1, 100, 1);
    itemBoxSizer53->Add(m_ScrollbarVL, 0, wxGROW|wxALL, 1);

    m_CanvasR = new wxGLCanvas( itemPanel46, ID_GLCANVAS2, wxDefaultPosition, wxDefaultSize, 0 );
    itemBoxSizer53->Add(m_CanvasR, 1, wxGROW|wxALL, 2);

    m_ScrollbarRL = new wxScrollBar( itemPanel46, ID_SCROLLBAR5, wxDefaultPosition, wxDefaultSize, wxSB_VERTICAL );
    m_ScrollbarRL->SetScrollbar(0, 1, 100, 1);
    itemBoxSizer53->Add(m_ScrollbarRL, 0, wxGROW|wxALL, 1);

    wxBoxSizer* itemBoxSizer58 = new wxBoxSizer(wxHORIZONTAL);
    itemBoxSizer47->Add(itemBoxSizer58, 0, wxGROW|wxALL, 0);

    m_ScrollbarHL1 = new wxScrollBar( itemPanel46, ID_SCROLLBAR3, wxDefaultPosition, wxDefaultSize, wxSB_HORIZONTAL );
    m_ScrollbarHL1->SetScrollbar(0, 1, 100, 1);
    itemBoxSizer58->Add(m_ScrollbarHL1, 1, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText60 = new wxStaticText( itemPanel46, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x5EA7 + (wxChar) 0x6A19), wxDefaultPosition, wxDefaultSize, 0 );
    itemBoxSizer58->Add(itemStaticText60, 0, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    m_ScrollbarHR1 = new wxScrollBar( itemPanel46, ID_SCROLLBAR4, wxDefaultPosition, wxDefaultSize, wxSB_HORIZONTAL );
    m_ScrollbarHR1->SetScrollbar(0, 1, 100, 1);
    itemBoxSizer58->Add(m_ScrollbarHR1, 1, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxBoxSizer* itemBoxSizer62 = new wxBoxSizer(wxHORIZONTAL);
    itemBoxSizer47->Add(itemBoxSizer62, 0, wxGROW|wxALL, 0);

    m_ScrollbarHL2 = new wxScrollBar( itemPanel46, ID_SCROLLBAR1, wxDefaultPosition, wxDefaultSize, wxSB_HORIZONTAL );
    m_ScrollbarHL2->SetScrollbar(0, 1, 100, 1);
    itemBoxSizer62->Add(m_ScrollbarHL2, 1, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText64 = new wxStaticText( itemPanel46, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6DF1 + (wxChar) 0x5EA6), wxDefaultPosition, wxDefaultSize, 0 );
    itemBoxSizer62->Add(itemStaticText64, 0, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    m_ScrollbarHR2 = new wxScrollBar( itemPanel46, ID_SCROLLBAR2, wxDefaultPosition, wxDefaultSize, wxSB_HORIZONTAL );
    m_ScrollbarHR2->SetScrollbar(0, 1, 100, 1);
    itemBoxSizer62->Add(m_ScrollbarHR2, 1, wxALIGN_CENTER_VERTICAL|wxALL, 0);

    // Fit to content
    itemFrame1->GetAuiManager().GetPane(_T("ID_PANEL3")).BestSize(itemPanel46->GetSizer()->Fit(itemPanel46)).MinSize(itemPanel46->GetSizer()->GetMinSize());

    m_area_infopanel = new wxPanel( itemFrame1, ID_PANEL1, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
    itemFrame1->GetAuiManager().AddPane(m_area_infopanel, wxAuiPaneInfo()
        .Name(_T("ID_PANEL1")).Caption(wxGetTranslation(wxString() + (wxChar) 0x7BC4 + (wxChar) 0x570D + (wxChar) 0x5EA7 + (wxChar) 0x6A19)).Dockable(false).CloseButton(false).DestroyOnClose(false).Resizable(false).FloatingPosition(wxPoint(0, 600)).FloatingSize(wxSize(200, 200)).Hide().PaneBorder(false));

    wxGridBagSizer* itemGridBagSizer67 = new wxGridBagSizer(0, 0);
    itemGridBagSizer67->SetEmptyCellSize(wxSize(10, 10));
    m_area_infopanel->SetSizer(itemGridBagSizer67);

    wxStaticText* itemStaticText68 = new wxStaticText( m_area_infopanel, wxID_STATIC, wxGetTranslation(wxString(wxT("Maxmum EW,X,lon,")) + (wxChar) 0x7D93), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemStaticText68, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl69 = new wxTextCtrl( m_area_infopanel, ID_TEXTCTRL18, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemTextCtrl69, wxGBPosition(0, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText70 = new wxStaticText( m_area_infopanel, wxID_STATIC, wxGetTranslation(wxString(wxT("Minmum EW,X,lon,")) + (wxChar) 0x7D93), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemStaticText70, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl71 = new wxTextCtrl( m_area_infopanel, ID_TEXTCTRL19, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemTextCtrl71, wxGBPosition(1, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText72 = new wxStaticText( m_area_infopanel, wxID_STATIC, wxGetTranslation(wxString(wxT("Maxmum NS,Y,lat,")) + (wxChar) 0x7DEF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemStaticText72, wxGBPosition(2, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl73 = new wxTextCtrl( m_area_infopanel, ID_TEXTCTRL20, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemTextCtrl73, wxGBPosition(2, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText74 = new wxStaticText( m_area_infopanel, wxID_STATIC, wxGetTranslation(wxString(wxT("Minmum NS,Y,lat,")) + (wxChar) 0x7DEF), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemStaticText74, wxGBPosition(3, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl75 = new wxTextCtrl( m_area_infopanel, ID_TEXTCTRL21, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemTextCtrl75, wxGBPosition(3, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText76 = new wxStaticText( m_area_infopanel, wxID_STATIC, _("Maxmum Y,Height"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemStaticText76, wxGBPosition(4, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl77 = new wxTextCtrl( m_area_infopanel, ID_TEXTCTRL22, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemTextCtrl77, wxGBPosition(4, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxStaticText* itemStaticText78 = new wxStaticText( m_area_infopanel, wxID_STATIC, _("Minmum Y,Height"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemStaticText78, wxGBPosition(5, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    wxTextCtrl* itemTextCtrl79 = new wxTextCtrl( m_area_infopanel, ID_TEXTCTRL23, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer67->Add(itemTextCtrl79, wxGBPosition(5, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 0);

    // Fit to content
    itemFrame1->GetAuiManager().GetPane(_T("ID_PANEL1")).BestSize(m_area_infopanel->GetSizer()->Fit(m_area_infopanel)).MinSize(m_area_infopanel->GetSizer()->GetMinSize());

    wxAuiToolBar* itemAuiToolBar80 = new wxAuiToolBar( itemFrame1, ID_AUITOOLBAR, wxDefaultPosition, wxDefaultSize, wxAUI_TB_GRIPPER );
    wxBitmap itemtool81Bitmap(itemFrame1->GetBitmapResource(wxT("fileopen.xpm")));
    wxBitmap itemtool81BitmapDisabled;
    itemAuiToolBar80->AddTool(ID_TOOL1, wxEmptyString, itemtool81Bitmap, itemtool81BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool82Bitmap(itemFrame1->GetBitmapResource(wxT("filesave.xpm")));
    wxBitmap itemtool82BitmapDisabled;
    itemAuiToolBar80->AddTool(ID_TOOL2, wxEmptyString, itemtool82Bitmap, itemtool82BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    itemAuiToolBar80->AddSeparator();
    wxTextCtrl* itemTextCtrl84 = new wxTextCtrl( itemAuiToolBar80, ID_TEXTCTRL26, wxGetTranslation(wxString() + (wxChar) 0x8996 + (wxChar) 0x89D2 + (wxChar) 0x64CD + (wxChar) 0x4F5C), wxDefaultPosition, wxSize(65, -1), wxTE_READONLY|wxNO_BORDER|wxFULL_REPAINT_ON_RESIZE );
    itemTextCtrl84->SetBackgroundColour(wxColour(211, 211, 211));
    itemAuiToolBar80->AddControl(itemTextCtrl84);
    wxBitmap itemtool85Bitmap(itemFrame1->GetBitmapResource(wxT("find.xpm")));
    wxBitmap itemtool85BitmapDisabled;
    itemAuiToolBar80->AddTool(ID_TOOL, wxEmptyString, itemtool85Bitmap, itemtool85BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool86Bitmap(itemFrame1->GetBitmapResource(wxT("findrepl.xpm")));
    wxBitmap itemtool86BitmapDisabled;
    itemAuiToolBar80->AddTool(ID_TOOL8, wxEmptyString, itemtool86Bitmap, itemtool86BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool87Bitmap(itemFrame1->GetBitmapResource(wxT("home.xpm")));
    wxBitmap itemtool87BitmapDisabled;
    itemAuiToolBar80->AddTool(ID_TOOL3, wxEmptyString, itemtool87Bitmap, itemtool87BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool88Bitmap(itemFrame1->GetBitmapResource(wxT("back.xpm")));
    wxBitmap itemtool88BitmapDisabled;
    itemAuiToolBar80->AddTool(ID_TOOL5, wxEmptyString, itemtool88Bitmap, itemtool88BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool89Bitmap(itemFrame1->GetBitmapResource(wxT("up.xpm")));
    wxBitmap itemtool89BitmapDisabled;
    itemAuiToolBar80->AddTool(ID_TOOL4, wxEmptyString, itemtool89Bitmap, itemtool89BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool90Bitmap(itemFrame1->GetBitmapResource(wxT("down.xpm")));
    wxBitmap itemtool90BitmapDisabled;
    itemAuiToolBar80->AddTool(ID_TOOL6, wxEmptyString, itemtool90Bitmap, itemtool90BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool91Bitmap(itemFrame1->GetBitmapResource(wxT("forward.xpm")));
    wxBitmap itemtool91BitmapDisabled;
    itemAuiToolBar80->AddTool(ID_TOOL7, wxEmptyString, itemtool91Bitmap, itemtool91BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    itemAuiToolBar80->AddSeparator();
    itemAuiToolBar80->Realize();
    itemFrame1->GetAuiManager().AddPane(itemAuiToolBar80, wxAuiPaneInfo()
        .ToolbarPane().Name(_T("Pane1")).Top().Layer(10).CaptionVisible(false).CloseButton(false).DestroyOnClose(false).Resizable(false).Gripper(true));

    m_well_infopanel = new wxPanel( itemFrame1, ID_PANEL, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxTAB_TRAVERSAL );
    itemFrame1->GetAuiManager().AddPane(m_well_infopanel, wxAuiPaneInfo()
        .Name(_T("ID_PANEL")).Caption(wxGetTranslation(wxString() + (wxChar) 0x4E95 + (wxChar) 0x6E2C + (wxChar) 0x8CC7 + (wxChar) 0x6599)).Dockable(false).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingPosition(wxPoint(600, 200)).FloatingSize(wxSize(600, 800)).Hide().PaneBorder(false));

    wxBoxSizer* itemBoxSizer94 = new wxBoxSizer(wxHORIZONTAL);
    m_well_infopanel->SetSizer(itemBoxSizer94);

    wxGridBagSizer* itemGridBagSizer95 = new wxGridBagSizer(0, 0);
    itemGridBagSizer95->SetEmptyCellSize(wxSize(10, 20));
    itemBoxSizer94->Add(itemGridBagSizer95, 0, wxGROW|wxALL, 5);

    wxButton* itemButton96 = new wxButton( m_well_infopanel, ID_BUTTON4, wxGetTranslation(wxString() + (wxChar) 0x78BA + (wxChar) 0x5B9A), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer95->Add(itemButton96, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton97 = new wxButton( m_well_infopanel, ID_BUTTON11, wxGetTranslation(wxString() + (wxChar) 0x53D6 + (wxChar) 0x6D88), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer95->Add(itemButton97, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxCheckBox* itemCheckBox98 = new wxCheckBox( m_well_infopanel, ID_CHECKBOX4, wxGetTranslation(wxString() + (wxChar) 0x552F + (wxChar) 0x8B80), wxDefaultPosition, wxDefaultSize, 0 );
    itemCheckBox98->SetValue(true);
    itemGridBagSizer95->Add(itemCheckBox98, wxGBPosition(2, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxGrid* itemGrid99 = new wxGrid( m_well_infopanel, ID_GRID1, wxDefaultPosition, wxSize(200, 150), wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL );
    itemGrid99->SetDefaultColSize(50);
    itemGrid99->SetDefaultRowSize(25);
    itemGrid99->SetColLabelSize(25);
    itemGrid99->SetRowLabelSize(50);
    itemGrid99->CreateGrid(5, 5, wxGrid::wxGridSelectCells);
    itemBoxSizer94->Add(itemGrid99, 1, wxGROW|wxALL, 5);

    // Fit to content
    itemFrame1->GetAuiManager().GetPane(_T("ID_PANEL")).BestSize(m_well_infopanel->GetSizer()->Fit(m_well_infopanel)).MinSize(m_well_infopanel->GetSizer()->GetMinSize());

    GetAuiManager().Update();

    // Connect events and objects
    m_CanvasL->Connect(ID_GLCANVAS1, wxEVT_SIZE, wxSizeEventHandler(Taiwan::OnCanvasLSize), NULL, this);
    m_CanvasL->Connect(ID_GLCANVAS1, wxEVT_PAINT, wxPaintEventHandler(Taiwan::OnCanvasLPaint), NULL, this);
    m_CanvasL->Connect(ID_GLCANVAS1, wxEVT_MOTION, wxMouseEventHandler(Taiwan::OnCanvasLMotion), NULL, this);
    m_CanvasR->Connect(ID_GLCANVAS2, wxEVT_SIZE, wxSizeEventHandler(Taiwan::OnCanvasRSize), NULL, this);
    m_CanvasR->Connect(ID_GLCANVAS2, wxEVT_PAINT, wxPaintEventHandler(Taiwan::OnCanvasRPaint), NULL, this);
    m_CanvasR->Connect(ID_GLCANVAS2, wxEVT_MOTION, wxMouseEventHandler(Taiwan::OnCanvasRMotion), NULL, this);
	////@end Taiwan content construction
	m_FirstMain = NULL;
	m_SolidCtrlL.SetHwnd((HWND)m_CanvasL->GetHandle());
	m_SolidCtrlR.SetHwnd((HWND)m_CanvasR->GetHandle());
	m_timego = false;
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
    if (name == _T("fileopen.xpm"))
    {
        wxBitmap bitmap(fileopen_xpm);
        return bitmap;
    }
    else if (name == _T("filesave.xpm"))
    {
        wxBitmap bitmap(filesave_xpm);
        return bitmap;
    }
    else if (name == _T("find.xpm"))
    {
        wxBitmap bitmap(find_xpm);
        return bitmap;
    }
    else if (name == _T("findrepl.xpm"))
    {
        wxBitmap bitmap(findrepl_xpm);
        return bitmap;
    }
    else if (name == _T("home.xpm"))
    {
        wxBitmap bitmap(home_xpm);
        return bitmap;
    }
    else if (name == _T("back.xpm"))
    {
        wxBitmap bitmap(back_xpm);
        return bitmap;
    }
    else if (name == _T("up.xpm"))
    {
        wxBitmap bitmap(up_xpm);
        return bitmap;
    }
    else if (name == _T("down.xpm"))
    {
        wxBitmap bitmap(down_xpm);
        return bitmap;
    }
    else if (name == _T("forward.xpm"))
    {
        wxBitmap bitmap(forward_xpm);
        return bitmap;
    }
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


/*
* wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON10
*/

void Taiwan::OnShowResult( wxCommandEvent& event )
{
	m_FirstMain = new FirstMain(this);
	m_FirstMain->Show();
	event.Skip(false);
}


/*
* wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON8
*/

void Taiwan::OnModifyData( wxCommandEvent& event )
{
	vtkPoints_Sptr points2 = vtkSmartNew;
	vtkPolyData_Sptr polydata2 = vtkSmartNew;
	vtkDoubleArray_Sptr scalars2 = vtkSmartNew;
	std::ifstream istr1("Grid_use InverseDistanceH.dat");
	double nx, ny, nz;
	int i=0;
	if (istr1.good())
	{
		istr1 >>nx>>ny>>nz;
		for (;!istr1.eof();)
		{
			i++;
			double x, y, z, s;
			istr1 >>x>>y>>z>>s;
			points2->InsertNextPoint(x,y,z);
			scalars2->InsertNextTuple1(s);
		}
	}
	else
	{
		MessageBoxA(0, "can't read Grid_use InverseDistanceH.dat", "error!", 0);
		return;
	}
// 	char buffer[10];
// 	sprintf(buffer, "%d", i);
// 	MessageBoxA(0, buffer, "", 0);
	polydata2->SetPoints(points2);
	polydata2->GetPointData()->SetScalars(scalars2);
	SolidView_Sptr clip;
	SEffect_Sptr Setting = SEffect::New(SEffect::BOUNDING_BOX);
	m_SolidCtrlL.SetGridedData(polydata2, nx, ny, nz);
	m_SolidCtrlL.NewSEffect(Setting);
	m_SolidCtrlL.AddTaiwan();
	Setting = SEffect::New(SEffect::AXES);
	m_SolidCtrlL.NewSEffect(Setting);
	m_SolidCtrlL.Render();
	m_SolidCtrlR.SetGridedData(polydata2, nx, ny, nz);
	
	m_SolidCtrlR.NewSEffect(Setting);
	Setting = SEffect::New(SEffect::CLIP_PLANE);
	clip = m_SolidCtrlR.NewSEffect(Setting);
	Setting = SEffect::New(SEffect::CLIP_PLANE);
	clip = m_SolidCtrlR.NewSEffect(Setting);
	((ClipPlane_Setting*)Setting.get())->m_Axes = 1;
	clip->Update();
	Setting = SEffect::New(SEffect::CLIP_PLANE);
	clip = m_SolidCtrlR.NewSEffect(Setting);
	((ClipPlane_Setting*)Setting.get())->m_Axes = 2;
	clip->Update();
	m_SolidCtrlR.Render();
	event.Skip(false);
}


/*
* wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX
*/

void Taiwan::OnShowInfo( wxCommandEvent& event )
{
	GetAuiManager().GetPane(m_area_infopanel).Float();
	if (event.IsChecked())
		GetAuiManager().GetPane(m_area_infopanel).Show();
	else
		GetAuiManager().GetPane(m_area_infopanel).Show(false);
	GetAuiManager().Update();
	event.Skip(false);
}


/*
* wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX3
*/

void Taiwan::OnCheckShowWellInfo( wxCommandEvent& event )
{
	GetAuiManager().GetPane(m_well_infopanel).Float();

	if (event.IsChecked())
		GetAuiManager().GetPane(m_well_infopanel).Show();
	else
		GetAuiManager().GetPane(m_well_infopanel).Show(false);
	GetAuiManager().Update();
	event.Skip(false);
}


/*
* wxEVT_SIZE event handler for ID_GLCANVAS1
*/

void Taiwan::OnCanvasLSize( wxSizeEvent& event )
{
	m_SolidCtrlL.ReSize(event.GetSize().GetWidth(),event.GetSize().GetHeight());
	m_SolidCtrlL.Render();
	event.Skip(false);
}


/*
* wxEVT_PAINT event handler for ID_GLCANVAS1
*/

void Taiwan::OnCanvasLPaint( wxPaintEvent& event )
{
	wxPaintDC dc(wxDynamicCast(event.GetEventObject(), wxWindow));
	m_SolidCtrlL.ReSize(m_CanvasL->GetSize().GetWidth(), 
		m_CanvasL->GetSize().GetHeight());
	m_SolidCtrlL.Render();
}


/*
* wxEVT_SIZE event handler for ID_GLCANVAS2
*/

void Taiwan::OnCanvasRSize( wxSizeEvent& event )
{
	m_SolidCtrlR.ReSize(event.GetSize().GetWidth(),event.GetSize().GetHeight());
	m_SolidCtrlR.Render();
	event.Skip(false);
}


/*
* wxEVT_PAINT event handler for ID_GLCANVAS2
*/

void Taiwan::OnCanvasRPaint( wxPaintEvent& event )
{
	wxPaintDC dc(wxDynamicCast(event.GetEventObject(), wxWindow));
	m_SolidCtrlR.ReSize(m_CanvasR->GetSize().GetWidth(), 
		m_CanvasR->GetSize().GetHeight());
	m_SolidCtrlR.Render();
}


/*
* wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR
*/

void Taiwan::OnScrollbarVLUpdated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR in Taiwan.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR in Taiwan. 
}


/*
* wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR5
*/

void Taiwan::OnScrollbarRLUpdated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR5 in Taiwan.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR5 in Taiwan. 
}


/*
* wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR3
*/

void Taiwan::OnScrollbarHL1Updated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR3 in Taiwan.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR3 in Taiwan. 
}


/*
* wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR1
*/

void Taiwan::OnScrollbarHL2Updated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR1 in Taiwan.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR1 in Taiwan. 
}


/*
* wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR4
*/

void Taiwan::OnScrollbarHR1Updated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR4 in Taiwan.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR4 in Taiwan. 
}


/*
* wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR2
*/

void Taiwan::OnScrollbarHR2Updated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR2 in Taiwan.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR2 in Taiwan. 
}

Taiwan* mf;
/*
* wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX1
*/

void Taiwan::OnCheckboxAxis_Sync( wxCommandEvent& event )
{
	if (!m_timego && event.IsChecked())
	{
		SetTimer ((HWND)(this->GetHandle()), DRAW_TIMER, 0.5, TimerProc);
		m_timego = true;
		mf = this;
	}
	else if (m_timego && !event.IsChecked())
	{
		KillTimer((HWND)(this->GetHandle()), DRAW_TIMER);
	}
	event.Skip(false);
}


/*
* wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX2
*/

void Taiwan::OnCheckboxDepth_Sync( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX2 in Taiwan.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX2 in Taiwan. 
}


/*
 * wxEVT_MOTION event handler for ID_GLCANVAS1
 */

void Taiwan::OnCanvasLMotion( wxMouseEvent& event )
{
	event.Skip(false);
}


/*
 * wxEVT_MOTION event handler for ID_GLCANVAS2
 */

void Taiwan::OnCanvasRMotion( wxMouseEvent& event )
{
	event.Skip(false);
}

VOID CALLBACK TimerProc ( HWND hParent, UINT uMsg, UINT uEventID, DWORD dwTimer )
{
	if (mf->m_timego)
	{
		mf->m_SolidCtrlR.SetCamera(mf->m_SolidCtrlL.m_Camera);
	}
	mf->m_SolidCtrlR.Render();
}