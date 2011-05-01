////////////////////////////////////////////////////////////////////////////
// Name:        convertdialog.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     24/02/2011 06:38:39
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////
#include "StdWxVtk.h"
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
#include "DW/PolyDataHandler.h"

/*
 * ConvertDialog type definition
 */

IMPLEMENT_DYNAMIC_CLASS( ConvertDialog, wxDialog )


/*
 * ConvertDialog event table definition
 */

BEGIN_EVENT_TABLE( ConvertDialog, wxDialog )

////@begin ConvertDialog event table entries
    EVT_BUTTON( ID_CLOSE, ConvertDialog::OnCloseClick )

    EVT_BUTTON( ID_CONVERT, ConvertDialog::OnConvertClick )

    EVT_BUTTON( ID_BUTTON2, ConvertDialog::OnLoadClick )

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
    m_in_xmin = NULL;
    m_in_ymin = NULL;
    m_in_zmin = NULL;
    m_in_xmax = NULL;
    m_in_ymax = NULL;
    m_in_zmax = NULL;
    m_out_xmin = NULL;
    m_out_ymin = NULL;
    m_out_zmin = NULL;
    m_out_xmax = NULL;
    m_out_ymax = NULL;
    m_out_zmax = NULL;
    m_out_xinterval = NULL;
    m_out_yinterval = NULL;
    m_out_zinterval = NULL;
    m_filectrl_input = NULL;
    m_filectrl_output = NULL;
    m_in_datatotal = NULL;
    m_out_datatotal = NULL;
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

    wxStaticText* itemStaticText9 = new wxStaticText( itemDialog1, wxID_STATIC, _("input"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText9, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText10 = new wxStaticText( itemDialog1, wxID_STATIC, _("output"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText10, wxGBPosition(5, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_in_xmin = new wxTextCtrl( itemDialog1, ID_TEXTCTRL, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    m_in_xmin->Enable(false);
    itemGridBagSizer2->Add(m_in_xmin, wxGBPosition(1, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_in_ymin = new wxTextCtrl( itemDialog1, ID_TEXTCTRL1, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    m_in_ymin->Enable(false);
    itemGridBagSizer2->Add(m_in_ymin, wxGBPosition(1, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_in_zmin = new wxTextCtrl( itemDialog1, ID_TEXTCTRL2, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    m_in_zmin->Enable(false);
    itemGridBagSizer2->Add(m_in_zmin, wxGBPosition(1, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_in_xmax = new wxTextCtrl( itemDialog1, ID_TEXTCTRL3, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    m_in_xmax->Enable(false);
    itemGridBagSizer2->Add(m_in_xmax, wxGBPosition(2, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_in_ymax = new wxTextCtrl( itemDialog1, ID_TEXTCTRL4, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    m_in_ymax->Enable(false);
    itemGridBagSizer2->Add(m_in_ymax, wxGBPosition(2, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_in_zmax = new wxTextCtrl( itemDialog1, ID_TEXTCTRL5, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    m_in_zmax->Enable(false);
    itemGridBagSizer2->Add(m_in_zmax, wxGBPosition(2, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_out_xmin = new wxTextCtrl( itemDialog1, ID_TEXTCTRL6, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    itemGridBagSizer2->Add(m_out_xmin, wxGBPosition(6, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

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

    m_out_ymin = new wxTextCtrl( itemDialog1, ID_TEXTCTRL7, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    itemGridBagSizer2->Add(m_out_ymin, wxGBPosition(6, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_out_zmin = new wxTextCtrl( itemDialog1, ID_TEXTCTRL8, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    itemGridBagSizer2->Add(m_out_zmin, wxGBPosition(6, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_out_xmax = new wxTextCtrl( itemDialog1, ID_TEXTCTRL9, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    itemGridBagSizer2->Add(m_out_xmax, wxGBPosition(7, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_out_ymax = new wxTextCtrl( itemDialog1, ID_TEXTCTRL10, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    itemGridBagSizer2->Add(m_out_ymax, wxGBPosition(7, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_out_zmax = new wxTextCtrl( itemDialog1, ID_TEXTCTRL11, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    itemGridBagSizer2->Add(m_out_zmax, wxGBPosition(7, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_out_xinterval = new wxTextCtrl( itemDialog1, ID_TEXTCTRL12, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    itemGridBagSizer2->Add(m_out_xinterval, wxGBPosition(8, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_out_yinterval = new wxTextCtrl( itemDialog1, ID_TEXTCTRL13, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    itemGridBagSizer2->Add(m_out_yinterval, wxGBPosition(8, 3), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_out_zinterval = new wxTextCtrl( itemDialog1, ID_TEXTCTRL14, wxEmptyString, wxDefaultPosition, wxSize(120, -1), 0 );
    itemGridBagSizer2->Add(m_out_zinterval, wxGBPosition(8, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton33 = new wxButton( itemDialog1, ID_CLOSE, _("Close"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton33, wxGBPosition(12, 5), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxButton* itemButton34 = new wxButton( itemDialog1, ID_CONVERT, _("Convert"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton34, wxGBPosition(12, 4), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_filectrl_input = new wxFilePickerCtrl( itemDialog1, ID_FILECTRL_INPUT, wxEmptyString, wxEmptyString, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
    itemGridBagSizer2->Add(m_filectrl_input, wxGBPosition(4, 0), wxGBSpan(1, 5), wxALIGN_CENTER_HORIZONTAL|wxGROW|wxALL, 5);

    wxButton* itemButton36 = new wxButton( itemDialog1, ID_BUTTON2, _("Load"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemButton36, wxGBPosition(4, 5), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_filectrl_output = new wxFilePickerCtrl( itemDialog1, ID_FILEPICKERCTRL, wxEmptyString, wxEmptyString, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
    itemGridBagSizer2->Add(m_filectrl_output, wxGBPosition(10, 0), wxGBSpan(1, 5), wxALIGN_CENTER_HORIZONTAL|wxGROW|wxALL, 5);

    wxStaticText* itemStaticText38 = new wxStaticText( itemDialog1, wxID_STATIC, _("data total"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText38, wxGBPosition(3, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText39 = new wxStaticText( itemDialog1, wxID_STATIC, _("data total"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText39, wxGBPosition(9, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_in_datatotal = new wxTextCtrl( itemDialog1, ID_TEXTCTRL15, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    m_in_datatotal->Enable(false);
    itemGridBagSizer2->Add(m_in_datatotal, wxGBPosition(3, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_out_datatotal = new wxTextCtrl( itemDialog1, ID_TEXTCTRL16, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    m_out_datatotal->Enable(false);
    itemGridBagSizer2->Add(m_out_datatotal, wxGBPosition(9, 2), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxTextCtrl* itemTextCtrl42 = new wxTextCtrl( itemDialog1, ID_TEXTCTRL17, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE );
    itemGridBagSizer2->Add(itemTextCtrl42, wxGBPosition(11, 0), wxGBSpan(2, 3), wxALIGN_CENTER_HORIZONTAL|wxGROW|wxALL, 5);

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


/*
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON2
 */

void ConvertDialog::OnLoadClick( wxCommandEvent& event )
{
	event.Skip(false);
	if (	m_filectrl_input->GetTextCtrlValue().size() == 0 || 
		!m_filectrl_input->CheckPath(m_filectrl_input->GetTextCtrlValue()))
	{
		wxString mess;
		mess << wxT("你沒有設定正確的儲存路徑!!!");
		wxMessageDialog dialog(NULL, mess, wxT("注意！"));
		dialog.ShowModal();
	}
	else
	{
		vtkPolyData_Sptrs polydatas = PolyDataHandler::LoadFileFromNative(
			m_filectrl_input->GetTextCtrlValue().wc_str()
			);
		vtkBounds bounds;
		polydatas[0]->GetBounds(bounds);
		wxString value;
		value << bounds.xmin;
		m_in_xmin->SetValue(value);
		value.clear();
		value << bounds.xmax;
		m_in_xmax->SetValue(value);
		value.clear();
		value << bounds.ymin;
		m_in_ymin->SetValue(value);
		value.clear();
		value << bounds.ymax;
		m_in_ymax->SetValue(value);
		value.clear();
		value << bounds.zmin;
		m_in_zmin->SetValue(value);
		value.clear();
		value << bounds.zmax;
		m_in_zmax->SetValue(value);
		value.clear();
		value << polydatas[0]->GetNumberOfPoints();
		m_in_datatotal->SetValue(value);

		InterpolationInfo info(polydatas[0]->GetNumberOfPoints());
		info.interval[0] = 20;
		info.interval[1] = 20;
		info.interval[2] = 20;
		info.min[0] = bounds.xmin;
		info.min[1] = bounds.ymin;
		info.min[2] = bounds.zmin;
		info.max[0] = bounds.xmax;
		info.max[1] = bounds.ymax;
		info.max[2] = bounds.zmax;

		PolyDataHandler::InterpolationPolyData(polydatas, &info);
		PolyDataHandler::SavePolyDatasToEvrA(polydatas, L"c:\\koko", L"koko");
	}
	
}


/*
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_CONVERT
 */

void ConvertDialog::OnConvertClick( wxCommandEvent& event )
{
////@begin wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_CONVERT in ConvertDialog.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_CONVERT in ConvertDialog. 
}


/*
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_CLOSE
 */

void ConvertDialog::OnCloseClick( wxCommandEvent& event )
{
	event.Skip(false);
	this->EndDialog(0);
}

