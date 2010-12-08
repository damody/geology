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
#include "rs232.h"
#include <windows.h>
/*
 * mainframe type definition
 */

IMPLEMENT_CLASS( mainframe, wxFrame )


/*
 * mainframe event table definition
 */

BEGIN_EVENT_TABLE( mainframe, wxFrame )

////@begin mainframe event table entries
    EVT_CHOICE( ID_CHOICE1, mainframe::OnChoice1Selected )

    EVT_CHOICE( ID_CHOICE, mainframe::OnChoiceSelected )

    EVT_BUTTON( ID_BUTTON, mainframe::OnStartGetClick )

    EVT_CHECKBOX( ID_CHECKBOX, mainframe::OnCheckboxClick )

    EVT_BUTTON( ID_BUTTON1, mainframe::OnStopGetClick )

    EVT_FILEPICKER_CHANGED( ID_FILECTRL, mainframe::OnFilectrlFilePickerChanged )

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
    m_BoundRate = NULL;
    m_Combo_ComPort = NULL;
    m_BtnStartGet = NULL;
    m_OutputText = NULL;
    m_BtnStopGet = NULL;
    m_Browse = NULL;
    m_WaterDepth = NULL;
    m_Longitude = NULL;
    m_Latitude = NULL;
    m_DataTotal = NULL;
    m_GLCanvas = NULL;
////@end mainframe member initialisation
    m_port = NULL;
    m_open = NULL;
}


/*
 * Control creation for mainframe
 */

void mainframe::CreateControls()
{    
////@begin mainframe content construction
    mainframe* itemFrame1 = this;

    wxMenuBar* menuBar = new wxMenuBar;
    wxMenu* itemMenu25 = new wxMenu;
    menuBar->Append(itemMenu25, _("File"));
    itemFrame1->SetMenuBar(menuBar);

    wxGridBagSizer* itemGridBagSizer2 = new wxGridBagSizer(0, 0);
    itemGridBagSizer2->SetEmptyCellSize(wxSize(10, 20));
    itemFrame1->SetSizer(itemGridBagSizer2);

    wxArrayString m_BoundRateStrings;
    m_BoundRateStrings.Add(_("110"));
    m_BoundRateStrings.Add(_("300"));
    m_BoundRateStrings.Add(_("600"));
    m_BoundRateStrings.Add(_("1200"));
    m_BoundRateStrings.Add(_("2400"));
    m_BoundRateStrings.Add(_("4800"));
    m_BoundRateStrings.Add(_("9600"));
    m_BoundRateStrings.Add(_("19200"));
    m_BoundRateStrings.Add(_("38400"));
    m_BoundRateStrings.Add(_("57600"));
    m_BoundRateStrings.Add(_("115200"));
    m_BoundRateStrings.Add(_("128000"));
    m_BoundRateStrings.Add(_("256000"));
    m_BoundRate = new wxChoice( itemFrame1, ID_CHOICE1, wxDefaultPosition, wxDefaultSize, m_BoundRateStrings, 0 );
    m_BoundRate->SetStringSelection(_("38400"));
    itemGridBagSizer2->Add(m_BoundRate, wxGBPosition(0, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxArrayString m_Combo_ComPortStrings;
    m_Combo_ComPortStrings.Add(_("1"));
    m_Combo_ComPortStrings.Add(_("2"));
    m_Combo_ComPortStrings.Add(_("3"));
    m_Combo_ComPortStrings.Add(_("4"));
    m_Combo_ComPortStrings.Add(_("5"));
    m_Combo_ComPortStrings.Add(_("6"));
    m_Combo_ComPortStrings.Add(_("7"));
    m_Combo_ComPortStrings.Add(_("8"));
    m_Combo_ComPortStrings.Add(_("9"));
    m_Combo_ComPortStrings.Add(_("10"));
    m_Combo_ComPortStrings.Add(_("11"));
    m_Combo_ComPortStrings.Add(_("12"));
    m_Combo_ComPortStrings.Add(_("13"));
    m_Combo_ComPortStrings.Add(_("14"));
    m_Combo_ComPortStrings.Add(_("15"));
    m_Combo_ComPort = new wxChoice( itemFrame1, ID_CHOICE, wxDefaultPosition, wxDefaultSize, m_Combo_ComPortStrings, 0 );
    m_Combo_ComPort->SetStringSelection(_("1"));
    itemGridBagSizer2->Add(m_Combo_ComPort, wxGBPosition(1, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText5 = new wxStaticText( itemFrame1, wxID_STATIC, _("BoundRate"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText5, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText6 = new wxStaticText( itemFrame1, wxID_STATIC, _("ComPort"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText6, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_BtnStartGet = new wxButton( itemFrame1, ID_BUTTON, _("StartGet"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(m_BtnStartGet, wxGBPosition(2, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_OutputText = new wxTextCtrl( itemFrame1, ID_TEXTCTRL, wxEmptyString, wxDefaultPosition, wxSize(780, 100), wxTE_MULTILINE );
    itemGridBagSizer2->Add(m_OutputText, wxGBPosition(14, 0), wxGBSpan(2, 3), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxCheckBox* itemCheckBox9 = new wxCheckBox( itemFrame1, ID_CHECKBOX, _("SeeOutput"), wxDefaultPosition, wxDefaultSize, 0 );
    itemCheckBox9->SetValue(false);
    itemGridBagSizer2->Add(itemCheckBox9, wxGBPosition(13, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_BtnStopGet = new wxButton( itemFrame1, ID_BUTTON1, _("StopGet"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(m_BtnStopGet, wxGBPosition(2, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_Browse = new wxFilePickerCtrl( itemFrame1, ID_FILECTRL, wxEmptyString, wxEmptyString, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
    itemGridBagSizer2->Add(m_Browse, wxGBPosition(12, 0), wxGBSpan(1, 2), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText12 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x8CC7 + (wxChar) 0x6599 + (wxChar) 0x500B + (wxChar) 0x6578), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText12, wxGBPosition(3, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText13 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x7D93 + (wxChar) 0x5EA6 + wxT("E")), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText13, wxGBPosition(5, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_WaterDepth = new wxStaticText( itemFrame1, wxID_STATIC, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(m_WaterDepth, wxGBPosition(4, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText15 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6C34 + (wxChar) 0x6DF1), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText15, wxGBPosition(4, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_Longitude = new wxStaticText( itemFrame1, wxID_STATIC, _("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(m_Longitude, wxGBPosition(5, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText17 = new wxStaticText( itemFrame1, wxID_STATIC, _("0kb"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText17, wxGBPosition(10, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText18 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x7DEF + (wxChar) 0x5EA6 + wxT("N")), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText18, wxGBPosition(6, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_Latitude = new wxStaticText( itemFrame1, wxID_STATIC, _("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(m_Latitude, wxGBPosition(6, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_DataTotal = new wxStaticText( itemFrame1, wxID_STATIC, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(m_DataTotal, wxGBPosition(3, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText21 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6A94 + (wxChar) 0x6848 + (wxChar) 0x8DEF + (wxChar) 0x5F91), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText21, wxGBPosition(11, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    wxStaticText* itemStaticText22 = new wxStaticText( itemFrame1, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6A94 + (wxChar) 0x6848 + (wxChar) 0x5927 + (wxChar) 0x5C0F), wxDefaultPosition, wxDefaultSize, 0 );
    itemGridBagSizer2->Add(itemStaticText22, wxGBPosition(10, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

    m_GLCanvas = new wxGLCanvas( itemFrame1, ID_GLCANVAS, wxDefaultPosition, wxSize(570, 420), 0 );
    itemGridBagSizer2->Add(m_GLCanvas, wxGBPosition(0, 2), wxGBSpan(14, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

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


/*
 * wxEVT_FILEPICKER_CHANGED event handler for ID_FILECTRL
 */

void mainframe::OnFilectrlFilePickerChanged( wxFileDirPickerEvent& event )
{
////@begin wxEVT_FILEPICKER_CHANGED event handler for ID_FILECTRL in mainframe.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_FILEPICKER_CHANGED event handler for ID_FILECTRL in mainframe. 
}
mainframe* mf;
VOID CALLBACK TimerProc ( HWND hParent, UINT uMsg, UINT uEventID, DWORD dwTimer )
{
	if (mf->m_open)
	{
		char buffer[1024];
		memset(buffer, 0, sizeof(char)*1024);
		PollComport(mf->m_port-1, (unsigned char*)buffer, 1024);
		// 增加文字到OutputText
		wxString aText = wxString::FromAscii(buffer);
		mf->m_OutputText->AppendText(aText);
		char xbuf[1024];
		memset(xbuf,0,1024);
		int len = strlen(buffer);
		for (;len>10;)
		{
			for (int i=0;i<len;i++)
			{
				if (i!=0 && buffer[i]=='*')
				{
					memcpy(xbuf, buffer, i+3);
					xbuf[i+3] = '\r';
					xbuf[i+4] = '\n';
					xbuf[i+5] = '\0';
					mf->m_nCell.InputLine(std::string(xbuf));
					memcpy(buffer, buffer+i+3, len-i-2);
					break;
				}
				else if (i == len-1)
				{
					mf->m_nCell.InputLine(std::string(buffer));
					buffer[0] = '\0';
				}
			}
			len = strlen(buffer);
		}
		if (mf->m_nCell.GetLastData().lat != 0)
		{
			wxString mess;
			mess << mf->m_nCell.GetLastData().lat;
			mf->m_Latitude->SetLabel(mess);
		}
		if (mf->m_nCell.GetLastData().lon != 0)
		{
			wxString mess;
			mess << mf->m_nCell.GetLastData().lon;
			mf->m_Longitude->SetLabel(mess);
		}
		if (mf->m_nCell.GetLastData().depthinfo.depth_M != 0)
		{
			wxString mess;
			mess << mf->m_nCell.GetLastData().depthinfo.depth_M;
			mf->m_WaterDepth->SetLabel(mess);
		}
		if (mf->m_nCell.GetTotal() != 0)
		{
			wxString mess;
			mess << mf->m_nCell.GetTotal();
			mf->m_DataTotal->SetLabel(mess);
		}
	}
}


/*
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON
 */

void mainframe::OnStartGetClick( wxCommandEvent& event )
{
	if (m_open)
	{
		CloseComport(m_port-1);
		m_open = false;
	}
	char buffer[1024];
	long boundRate = 38400;
	m_Combo_ComPort->GetString(m_Combo_ComPort->GetSelection()).ToLong(&m_port);
	m_BoundRate->GetString(m_BoundRate->GetSelection()).ToLong(&boundRate);
	wxString mess;
	if (OpenComport(m_port-1, boundRate))
	{
		mess << wxT("comport open error boundRate:") << boundRate << wxT(" comport:") << m_port;
		wxMessageDialog dialog(NULL, mess, wxT("no found"));
		dialog.ShowModal();
	}
	else
	{
		mess << wxT("comport is open by boundRate:") << boundRate << wxT(" comport:") << m_port;
		wxMessageDialog dialog(NULL, mess, wxT("found"));
		dialog.ShowModal();
		m_open = true;
		UINT myTimer = SetTimer ( NULL, 1, 500, TimerProc );
		mf = this;
	}
}


/*
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON1
 */

void mainframe::OnStopGetClick( wxCommandEvent& event )
{
	if (m_open)
	{
		wxString mess;
		mess << wxT("close port:") << m_port;
		wxMessageDialog dialog(NULL, mess, wxT("found"));
		dialog.ShowModal();
		CloseComport(m_port-1);
		m_open = false;
	}
}


/*
 * wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_CHOICE1
 */

void mainframe::OnChoice1Selected( wxCommandEvent& event )
{
////@begin wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_CHOICE1 in mainframe.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_CHOICE1 in mainframe. 
}


/*
 * wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_CHOICE
 */

void mainframe::OnChoiceSelected( wxCommandEvent& event )
{
////@begin wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_CHOICE in mainframe.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_CHOICE in mainframe. 
}


/*
 * wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX
 */

void mainframe::OnCheckboxClick( wxCommandEvent& event )
{
////@begin wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX in mainframe.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX in mainframe. 
}



