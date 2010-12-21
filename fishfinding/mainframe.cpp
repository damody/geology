/////////////////////////////////////////////////////////////////////////////
// Name:        mainframe.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     19/11/2010 20:44:33Za
// RCS-ID:      
// Copyright:   ntust
// Licence:     
/////////////////////////////////////////////////////////////////////////////

// For compilers that support precompilation, includes "wx/wx.h".
#include "StdVtkWx.h"
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

	EVT_CHECKBOX( ID_CHECKBOX, mainframe::OnOutputTextVisable )

	EVT_SPINCTRL( ID_SPINCTRL2, mainframe::OnPointSizeUpdated )

	EVT_CHECKBOX( ID_CHECKBOX2, mainframe::OnCheckbox2Click )

	EVT_SPINCTRL( ID_SPINCTRL, mainframe::OnSpinctrlUpdated )

	EVT_BUTTON( ID_BUTTON1, mainframe::OnStopGetClick )

	EVT_FILEPICKER_CHANGED( ID_FILECTRL, mainframe::OnFilectrlFilePickerChanged )

	EVT_COLOURPICKER_CHANGED( ID_COLOURCTRL, mainframe::OnDeepColorChanged )

	EVT_COLOURPICKER_CHANGED( ID_COLOURPICKERCTRL, mainframe::OnhsColorChanged )

	EVT_BUTTON( ID_BUTTON2, mainframe::OnLoadFileDataClick )

	EVT_BUTTON( ID_BUTTON3, mainframe::OnClearDataClick )

	EVT_TEXT( ID_TEXTCTRL, mainframe::OnTimerUpdated )

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
	GetAuiManager().UnInit();
////@end mainframe destruction
}

mainframe* mf;
VOID CALLBACK TimerProc2 ( HWND hParent, UINT uMsg, UINT uEventID, DWORD dwTimer )
{
	mf->RenderFrame();
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
	m_CanOutput = NULL;
	m_point_size = NULL;
	m_NormalLook = NULL;
	m_spinctrl_height = NULL;
	m_FocusLast = NULL;
	m_BtnStopGet = NULL;
	m_Browse = NULL;
	m_WaterDepth = NULL;
	m_Longitude = NULL;
	m_Latitude = NULL;
	m_DataTotal = NULL;
	m_deColor = NULL;
	m_hsColor = NULL;
	m_UpdateInterval = NULL;
	m_GLCanvas = NULL;
	m_OutputText = NULL;
////@end mainframe member initialisation
	m_port = NULL;
	m_open = false;
	m_timer_go = false;
}


/*
 * Control creation for mainframe
 */

void mainframe::CreateControls()
{    
////@begin mainframe content construction
	mainframe* itemFrame1 = this;

	GetAuiManager().SetManagedWindow(this);

	wxPanel* itemPanel2 = new wxPanel( itemFrame1, ID_PANEL, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxTAB_TRAVERSAL );
	itemFrame1->GetAuiManager().AddPane(itemPanel2, wxAuiPaneInfo()
		.Name(_T("Pane1")).MinSize(wxSize(200, 500)).CloseButton(false).DestroyOnClose(false).Resizable(true).Floatable(false));

	wxGridBagSizer* itemGridBagSizer3 = new wxGridBagSizer(0, 0);
	itemGridBagSizer3->SetEmptyCellSize(wxSize(10, 20));
	itemPanel2->SetSizer(itemGridBagSizer3);

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
	m_BoundRate = new wxChoice( itemPanel2, ID_CHOICE1, wxDefaultPosition, wxDefaultSize, m_BoundRateStrings, 0 );
	m_BoundRate->SetStringSelection(_("38400"));
	itemGridBagSizer3->Add(m_BoundRate, wxGBPosition(0, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

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
	m_Combo_ComPort = new wxChoice( itemPanel2, ID_CHOICE, wxDefaultPosition, wxDefaultSize, m_Combo_ComPortStrings, 0 );
	m_Combo_ComPort->SetStringSelection(_("1"));
	itemGridBagSizer3->Add(m_Combo_ComPort, wxGBPosition(1, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText6 = new wxStaticText( itemPanel2, wxID_STATIC, _("BoundRate"), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText6, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText7 = new wxStaticText( itemPanel2, wxID_STATIC, _("ComPort"), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText7, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_BtnStartGet = new wxButton( itemPanel2, ID_BUTTON, wxGetTranslation(wxString() + (wxChar) 0x958B + (wxChar) 0x59CB + (wxChar) 0x63A5 + (wxChar) 0x6536), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(m_BtnStartGet, wxGBPosition(2, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_CanOutput = new wxCheckBox( itemPanel2, ID_CHECKBOX, wxGetTranslation(wxString() + (wxChar) 0x6587 + (wxChar) 0x5B57 + (wxChar) 0x8F38 + (wxChar) 0x51FA), wxDefaultPosition, wxDefaultSize, 0 );
	m_CanOutput->SetValue(false);
	itemGridBagSizer3->Add(m_CanOutput, wxGBPosition(10, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_point_size = new wxSpinCtrl( itemPanel2, ID_SPINCTRL2, _T("3"), wxDefaultPosition, wxSize(80, -1), wxSP_ARROW_KEYS, 0, 10, 3 );
	itemGridBagSizer3->Add(m_point_size, wxGBPosition(15, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText11 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x9EDE + (wxChar) 0x7684 + (wxChar) 0x5927 + (wxChar) 0x5C0F), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText11, wxGBPosition(15, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText12 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x66F4 + (wxChar) 0x65B0 + (wxChar) 0x901F + (wxChar) 0x5EA6), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText12, wxGBPosition(14, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText13 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6700 + (wxChar) 0x6DF1 + (wxChar) 0x6C34 + (wxChar) 0x6DF1), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText13, wxGBPosition(16, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_NormalLook = new wxCheckBox( itemPanel2, ID_CHECKBOX2, wxGetTranslation(wxString() + (wxChar) 0x5782 + (wxChar) 0x76F4 + (wxChar) 0x4FEF + (wxChar) 0x8996), wxDefaultPosition, wxDefaultSize, 0 );
	m_NormalLook->SetValue(true);
	itemGridBagSizer3->Add(m_NormalLook, wxGBPosition(11, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText15 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x89C0 + (wxChar) 0x770B + (wxChar) 0x9AD8 + (wxChar) 0x5EA6), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText15, wxGBPosition(9, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_spinctrl_height = new wxSpinCtrl( itemPanel2, ID_SPINCTRL, _T("3"), wxDefaultPosition, wxSize(80, -1), wxSP_ARROW_KEYS, -32760, 32760, 3 );
	itemGridBagSizer3->Add(m_spinctrl_height, wxGBPosition(9, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_FocusLast = new wxCheckBox( itemPanel2, ID_CHECKBOX1, wxGetTranslation(wxString() + (wxChar) 0x9396 + (wxChar) 0x5B9A + (wxChar) 0x8239 + (wxChar) 0x4F4D), wxDefaultPosition, wxDefaultSize, 0 );
	m_FocusLast->SetValue(true);
	itemGridBagSizer3->Add(m_FocusLast, wxGBPosition(10, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_BtnStopGet = new wxButton( itemPanel2, ID_BUTTON1, wxGetTranslation(wxString() + (wxChar) 0x505C + (wxChar) 0x6B62 + (wxChar) 0x63A5 + (wxChar) 0x6536), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(m_BtnStopGet, wxGBPosition(2, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_Browse = new wxFilePickerCtrl( itemPanel2, ID_FILECTRL, wxEmptyString, wxGetTranslation(wxString() + (wxChar) 0x6A94 + (wxChar) 0x6848 + (wxChar) 0x8DEF + (wxChar) 0x5F91), wxEmptyString, wxDefaultPosition, wxDefaultSize, wxFLP_USE_TEXTCTRL|wxFLP_SAVE );
	itemGridBagSizer3->Add(m_Browse, wxGBPosition(12, 0), wxGBSpan(1, 2), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText20 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x8CC7 + (wxChar) 0x6599 + (wxChar) 0x500B + (wxChar) 0x6578), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText20, wxGBPosition(3, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText21 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x7D93 + (wxChar) 0x5EA6 + wxT("E")), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText21, wxGBPosition(5, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_WaterDepth = new wxStaticText( itemPanel2, wxID_STATIC, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(m_WaterDepth, wxGBPosition(4, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText23 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6C34 + (wxChar) 0x6DF1), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText23, wxGBPosition(4, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_Longitude = new wxStaticText( itemPanel2, wxID_STATIC, _("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(m_Longitude, wxGBPosition(5, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText25 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x7DEF + (wxChar) 0x5EA6 + wxT("N")), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText25, wxGBPosition(6, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_Latitude = new wxStaticText( itemPanel2, wxID_STATIC, _("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(m_Latitude, wxGBPosition(6, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_DataTotal = new wxStaticText( itemPanel2, wxID_STATIC, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(m_DataTotal, wxGBPosition(3, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText28 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6A94 + (wxChar) 0x6848 + (wxChar) 0x8DEF + (wxChar) 0x5F91), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText28, wxGBPosition(11, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText29 = new wxStaticText( itemPanel2, wxID_STATIC, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText29, wxGBPosition(16, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_deColor = new wxColourPickerCtrl( itemPanel2, ID_COLOURCTRL, wxColour(222, 113, 39), wxDefaultPosition, wxDefaultSize, wxCLRP_DEFAULT_STYLE );
	itemGridBagSizer3->Add(m_deColor, wxGBPosition(7, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_hsColor = new wxColourPickerCtrl( itemPanel2, ID_COLOURPICKERCTRL, wxColour(49, 249, 169), wxDefaultPosition, wxDefaultSize, wxCLRP_DEFAULT_STYLE );
	itemGridBagSizer3->Add(m_hsColor, wxGBPosition(8, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText32 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6DF1 + (wxChar) 0x5EA6 + (wxChar) 0x984F + (wxChar) 0x8272), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText32, wxGBPosition(7, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxStaticText* itemStaticText33 = new wxStaticText( itemPanel2, wxID_STATIC, wxGetTranslation(wxString() + (wxChar) 0x6C34 + (wxChar) 0x9762 + (wxChar) 0x984F + (wxChar) 0x8272), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemStaticText33, wxGBPosition(8, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	m_UpdateInterval = new wxSpinCtrl( itemPanel2, ID_SPINCTRL1, _T("500"), wxDefaultPosition, wxSize(80, -1), wxSP_ARROW_KEYS, 100, 5000, 500 );
	itemGridBagSizer3->Add(m_UpdateInterval, wxGBPosition(14, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxButton* itemButton35 = new wxButton( itemPanel2, ID_BUTTON2, wxGetTranslation(wxString() + (wxChar) 0x8F09 + (wxChar) 0x5165 + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemButton35, wxGBPosition(13, 0), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	wxButton* itemButton36 = new wxButton( itemPanel2, ID_BUTTON3, wxGetTranslation(wxString() + (wxChar) 0x6E05 + (wxChar) 0x7A7A + (wxChar) 0x8CC7 + (wxChar) 0x6599), wxDefaultPosition, wxDefaultSize, 0 );
	itemGridBagSizer3->Add(itemButton36, wxGBPosition(13, 1), wxGBSpan(1, 1), wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5);

	// Fit to content
	itemFrame1->GetAuiManager().GetPane(_T("Pane1")).BestSize(itemPanel2->GetSizer()->Fit(itemPanel2)).MinSize(itemPanel2->GetSizer()->GetMinSize());

	m_GLCanvas = new wxGLCanvas( itemFrame1, ID_GLCANVAS, wxDefaultPosition, wxSize(570, 420), 0 );
	itemFrame1->GetAuiManager().AddPane(m_GLCanvas, wxAuiPaneInfo()
		.Name(_T("ID_GLCANVAS")).Caption(_("DrawView")).Centre().BestSize(wxSize(500, 500)).Row(1).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingSize(wxSize(800, 600)).PinButton(true));

	m_OutputText = new wxTextCtrl( itemFrame1, ID_TEXTCTRL, wxEmptyString, wxDefaultPosition, wxSize(780, 100), wxTE_MULTILINE );
	itemFrame1->GetAuiManager().AddPane(m_OutputText, wxAuiPaneInfo()
		.Name(_T("ID_TEXTCTRL")).Caption(_("InputText")).Bottom().BestSize(wxSize(600, 100)).Row(1).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingSize(wxSize(800, 400)).Hide().PinButton(true));

	GetAuiManager().Update();

	// Connect events and objects
	m_GLCanvas->Connect(ID_GLCANVAS, wxEVT_SIZE, wxSizeEventHandler(mainframe::OnCanvasSize), NULL, this);
	m_GLCanvas->Connect(ID_GLCANVAS, wxEVT_PAINT, wxPaintEventHandler(mainframe::OnPaint), NULL, this);
////@end mainframe content construction
	m_DrawView.SetPointSize(m_point_size->GetValue());
	SetTimer ( (HWND)(m_GLCanvas->GetHandle()), CHECK_NORMAL_TIMER, 1000, TimerProc2 );
	mf = this;
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

VOID CALLBACK TimerProc ( HWND hParent, UINT uMsg, UINT uEventID, DWORD dwTimer )
{
	assert(mf->m_timer_go);
	if (mf->m_open)
	{
		char buffer[4096];
		memset(buffer, 0, 4096);
		PollComport(mf->m_port-1, (unsigned char*)buffer, 4096);
		mf->m_nCell.InputRawData(buffer, 4096);
		mf->m_OutputText->AppendText(wxString::FromAscii(buffer));
		while (mf->m_nCell.GetOneIndex()+1 < mf->m_nCell.GetTotal())
		{
			mf->m_DrawView.AddData(mf->m_nCell.GetOne());
			if (mf->m_Browse->CheckPath(mf->m_Browse->GetTextCtrlValue()))
				mf->m_nCell.SaveFile(mf->m_Browse->GetTextCtrlValue().wc_str());
		}
		mf->UpdateDataToUI();
	}
	mf->RenderFrame();
}

/*
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON
 */

void mainframe::OnStartGetClick( wxCommandEvent& event )
{
	ColorLine();
	if (m_open)
	{
		CloseComport(m_port-1);
		m_open = false;
	}
	long boundRate = 38400;
	m_Combo_ComPort->GetString(m_Combo_ComPort->GetSelection()).ToLong(&m_port);
	m_BoundRate->GetString(m_BoundRate->GetSelection()).ToLong(&boundRate);
	wxString mess;
	if (OpenComport(m_port-1, boundRate))
	{
		mess << wxT("打開失敗 boundRate:") << boundRate << wxT(" comport:") << m_port;
		wxMessageDialog dialog(NULL, mess, wxT("注意！"));
		dialog.ShowModal();
	}
	else
	{
		mess << wxT("打開成功 boundRate:") << boundRate << wxT(" comport:") << m_port;
		wxMessageDialog dialog(NULL, mess, wxT("注意！"));
		dialog.ShowModal();
		m_open = true;
		if (!m_timer_go)
		{
			m_timer_go = true;
			SetTimer ( (HWND)(m_GLCanvas->GetHandle()), DRAW_TIMER, m_UpdateInterval->GetValue(), TimerProc );
			m_lastUpdateTime = m_UpdateInterval->GetValue();
		}
		if (m_Browse->GetTextCtrlValue().size() == 0 || !m_Browse->CheckPath(m_Browse->GetTextCtrlValue()))
		{
			wxString mess;
			mess << wxT("你沒有設定正確的儲存路徑!!!");
			wxMessageDialog dialog(NULL, mess, wxT("注意！"));
			dialog.ShowModal();
		}
	}
}

/*
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON1
 */

void mainframe::OnStopGetClick( wxCommandEvent& event )
{
	if (m_open)
	{
		if (m_timer_go)
		{
			m_timer_go = false;
			KillTimer((HWND)(m_GLCanvas->GetHandle()), DRAW_TIMER); 
		}
		CloseComport(m_port-1);
		m_open = false;
		wxString mess;
		mess << wxT("關閉成功 comport:") << m_port;
		wxMessageDialog dialog(NULL, mess, wxT("關閉comport"));
		dialog.ShowModal();
	}
	m_DrawView.AddTest();
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

void mainframe::OnOutputTextVisable( wxCommandEvent& event )
{
	if (m_CanOutput->GetValue())
		GetAuiManager().GetPane(wxT("ID_TEXTCTRL")).Show();
	else
		GetAuiManager().GetPane(wxT("ID_TEXTCTRL")).Hide();
	GetAuiManager().Update();
	event.Skip();
}

/*
 * wxEVT_PAINT event handler for ID_GLCANVAS
 */

void mainframe::OnPaint( wxPaintEvent& event )
{
	wxPaintDC dc(wxDynamicCast(event.GetEventObject(), wxWindow));
	int x, y;
	m_GLCanvas->GetSize(&x, &y);
	m_DrawView.ReSize(x, y);
	RenderFrame();
}

void mainframe::RenderFrame()
{
	m_DrawView.SetHwnd((HWND)m_GLCanvas->GetHandle());
	if (m_FocusLast->GetValue())
		m_DrawView.FocusLast();
	if (m_NormalLook->GetValue())
		m_DrawView.NormalLook();
	m_DrawView.Render();
}


/*
 * wxEVT_SIZE event handler for ID_MAINFRAME
 */

void mainframe::OnCanvasSize( wxSizeEvent& event )
{
	m_DrawView.ReSize(event.GetSize().GetWidth(),event.GetSize().GetHeight());
	event.Skip(); 
}


/*
 * wxEVT_COMMAND_SPINCTRL_UPDATED event handler for ID_SPINCTRL
 */

void mainframe::OnSpinctrlUpdated( wxSpinEvent& event )
{
	m_DrawView.SetFocusHeight(m_spinctrl_height->GetValue());
	event.Skip();
}


/*
 * wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX2
 */

void mainframe::OnCheckbox2Click( wxCommandEvent& event )
{
////@begin wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX2 in mainframe.
	// Before editing this code, remove the block markers.
	event.Skip();
////@end wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX2 in mainframe. 
}


/*
 * wxEVT_COLOURPICKER_CHANGED event handler for ID_COLOURCTRL
 */

void mainframe::OnDeepColorChanged( wxColourPickerEvent& event )
{
	wxColour color = event.GetColour();
	m_DrawView.SetDEColor(color.Red(), color.Green(), color.Blue());
	event.Skip();
}


/*
 * wxEVT_COLOURPICKER_CHANGED event handler for ID_COLOURPICKERCTRL
 */

void mainframe::OnhsColorChanged( wxColourPickerEvent& event )
{
	wxColour color = event.GetColour();
	m_DrawView.SetHSColor(color.Red(), color.Green(), color.Blue());
	event.Skip();
}


/*
 * wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL
 */

void mainframe::OnTimerUpdated( wxCommandEvent& event )
{
	if (m_timer_go && m_lastUpdateTime != m_UpdateInterval->GetValue())
	{
		m_timer_go = false;
		KillTimer((HWND)(m_GLCanvas->GetHandle()), DRAW_TIMER);
		SetTimer ((HWND)(m_GLCanvas->GetHandle()), DRAW_TIMER, m_UpdateInterval->GetValue(), TimerProc);
		m_timer_go = true;
		m_lastUpdateTime = m_UpdateInterval->GetValue();
	}
	event.Skip();
}


/*
 * wxEVT_COMMAND_SPINCTRL_UPDATED event handler for ID_SPINCTRL2
 */

void mainframe::OnPointSizeUpdated( wxSpinEvent& event )
{
	m_DrawView.SetPointSize(m_point_size->GetValue());
	event.Skip();
}


/*
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON2
 */

void mainframe::OnLoadFileDataClick( wxCommandEvent& event )
{
	ColorLine();
	std::ifstream is;
	is.open(m_Browse->GetTextCtrlValue().wc_str());
	if (is)
	{
		char buffer[4096];
		while (!is.eof())
		{
			memset(buffer, 0, 4096);
			is.read(buffer, 4096);
			m_nCell.InputRawData(buffer, 4096);
		}
		while (m_nCell.GetOneIndex()+1 < m_nCell.GetTotal())
		{
			m_DrawView.AddData(m_nCell.GetOne());
		}
		m_nCell.m_buffer_str = "";
	}
	else
	{
		wxMessageDialog dialog(NULL, wxT("讀檔失敗！"), wxT("注意！"));
		dialog.ShowModal();
	}
	UpdateDataToUI();
	is.close();
	event.Skip();
}

/*
 * wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON3
 */

void mainframe::OnClearDataClick( wxCommandEvent& event )
{
	m_DrawView.Clear();
	m_nCell.Clear();
	ColorLine();
	m_Latitude->SetLabel(wxT("0"));
	m_Longitude->SetLabel(wxT("0"));
	m_WaterDepth->SetLabel(wxT("0米"));
	m_DataTotal->SetLabel(wxT("0筆"));
	m_DrawView.SetPointSize(m_point_size->GetValue());
	event.Skip();
}

void mainframe::ColorLine()
{
	wxColour color = m_hsColor->GetColour();
	m_DrawView.SetHSColor(color.Red(), color.Green(), color.Blue());
	color = m_deColor->GetColour();
	m_DrawView.SetDEColor(color.Red(), color.Green(), color.Blue());
}

void mainframe::UpdateDataToUI()
{
	// 更新lat
	if (m_nCell.GetLastData().lat != 0)
	{
		wxString mess;
		mess << m_nCell.GetLastData().lat;
		m_Latitude->SetLabel(mess);
	}
	// 更新lon
	if (m_nCell.GetLastData().lon != 0)
	{
		wxString mess;
		mess << m_nCell.GetLastData().lon;
		m_Longitude->SetLabel(mess);
	}
	// 更新深度
	if (m_nCell.GetLastData().depthinfo.depth_M != 0)
	{
		wxString mess;
		mess << m_nCell.GetLastData().depthinfo.depth_M << wxT("米");
		m_WaterDepth->SetLabel(mess);
	}
	if (m_nCell.GetTotal() != 0)
	{
		wxString mess;
		mess << m_nCell.GetTotal() << wxT("筆");
		m_DataTotal->SetLabel(mess);
	}
	m_DrawView.SetPointSize(m_point_size->GetValue());
}

