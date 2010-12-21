/////////////////////////////////////////////////////////////////////////////
// Name:        mainframe.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     19/11/2010 20:44:33
// RCS-ID:      
// Copyright:   ntust
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _MAINFRAME_H_
#define _MAINFRAME_H_


/*!
* Includes
*/

////@begin includes
#include "wx/aui/framemanager.h"
#include "wx/frame.h"
#include "wx/gbsizer.h"
#include "wx/spinctrl.h"
#include "wx/filepicker.h"
#include "wx/clrpicker.h"
#include "wx/glcanvas.h"
////@end includes
#include "NmeaCell.h"
#include "DrawView.h"
/*!
* Forward declarations
*/

////@begin forward declarations
class wxSpinCtrl;
class wxFilePickerCtrl;
class wxColourPickerCtrl;
class wxGLCanvas;
////@end forward declarations

/*!
* Control identifiers
*/

////@begin control identifiers
#define ID_MAINFRAME 10000
#define ID_PANEL 10006
#define ID_CHOICE1 10002
#define ID_CHOICE 10001
#define ID_BUTTON 10003
#define ID_CHECKBOX 10005
#define ID_SPINCTRL2 10016
#define ID_CHECKBOX2 10012
#define ID_SPINCTRL 10011
#define ID_CHECKBOX1 10010
#define ID_BUTTON1 10004
#define ID_FILECTRL 10007
#define ID_COLOURCTRL 10013
#define ID_COLOURPICKERCTRL 10014
#define ID_SPINCTRL1 10015
#define ID_BUTTON2 10017
#define ID_BUTTON3 10018
#define ID_GLCANVAS 10009
#define ID_TEXTCTRL 10008
#define SYMBOL_MAINFRAME_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxMINIMIZE_BOX|wxMAXIMIZE_BOX|wxCLOSE_BOX
#define SYMBOL_MAINFRAME_TITLE _("Bottom plotter 1.3")
#define SYMBOL_MAINFRAME_IDNAME ID_MAINFRAME
#define SYMBOL_MAINFRAME_SIZE wxSize(800, 600)
#define SYMBOL_MAINFRAME_POSITION wxDefaultPosition
////@end control identifiers

enum
{
	DRAW_TIMER = 1,
	CHECK_NORMAL_TIMER
};
/*!
* mainframe class declaration
*/

class mainframe: public wxFrame
{    
	DECLARE_CLASS( mainframe )
	DECLARE_EVENT_TABLE()

public:
	/// Constructors
	mainframe();
	mainframe( wxWindow* parent, wxWindowID id = SYMBOL_MAINFRAME_IDNAME, const wxString& caption = SYMBOL_MAINFRAME_TITLE, const wxPoint& pos = SYMBOL_MAINFRAME_POSITION, const wxSize& size = SYMBOL_MAINFRAME_SIZE, long style = SYMBOL_MAINFRAME_STYLE );

	bool Create( wxWindow* parent, wxWindowID id = SYMBOL_MAINFRAME_IDNAME, const wxString& caption = SYMBOL_MAINFRAME_TITLE, const wxPoint& pos = SYMBOL_MAINFRAME_POSITION, const wxSize& size = SYMBOL_MAINFRAME_SIZE, long style = SYMBOL_MAINFRAME_STYLE );

	/// Destructor
	~mainframe();

	/// Initialises member variables
	void Init();

	/// Creates the controls and sizers
	void CreateControls();

	////@begin mainframe event handler declarations

	/// wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_CHOICE1
	void OnChoice1Selected( wxCommandEvent& event );

	/// wxEVT_COMMAND_CHOICE_SELECTED event handler for ID_CHOICE
	void OnChoiceSelected( wxCommandEvent& event );

	/// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON
	void OnStartGetClick( wxCommandEvent& event );

	/// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX
	void OnOutputTextVisable( wxCommandEvent& event );

	/// wxEVT_COMMAND_SPINCTRL_UPDATED event handler for ID_SPINCTRL2
	void OnPointSizeUpdated( wxSpinEvent& event );

	/// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX2
	void OnCheckbox2Click( wxCommandEvent& event );

	/// wxEVT_COMMAND_SPINCTRL_UPDATED event handler for ID_SPINCTRL
	void OnSpinctrlUpdated( wxSpinEvent& event );

	/// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON1
	void OnStopGetClick( wxCommandEvent& event );

	/// wxEVT_FILEPICKER_CHANGED event handler for ID_FILECTRL
	void OnFilectrlFilePickerChanged( wxFileDirPickerEvent& event );

	/// wxEVT_COLOURPICKER_CHANGED event handler for ID_COLOURCTRL
	void OnDeepColorChanged( wxColourPickerEvent& event );

	/// wxEVT_COLOURPICKER_CHANGED event handler for ID_COLOURPICKERCTRL
	void OnhsColorChanged( wxColourPickerEvent& event );

	/// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON2
	void OnLoadFileDataClick( wxCommandEvent& event );

	/// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON3
	void OnClearDataClick( wxCommandEvent& event );

	/// wxEVT_SIZE event handler for ID_GLCANVAS
	void OnCanvasSize( wxSizeEvent& event );

	/// wxEVT_PAINT event handler for ID_GLCANVAS
	void OnPaint( wxPaintEvent& event );

	/// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL
	void OnTimerUpdated( wxCommandEvent& event );

	////@end mainframe event handler declarations

	////@begin mainframe member function declarations

	/// Returns the AUI manager object
	wxAuiManager& GetAuiManager() { return m_auiManager; }

	/// Retrieves bitmap resources
	wxBitmap GetBitmapResource( const wxString& name );

	/// Retrieves icon resources
	wxIcon GetIconResource( const wxString& name );
	////@end mainframe member function declarations

	/// Should we show tooltips?
	static bool ShowToolTips();
	void RenderFrame();
	void UpdateDataToUI();
	void ColorLine();
	////@begin mainframe member variables
	wxAuiManager m_auiManager;
	wxChoice* m_BoundRate;
	wxChoice* m_Combo_ComPort;
	wxButton* m_BtnStartGet;
	wxCheckBox* m_CanOutput;
	wxSpinCtrl* m_point_size;
	wxCheckBox* m_NormalLook;
	wxSpinCtrl* m_spinctrl_height;
	wxCheckBox* m_FocusLast;
	wxButton* m_BtnStopGet;
	wxFilePickerCtrl* m_Browse;
	wxStaticText* m_WaterDepth;
	wxStaticText* m_Longitude;
	wxStaticText* m_Latitude;
	wxStaticText* m_DataTotal;
	wxColourPickerCtrl* m_deColor;
	wxColourPickerCtrl* m_hsColor;
	wxSpinCtrl* m_UpdateInterval;
	wxGLCanvas* m_GLCanvas;
	wxTextCtrl* m_OutputText;
	////@end mainframe member variables
	long		m_port;
	bool		m_open,
			m_timer_go;
	NmeaCell	m_nCell;
	DrawView	m_DrawView;
	int		m_lastUpdateTime;
};

#endif
// _MAINFRAME_H_
