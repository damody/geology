/////////////////////////////////////////////////////////////////////////////
// Name:        taiwan.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     20/04/2011 02:44:01
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _TAIWAN_H_
#define _TAIWAN_H_
#include "DW/SolidCtrl.h"
#include "DW/SolidView.h"
#include "DW/SolidDoc.h"
#include "DW/SelectionSphere.h"
#include "DW/SelctionBounding.h"
#include "DW/Interpolation/vtkHeatTranslationFilter.h"
SHARE_PTR(SelectionSphere)
SHARE_PTR(SelctionBounding)
VTKSMART_PTR(vtkHeatTranslationFilter)

/*!
 * Includes
 */

////@begin includes
#include "wx/aui/framemanager.h"
#include "wx/frame.h"
#include "wx/gbsizer.h"
#include "wx/spinctrl.h"
#include "wx/glcanvas.h"
#include "wx/scrolbar.h"
#include "wx/aui/auibar.h"
#include "wx/grid.h"
////@end includes
class FirstMain;
class DataModifyWindow;
/*!
 * Forward declarations
 */

////@begin forward declarations
class wxSpinCtrl;
class wxGLCanvas;
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_TAIWAN 10004
#define ID_MENUITEM 10077
#define ID_MENUITEM1 10078
#define ID_MENUITEM2 10079
#define ID_MENUITEM3 10081
#define ID_MENUITEM4 10110
#define ID_MENUITEM5 10111
#define ID_MENUITEM7 10113
#define ID_MENUITEM6 10112
#define ID_MENUITEM8 10114
#define ID_PANEL2 10096
#define ID_BUTTON 10020
#define ID_BUTTON1 10021
#define ID_BUTTON5 10024
#define ID_BUTTON6 10025
#define ID_BUTTON7 10026
#define ID_BUTTON8 10046
#define ID_BUTTON9 10060
#define ID_TEXTCTRL25 10120
#define ID_BUTTON10 10069
#define ID_CHOICE 10089
#define ID_SLIDER 10086
#define ID_CHECKBOX 10076
#define ID_CHOICE1 10092
#define ID_SPINCTRL 10082
#define ID_TEXTCTRL24 10091
#define ID_CHECKBOX5 10118
#define ID_BUTTON3 10109
#define ID_CHECKBOX3 10000
#define ID_LISTBOX 10093
#define ID_BUTTON12 10119
#define ID_TEXTCTRL 10001
#define ID_TEXTCTRL1 10002
#define ID_TEXTCTRL2 10003
#define ID_TEXTCTRL3 10005
#define ID_TEXTCTRL4 10006
#define ID_PANEL3 10097
#define ID_CHECKBOX1 10022
#define ID_CHECKBOX2 10108
#define ID_GLCANVAS1 10080
#define ID_SCROLLBAR 10087
#define ID_GLCANVAS2 10023
#define ID_SCROLLBAR5 10400
#define ID_SCROLLBAR3 10300
#define ID_SCROLLBAR4 10301
#define ID_SCROLLBAR1 10088
#define ID_SCROLLBAR2 10095
#define ID_PANEL1 10090
#define ID_TEXTCTRL18 10070
#define ID_TEXTCTRL19 10071
#define ID_TEXTCTRL20 10072
#define ID_TEXTCTRL21 10073
#define ID_TEXTCTRL22 10074
#define ID_TEXTCTRL23 10075
#define ID_AUITOOLBAR 10094
#define ID_TOOL1 10099
#define ID_TOOL2 10101
#define ID_TEXTCTRL26 10115
#define ID_TOOL 10098
#define ID_TOOL8 10107
#define ID_TOOL3 10102
#define ID_TOOL5 10104
#define ID_TOOL4 10103
#define ID_TOOL6 10105
#define ID_TOOL7 10106
#define ID_PANEL 10116
#define ID_BUTTON4 10083
#define ID_BUTTON11 10084
#define ID_CHECKBOX4 10117
#define ID_GRID1 10085
#define SYMBOL_TAIWAN_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxMINIMIZE_BOX|wxMAXIMIZE_BOX|wxCLOSE_BOX
#define SYMBOL_TAIWAN_TITLE _("Taiwan")
#define SYMBOL_TAIWAN_IDNAME ID_TAIWAN
#define SYMBOL_TAIWAN_SIZE wxSize(1200, 800)
#define SYMBOL_TAIWAN_POSITION wxDefaultPosition
////@end control identifiers


/*!
 * Taiwan class declaration
 */

class Taiwan: public wxFrame
{    
    DECLARE_CLASS( Taiwan )
    DECLARE_EVENT_TABLE()

public:
    /// Constructors
    Taiwan();
    Taiwan( wxWindow* parent, wxWindowID id = SYMBOL_TAIWAN_IDNAME, const wxString& caption = SYMBOL_TAIWAN_TITLE, const wxPoint& pos = SYMBOL_TAIWAN_POSITION, const wxSize& size = SYMBOL_TAIWAN_SIZE, long style = SYMBOL_TAIWAN_STYLE );

    bool Create( wxWindow* parent, wxWindowID id = SYMBOL_TAIWAN_IDNAME, const wxString& caption = SYMBOL_TAIWAN_TITLE, const wxPoint& pos = SYMBOL_TAIWAN_POSITION, const wxSize& size = SYMBOL_TAIWAN_SIZE, long style = SYMBOL_TAIWAN_STYLE );

    /// Destructor
    ~Taiwan();

    /// Initialises member variables
    void Init();

    /// Creates the controls and sizers
    void CreateControls();

////@begin Taiwan event handler declarations

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON
    void OnGetRegionClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON5
    void OnComputeRegionHeatClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON8
    void OnModifyData( wxCommandEvent& event );

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON10
    void OnShowResult( wxCommandEvent& event );

    /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX
    void OnShowInfo( wxCommandEvent& event );

    /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX3
    void OnCheckShowWellInfo( wxCommandEvent& event );

    /// wxEVT_COMMAND_LISTBOX_SELECTED event handler for ID_LISTBOX
    void OnRegionListboxSelected( wxCommandEvent& event );

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON12
    void OnOpenDataClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX1
    void OnCheckboxAxis_Sync( wxCommandEvent& event );

    /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX2
    void OnCheckboxDepth_Sync( wxCommandEvent& event );

    /// wxEVT_SIZE event handler for ID_GLCANVAS1
    void OnCanvasLSize( wxSizeEvent& event );

    /// wxEVT_PAINT event handler for ID_GLCANVAS1
    void OnCanvasLPaint( wxPaintEvent& event );

    /// wxEVT_MOTION event handler for ID_GLCANVAS1
    void OnCanvasLMotion( wxMouseEvent& event );

    /// wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR
    void OnScrollbarVLUpdated( wxCommandEvent& event );

    /// wxEVT_SIZE event handler for ID_GLCANVAS2
    void OnCanvasRSize( wxSizeEvent& event );

    /// wxEVT_PAINT event handler for ID_GLCANVAS2
    void OnCanvasRPaint( wxPaintEvent& event );

    /// wxEVT_MOTION event handler for ID_GLCANVAS2
    void OnCanvasRMotion( wxMouseEvent& event );

    /// wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR5
    void OnScrollbarRLUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR3
    void OnScrollbarHL1Updated( wxCommandEvent& event );

    /// wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR4
    void OnScrollbarHR1Updated( wxCommandEvent& event );

    /// wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR1
    void OnScrollbarHL2Updated( wxCommandEvent& event );

    /// wxEVT_COMMAND_SCROLLBAR_UPDATED event handler for ID_SCROLLBAR2
    void OnScrollbarHR2Updated( wxCommandEvent& event );

////@end Taiwan event handler declarations

////@begin Taiwan member function declarations

    /// Returns the AUI manager object
    wxAuiManager& GetAuiManager() { return m_auiManager; }

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end Taiwan member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();

////@begin Taiwan member variables
    wxAuiManager m_auiManager;
    wxTextCtrl* m_Hv;
    wxSlider* m_ruler_slider;
    wxSpinCtrl* m_ruler_spinctrl;
    wxListBox* m_RegionList;
    wxTextCtrl* m_Tzero;
    wxTextCtrl* m_Rt;
    wxTextCtrl* m_Fppc;
    wxTextCtrl* m_Life;
    wxTextCtrl* m_LimitTemperature;
    wxCheckBox* m_Checkbox_Axis_Sync;
    wxCheckBox* m_Checkbox_Depth_Sync;
    wxGLCanvas* m_CanvasL;
    wxScrollBar* m_ScrollbarVL;
    wxGLCanvas* m_CanvasR;
    wxScrollBar* m_ScrollbarRL;
    wxScrollBar* m_ScrollbarHL1;
    wxScrollBar* m_ScrollbarHR1;
    wxScrollBar* m_ScrollbarHL2;
    wxScrollBar* m_ScrollbarHR2;
    wxPanel* m_area_infopanel;
    wxPanel* m_well_infopanel;
////@end Taiwan member variables
private:
	FirstMain* m_FirstMain;
	DataModifyWindow* m_DataModifyWindow;
	SelctionBounding_Sptr m_selectionBounding;
	
	SolidView_Sptr	m_Volume;
	SolidView_Sptr	m_Terrain;
	SolidView_Sptr	m_ClipPlane;
public:
	SolidCtrl	m_SolidCtrlL;
	SolidCtrl	m_SolidCtrlR;
	SelectionSphere_Sptr m_SelectionSphere;
	bool		m_timego;
	enum	{
		DRAW_TIMER
	};
};

#endif
    // _TAIWAN_H_
