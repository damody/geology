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

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON8
    void OnModifyData( wxCommandEvent& event );

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON10
    void OnShowResult( wxCommandEvent& event );

    /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX
    void OnShowInfo( wxCommandEvent& event );

    /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX3
    void OnCheckShowWellInfo( wxCommandEvent& event );

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
    wxSlider* m_ruler_slider;
    wxSpinCtrl* m_ruler_spinctrl;
    wxPanel* m_area_infopanel;
    wxPanel* m_well_infopanel;
////@end Taiwan member variables
private:
	FirstMain* m_FirstMain;
	DataModifyWindow* m_DataModifyWindow;
};

#endif
    // _TAIWAN_H_
