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
#include "wx/frame.h"
#include "wx/gbsizer.h"
#include "wx/combo.h"
#include "wx/glcanvas.h"
#include "wx/scrolbar.h"
////@end includes

/*!
 * Forward declarations
 */

////@begin forward declarations
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_TAIWAN 10004
#define ID_BUTTON 10020
#define ID_BUTTON1 10021
#define ID_BUTTON3 10022
#define ID_BUTTON5 10024
#define ID_BUTTON6 10025
#define ID_BUTTON7 10026
#define ID_BUTTON8 10046
#define ID_BUTTON9 10060
#define ID_TEXTCTRL18 10070
#define ID_TEXTCTRL19 10071
#define ID_TEXTCTRL20 10072
#define ID_TEXTCTRL21 10073
#define ID_TEXTCTRL22 10074
#define ID_TEXTCTRL23 10075
#define ID_BUTTON10 10069
#define ID_COMBOCTRL 10023
#define ID_SLIDER 10086
#define ID_GLCANVAS1 10080
#define ID_SCROLLBAR 10087
#define ID_SCROLLBAR1 10088
#define ID_MENUITEM 10077
#define ID_MENUITEM1 10078
#define ID_MENUITEM2 10079
#define ID_MENUITEM3 10081
#define SYMBOL_TAIWAN_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX
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

////@end Taiwan event handler declarations

////@begin Taiwan member function declarations

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end Taiwan member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();

////@begin Taiwan member variables
////@end Taiwan member variables
};

#endif
    // _TAIWAN_H_
