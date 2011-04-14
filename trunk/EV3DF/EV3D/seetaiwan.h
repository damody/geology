/////////////////////////////////////////////////////////////////////////////
// Name:        seetaiwan.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     11/04/2011 16:17:51
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _SEETAIWAN_H_
#define _SEETAIWAN_H_


/*!
 * Includes
 */

////@begin includes
#include "wx/frame.h"
#include "wx/gbsizer.h"
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
#define ID_SEETAIWAN 10004
#define ID_BUTTON2 10020
#define ID_BUTTON1 10021
#define ID_BUTTON3 10022
#define ID_BUTTON4 10023
#define ID_BUTTON5 10024
#define SYMBOL_SEETAIWAN_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX
#define SYMBOL_SEETAIWAN_TITLE _("SeeTaiwan")
#define SYMBOL_SEETAIWAN_IDNAME ID_SEETAIWAN
#define SYMBOL_SEETAIWAN_SIZE wxSize(400, 300)
#define SYMBOL_SEETAIWAN_POSITION wxDefaultPosition
////@end control identifiers


/*!
 * SeeTaiwan class declaration
 */

class SeeTaiwan: public wxFrame
{    
    DECLARE_CLASS( SeeTaiwan )
    DECLARE_EVENT_TABLE()

public:
    /// Constructors
    SeeTaiwan();
    SeeTaiwan( wxWindow* parent, wxWindowID id = SYMBOL_SEETAIWAN_IDNAME, const wxString& caption = SYMBOL_SEETAIWAN_TITLE, const wxPoint& pos = SYMBOL_SEETAIWAN_POSITION, const wxSize& size = SYMBOL_SEETAIWAN_SIZE, long style = SYMBOL_SEETAIWAN_STYLE );

    bool Create( wxWindow* parent, wxWindowID id = SYMBOL_SEETAIWAN_IDNAME, const wxString& caption = SYMBOL_SEETAIWAN_TITLE, const wxPoint& pos = SYMBOL_SEETAIWAN_POSITION, const wxSize& size = SYMBOL_SEETAIWAN_SIZE, long style = SYMBOL_SEETAIWAN_STYLE );

    /// Destructor
    ~SeeTaiwan();

    /// Initialises member variables
    void Init();

    /// Creates the controls and sizers
    void CreateControls();

////@begin SeeTaiwan event handler declarations

////@end SeeTaiwan event handler declarations

////@begin SeeTaiwan member function declarations

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end SeeTaiwan member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();

////@begin SeeTaiwan member variables
////@end SeeTaiwan member variables
};

#endif
    // _SEETAIWAN_H_
