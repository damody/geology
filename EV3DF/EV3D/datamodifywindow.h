/////////////////////////////////////////////////////////////////////////////
// Name:        datamodifywindow.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     20/04/2011 03:26:56
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _DATAMODIFYWINDOW_H_
#define _DATAMODIFYWINDOW_H_


/*!
 * Includes
 */

////@begin includes
#include "wx/frame.h"
#include "wx/gbsizer.h"
#include "wx/grid.h"
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
#define ID_DATAMODIFYWINDOW 10082
#define ID_BUTTON4 10083
#define ID_BUTTON11 10084
#define ID_GRID1 10085
#define SYMBOL_DATAMODIFYWINDOW_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX
#define SYMBOL_DATAMODIFYWINDOW_TITLE _("DataModifyWindow")
#define SYMBOL_DATAMODIFYWINDOW_IDNAME ID_DATAMODIFYWINDOW
#define SYMBOL_DATAMODIFYWINDOW_SIZE wxSize(400, 300)
#define SYMBOL_DATAMODIFYWINDOW_POSITION wxDefaultPosition
////@end control identifiers


/*!
 * DataModifyWindow class declaration
 */

class DataModifyWindow: public wxFrame
{    
    DECLARE_CLASS( DataModifyWindow )
    DECLARE_EVENT_TABLE()

public:
    /// Constructors
    DataModifyWindow();
    DataModifyWindow( wxWindow* parent, wxWindowID id = SYMBOL_DATAMODIFYWINDOW_IDNAME, const wxString& caption = SYMBOL_DATAMODIFYWINDOW_TITLE, const wxPoint& pos = SYMBOL_DATAMODIFYWINDOW_POSITION, const wxSize& size = SYMBOL_DATAMODIFYWINDOW_SIZE, long style = SYMBOL_DATAMODIFYWINDOW_STYLE );

    bool Create( wxWindow* parent, wxWindowID id = SYMBOL_DATAMODIFYWINDOW_IDNAME, const wxString& caption = SYMBOL_DATAMODIFYWINDOW_TITLE, const wxPoint& pos = SYMBOL_DATAMODIFYWINDOW_POSITION, const wxSize& size = SYMBOL_DATAMODIFYWINDOW_SIZE, long style = SYMBOL_DATAMODIFYWINDOW_STYLE );

    /// Destructor
    ~DataModifyWindow();

    /// Initialises member variables
    void Init();

    /// Creates the controls and sizers
    void CreateControls();

////@begin DataModifyWindow event handler declarations

////@end DataModifyWindow event handler declarations

////@begin DataModifyWindow member function declarations

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end DataModifyWindow member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();

////@begin DataModifyWindow member variables
////@end DataModifyWindow member variables
};

#endif
    // _DATAMODIFYWINDOW_H_
