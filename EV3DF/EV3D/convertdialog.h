/////////////////////////////////////////////////////////////////////////////
// Name:        convertdialog.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     24/02/2011 06:38:39
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _CONVERTDIALOG_H_
#define _CONVERTDIALOG_H_


/*!
 * Includes
 */

////@begin includes
#include "wx/gbsizer.h"
#include "wx/filepicker.h"
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
#define ID_CONVERTDIALOG 10007
#define ID_TEXTCTRL 10011
#define ID_TEXTCTRL1 10012
#define ID_TEXTCTRL2 10013
#define ID_TEXTCTRL3 10014
#define ID_TEXTCTRL4 10015
#define ID_TEXTCTRL5 10016
#define ID_TEXTCTRL6 10036
#define ID_TEXTCTRL7 10037
#define ID_TEXTCTRL8 10038
#define ID_TEXTCTRL9 10039
#define ID_TEXTCTRL10 10040
#define ID_TEXTCTRL11 10041
#define ID_TEXTCTRL12 10042
#define ID_TEXTCTRL13 10048
#define ID_TEXTCTRL14 10049
#define ID_BUTTON 10050
#define ID_BUTTON1 10057
#define ID_FILECTRL 10058
#define ID_FILEPICKERCTRL 10059
#define ID_TEXTCTRL15 10063
#define ID_TEXTCTRL16 10064
#define ID_TEXTCTRL17 10066
#define SYMBOL_CONVERTDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_CONVERTDIALOG_TITLE _("ConvertDialog")
#define SYMBOL_CONVERTDIALOG_IDNAME ID_CONVERTDIALOG
#define SYMBOL_CONVERTDIALOG_SIZE wxSize(400, 400)
#define SYMBOL_CONVERTDIALOG_POSITION wxDefaultPosition
////@end control identifiers


/*!
 * ConvertDialog class declaration
 */

class ConvertDialog: public wxDialog
{    
    DECLARE_DYNAMIC_CLASS( ConvertDialog )
    DECLARE_EVENT_TABLE()

public:
    /// Constructors
    ConvertDialog();
    ConvertDialog( wxWindow* parent, wxWindowID id = SYMBOL_CONVERTDIALOG_IDNAME, const wxString& caption = SYMBOL_CONVERTDIALOG_TITLE, const wxPoint& pos = SYMBOL_CONVERTDIALOG_POSITION, const wxSize& size = SYMBOL_CONVERTDIALOG_SIZE, long style = SYMBOL_CONVERTDIALOG_STYLE );

    /// Creation
    bool Create( wxWindow* parent, wxWindowID id = SYMBOL_CONVERTDIALOG_IDNAME, const wxString& caption = SYMBOL_CONVERTDIALOG_TITLE, const wxPoint& pos = SYMBOL_CONVERTDIALOG_POSITION, const wxSize& size = SYMBOL_CONVERTDIALOG_SIZE, long style = SYMBOL_CONVERTDIALOG_STYLE );

    /// Destructor
    ~ConvertDialog();

    /// Initialises member variables
    void Init();

    /// Creates the controls and sizers
    void CreateControls();

////@begin ConvertDialog event handler declarations

////@end ConvertDialog event handler declarations

////@begin ConvertDialog member function declarations

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end ConvertDialog member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();

////@begin ConvertDialog member variables
////@end ConvertDialog member variables
};

#endif
    // _CONVERTDIALOG_H_
