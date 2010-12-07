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
#include "wx/frame.h"
#include "wx/gbsizer.h"
#include "wx/filepicker.h"
#include "wx/glcanvas.h"
////@end includes

/*!
 * Forward declarations
 */

////@begin forward declarations
class wxFilePickerCtrl;
class wxGLCanvas;
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_MAINFRAME 10000
#define ID_CHOICE1 10002
#define ID_CHOICE 10001
#define ID_BUTTON 10003
#define ID_TEXTCTRL 10008
#define ID_BUTTON1 10004
#define ID_FILECTRL 10007
#define ID_GLCANVAS 10009
#define SYMBOL_MAINFRAME_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX
#define SYMBOL_MAINFRAME_TITLE _("fishfinding")
#define SYMBOL_MAINFRAME_IDNAME ID_MAINFRAME
#define SYMBOL_MAINFRAME_SIZE wxSize(800, 600)
#define SYMBOL_MAINFRAME_POSITION wxDefaultPosition
////@end control identifiers


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
    void OnButtonClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON1
    void OnButton1Click( wxCommandEvent& event );

    /// wxEVT_FILEPICKER_CHANGED event handler for ID_FILECTRL
    void OnFilectrlFilePickerChanged( wxFileDirPickerEvent& event );

////@end mainframe event handler declarations

////@begin mainframe member function declarations

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end mainframe member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();

////@begin mainframe member variables
    wxChoice* m_BoundRate;
    wxChoice* m_Combo_ComPort;
    wxButton* m_BtnStartGet;
    wxTextCtrl* m_OutputText;
    wxButton* m_BtnStopGet;
    wxFilePickerCtrl* m_Browse;
    wxGLCanvas* m_GLCanvas;
////@end mainframe member variables
};

#endif
    // _MAINFRAME_H_
