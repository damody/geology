﻿// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
// Created:     24/02/2011 06:38:39


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
class wxFilePickerCtrl;
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
#define ID_CLOSE 10050
#define ID_CONVERT 10057
#define ID_FILECTRL_INPUT 10058
#define ID_BUTTON2 10019
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

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_CLOSE
    void OnCloseClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_CONVERT
    void OnConvertClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_BUTTON_CLICKED event handler for ID_BUTTON2
    void OnLoadClick( wxCommandEvent& event );

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
    wxTextCtrl* m_in_xmin;
    wxTextCtrl* m_in_ymin;
    wxTextCtrl* m_in_zmin;
    wxTextCtrl* m_in_xmax;
    wxTextCtrl* m_in_ymax;
    wxTextCtrl* m_in_zmax;
    wxTextCtrl* m_out_xmin;
    wxTextCtrl* m_out_ymin;
    wxTextCtrl* m_out_zmin;
    wxTextCtrl* m_out_xmax;
    wxTextCtrl* m_out_ymax;
    wxTextCtrl* m_out_zmax;
    wxTextCtrl* m_out_xinterval;
    wxTextCtrl* m_out_yinterval;
    wxTextCtrl* m_out_zinterval;
    wxFilePickerCtrl* m_filectrl_input;
    wxFilePickerCtrl* m_filectrl_output;
    wxTextCtrl* m_in_datatotal;
    wxTextCtrl* m_out_datatotal;
////@end ConvertDialog member variables
};

#endif
    // _CONVERTDIALOG_H_
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
