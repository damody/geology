/////////////////////////////////////////////////////////////////////////////
// Name:        firstmain.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     19/03/2010 13:11:58
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _FIRSTMAIN_H_
#define _FIRSTMAIN_H_
 

/*!
 * Includes
 */
#include "DW/ColorTable.h"
#include "DW/ConvStr.h"
#include "DW/HandleEvr.h"
#include "DW/ConvertToEvr.h"
#include "DW/SolidCtrl.h"
#include "DW/Solid.h"
#include "DW/VarStr.h"
#include "convertdialog.h"
////@begin includes
#include "wx/aui/framemanager.h"
#include "wx/frame.h"
#include "wx/glcanvas.h"
#include "wx/aui/auibar.h"
#include "wx/statusbr.h"
////@end includes

/*!
 * Forward declarations
 */

////@begin forward declarations
class wxGLCanvas;
class wxAuiToolBar;
class ColorGrid;
class MyTreeCtrl;
class MyGrid;
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_FIRSTMAIN 10401
#define ID_GLCANVAS 10017
#define ID_MENUOPENFILE 10043
#define ID_MENUSaveFile 10062
#define ID_MENU_CONVERT_FILE 10045
#define ID_MENUEXIT 10044
#define ID_FileEditToolbar 10052
#define ID_PositionEditToolbar 10053
#define ID_BoundEditToolbar 10056
#define ID_XYZchipEditToolbar 10054
#define ID_ColorTable 10055
#define ID_MENUPreciseToolbar 10061
#define ID_TOOLBAR 10001
#define ID_BTNOPENFILE 10002
#define ID_BTNSAVEFILE 10100
#define ID_BTNCOPY 10003
#define ID_BTNCUT 10005
#define ID_BTNPASTE 10006
#define ID_AUITOOLBAR1 10008
#define ID_LABEL3 10009
#define ID_XminText 10010
#define ID_LABEL4 10018
#define ID_XmaxText 10027
#define ID_LABEL6 10030
#define ID_YminText 10031
#define ID_LABEL5 10028
#define ID_YmaxText 10029
#define ID_LABEL8 10035
#define ID_ZminText 10033
#define ID_LABEL7 10032
#define ID_ZmaxText 10034
#define ID_ShowTypeCombo 10047
#define ID_STATUSBAR 10065
#define SYMBOL_FIRSTMAIN_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxMINIMIZE_BOX|wxMAXIMIZE_BOX|wxCLOSE_BOX
#define SYMBOL_FIRSTMAIN_TITLE _("FirstMain")
#define SYMBOL_FIRSTMAIN_IDNAME ID_FIRSTMAIN
#define SYMBOL_FIRSTMAIN_SIZE wxSize(1024, 768)
#define SYMBOL_FIRSTMAIN_POSITION wxDefaultPosition
////@end control identifiers


/*!
 * FirstMain class declaration
 */

class FirstMain: public wxFrame
{    
    DECLARE_CLASS( FirstMain )
    DECLARE_EVENT_TABLE()
 
public:
    /// Constructors
    FirstMain();
    FirstMain( wxWindow* parent, wxWindowID id = SYMBOL_FIRSTMAIN_IDNAME, const wxString& caption = SYMBOL_FIRSTMAIN_TITLE, const wxPoint& pos = SYMBOL_FIRSTMAIN_POSITION, const wxSize& size = SYMBOL_FIRSTMAIN_SIZE, long style = SYMBOL_FIRSTMAIN_STYLE );

    bool Create( wxWindow* parent, wxWindowID id = SYMBOL_FIRSTMAIN_IDNAME, const wxString& caption = SYMBOL_FIRSTMAIN_TITLE, const wxPoint& pos = SYMBOL_FIRSTMAIN_POSITION, const wxSize& size = SYMBOL_FIRSTMAIN_SIZE, long style = SYMBOL_FIRSTMAIN_STYLE );

    /// Destructor
    ~FirstMain();

    /// Initialises member variables
    void Init();

    /// Creates the controls and sizers
    void CreateControls();

    /// ToOpenFile
    void OpenFile();

    /// ToShowColorList
    void ShowColorTable(ColorTable* ict);

    /// InsertItem to ColorList
    void InsertColor(int i, double val, wxColour& iwc);

////@begin FirstMain event handler declarations

    /// wxEVT_DESTROY event handler for ID_FIRSTMAIN
    void OnDestroy( wxWindowDestroyEvent& event );

    /// wxEVT_SIZE event handler for ID_GLCANVAS
    void OnSize( wxSizeEvent& event );

    /// wxEVT_PAINT event handler for ID_GLCANVAS
    void OnPaint( wxPaintEvent& event );

    /// wxEVT_LEFT_DOWN event handler for ID_GLCANVAS
    void OnLeftDown( wxMouseEvent& event );

    /// wxEVT_LEFT_UP event handler for ID_GLCANVAS
    void OnLeftUp( wxMouseEvent& event );

    /// wxEVT_MIDDLE_DOWN event handler for ID_GLCANVAS
    void OnMiddleDown( wxMouseEvent& event );

    /// wxEVT_MIDDLE_UP event handler for ID_GLCANVAS
    void OnMiddleUp( wxMouseEvent& event );

    /// wxEVT_RIGHT_DOWN event handler for ID_GLCANVAS
    void OnRightDown( wxMouseEvent& event );

    /// wxEVT_RIGHT_UP event handler for ID_GLCANVAS
    void OnRightUp( wxMouseEvent& event );

    /// wxEVT_MOTION event handler for ID_GLCANVAS
    void OnMotion( wxMouseEvent& event );

    /// wxEVT_MOUSEWHEEL event handler for ID_GLCANVAS
    void OnMouseWheel( wxMouseEvent& event );

    /// wxEVT_UPDATE_UI event handler for ID_GLCANVAS
    void OnGlcanvasUpdate( wxUpdateUIEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUOPENFILE
    void OnMenuopenfileClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUSaveFile
    void OnMENUSaveFileClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENU_CONVERT_FILE
    void OnMenuConvertFileClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUEXIT
    void OnMenuexitClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_FileEditToolbar
    void OnFileEditToolbarClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_PositionEditToolbar
    void OnPositionEditToolbarClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_BoundEditToolbar
    void OnBoundEditToolbarClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_XYZchipEditToolbar
    void OnXYZchipEditToolbarClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_ColorTable
    void OnColorTableClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUPreciseToolbar
    void OnMENUPreciseToolbarClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_BTNOPENFILE
    void OnBtnopenfileClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_BTNSAVEFILE
    void OnBtnsavefileClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_BTNCOPY
    void OnBtncopyClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_BTNCUT
    void OnBtncutClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_BTNPASTE
    void OnBtnpasteClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_XminText
    void OnXminTextTextUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_XmaxText
    void OnXmaxTextTextUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_YminText
    void OnYminTextTextUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_YmaxText
    void OnYmaxTextTextUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_ZminText
    void OnZminTextTextUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_ZmaxText
    void OnZmaxTextTextUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_COMBOBOX_SELECTED event handler for ID_ShowTypeCombo
    void OnShowTypeComboSelected( wxCommandEvent& event );

////@end FirstMain event handler declarations

////@begin FirstMain member function declarations

    /// Returns the AUI manager object
    wxAuiManager& GetAuiManager() { return m_auiManager; }

    bool GetInit() const { return m_init ; }
    void SetInit(bool value) { m_init = value ; }

    wxPoint GetLastP() const { return lastP ; }
    void SetLastP(wxPoint value) { lastP = value ; }

    wxPoint GetP() const { return P ; }
    void SetP(wxPoint value) { P = value ; }

    bool GetFixMove() const { return m_bFixMove ; }
    void SetFixMove(bool value) { m_bFixMove = value ; }

    bool GetLoadColor() const { return m_LoadColor ; }
    void SetLoadColor(bool value) { m_LoadColor = value ; }

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end FirstMain member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();

	void RenderFrame();
	void OpenLuaFile(std::wstring str);

////@begin FirstMain member variables
    wxAuiManager m_auiManager;
    wxGLCanvas* itemGLCanvas;
    wxAuiToolBar* m_FileEditToolbar;
    wxAuiToolBar* m_BoundEditToolbar;
    wxTextCtrl* m_XminText;
    wxTextCtrl* m_XmaxText;
    wxTextCtrl* m_YminText;
    wxTextCtrl* m_YmaxText;
    wxTextCtrl* m_ZminText;
    wxTextCtrl* m_ZmaxText;
    wxComboBox* m_ShowTypeCombo;
    MyTreeCtrl* m_treectrl;
    MyGrid* m_grid;
    bool m_init;
    wxPoint lastP;
    wxPoint P;
    bool m_bFixMove;
    bool m_LoadColor;
////@end FirstMain member variables
    private:
	    HandleEvr*		m_hEvr;
	    ConvertToEvr	m_ConvEvr;
	    SJCScalarField3d*	m_psjcF3d;
	    ConvertDialog*	m_convertdialog;
    public:
	    SolidCtrl_Sptr	m_SolidCtrl;
	    SolidView_Sptr	m_ActiveView;
};

#endif    // _FIRSTMAIN_H_
 
