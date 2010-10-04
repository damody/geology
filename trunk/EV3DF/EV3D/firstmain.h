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
#include "DW/Solid.h"
////@begin includes
#include "wx/aui/framemanager.h"
#include "wx/frame.h"
#include "wx/glcanvas.h"
#include "wx/aui/auibar.h"
#include "wx/spinctrl.h"
#include "wx/listctrl.h"
#include "wx/statusbr.h"
#include "wx/notebook.h"
#include "wx/grid.h"
////@end includes

/*!
 * Forward declarations
 */

////@begin forward declarations
class wxGLCanvas;
class wxAuiToolBar;
class wxSpinCtrl;
class wxListCtrl;
class MyTreeCtrl;
class wxGrid;
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_FIRSTMAIN 10000
#define ID_GLCANVAS 10017
#define ID_MENUOPENFILE 10043
#define ID_MENUSaveFile 10062
#define ID_MENUSHOWFILEINFO 10045
#define ID_MENUEDITFILEINFO 10046
#define ID_MENUEXIT 10044
#define ID_FileEditToolbar 10052
#define ID_PositionEditToolbar 10053
#define ID_BoundEditToolbar 10056
#define ID_XYZchipEditToolbar 10054
#define ID_ColorTable 10055
#define ID_MENUPreciseToolbar 10061
#define ID_RenderBondingBox 10071
#define ID_RenderAxis 10019
#define ID_RenderCube 10020
#define ID_RenderFace1 10021
#define ID_RenderFace2 10022
#define ID_RenderFace3 10023
#define ID_RenderFace4 10024
#define ID_RenderFace5 10025
#define ID_RenderFace6 10026
#define ID_RenderXchip 10060
#define ID_TOOLBAR 10001
#define ID_BTNOPENFILE 10002
#define ID_BTNSAVEFILE 10100
#define ID_BTNCOPY 10003
#define ID_BTNCUT 10005
#define ID_BTNPASTE 10006
#define ID_AUITOOLBAR 10011
#define ID_USE_XCHIP 10007
#define ID_LABEL 10012
#define ID_text_chipX 10048
#define ID_SLIDER 10004
#define ID_USE_YCHIP 10069
#define ID_LABEL1 10013
#define ID_text_chipY 10049
#define ID_SLIDER1 10014
#define ID_USE_ZCHIP 10070
#define ID_LABEL2 10015
#define ID_text_chipZ 10050
#define ID_SLIDER2 10016
#define ID_ShowTypeCombo 10047
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
#define ID_AUITOOLBAR2 10040
#define ID_LABEL9 10036
#define ID_MiddleXText 10037
#define ID_LABEL10 10038
#define ID_MiddleYText 10039
#define ID_LABEL11 10041
#define ID_MiddleZText 10042
#define ID_AUITOOLBAR3 10057
#define ID_LABEL12 10058
#define ID_SPINCTRL 10059
#define ID_LABEL13 10063
#define ID_MarchCubeSet_SPINCTRL 10064
#define ID_LISTCTRL 10051
#define ID_STATUSBAR 10065
#define ID_NOTEBOOK 10066
#define ID_GRID 10068
#define SYMBOL_FIRSTMAIN_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxMAXIMIZE|wxMAXIMIZE_BOX|wxCLOSE_BOX
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

    /// wxEVT_CREATE event handler for ID_FIRSTMAIN
    void OnCreate( wxWindowCreateEvent& event );

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

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUSHOWFILEINFO
    void OnMenushowfileinfoClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUEDITFILEINFO
    void OnMenueditfileinfoClick( wxCommandEvent& event );

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

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderBondingBox
    void OnRenderBondingBoxClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderAxis
    void OnRenderAxisClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderCube
    void OnRenderCubeClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace1
    void OnRenderFace1Click( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace2
    void OnRenderFace2Click( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace3
    void OnRenderFace3Click( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace4
    void OnRenderFace4Click( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace5
    void OnRenderFace5Click( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace6
    void OnRenderFace6Click( wxCommandEvent& event );

    /// wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderXchip
    void OnRenderXchipClick( wxCommandEvent& event );

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

    /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_USE_XCHIP
    void OnUseXchipClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_text_chipX
    void OnTextChipXTextUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_SLIDER_UPDATED event handler for ID_SLIDER
    void OnSliderUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_USE_YCHIP
    void OnUseYchipClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_text_chipY
    void OnTextChipYTextUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_SLIDER_UPDATED event handler for ID_SLIDER1
    void OnSlider1Updated( wxCommandEvent& event );

    /// wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_USE_ZCHIP
    void OnUseZchipClick( wxCommandEvent& event );

    /// wxEVT_COMMAND_TEXT_UPDATED event handler for ID_text_chipZ
    void OnTextChipZTextUpdated( wxCommandEvent& event );

    /// wxEVT_COMMAND_SLIDER_UPDATED event handler for ID_SLIDER2
    void OnSlider2Updated( wxCommandEvent& event );

    /// wxEVT_COMMAND_COMBOBOX_SELECTED event handler for ID_ShowTypeCombo
    void OnShowTypeComboSelected( wxCommandEvent& event );

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

    /// wxEVT_COMMAND_SPINCTRL_UPDATED event handler for ID_SPINCTRL
    void OnSpinctrlUpdated( wxSpinEvent& event );

    /// wxEVT_COMMAND_SPINCTRL_UPDATED event handler for ID_MarchCubeSet_SPINCTRL
    void OnMarchCubeSetSPINCTRLUpdated( wxSpinEvent& event );

////@end FirstMain event handler declarations

////@begin FirstMain member function declarations

    /// Returns the AUI manager object
    wxAuiManager& GetAuiManager() { return m_auiManager; }

    bool GetRenderFace1() const { return m_RenderFace1 ; }
    void SetRenderFace1(bool value) { m_RenderFace1 = value ; }

    bool GetRenderFace2() const { return m_RenderFace2 ; }
    void SetRenderFace2(bool value) { m_RenderFace2 = value ; }

    bool GetRenderFace3() const { return m_RenderFace3 ; }
    void SetRenderFace3(bool value) { m_RenderFace3 = value ; }

    bool GetRenderFace4() const { return m_RenderFace4 ; }
    void SetRenderFace4(bool value) { m_RenderFace4 = value ; }

    bool GetRenderFace5() const { return m_RenderFace5 ; }
    void SetRenderFace5(bool value) { m_RenderFace5 = value ; }

    bool GetRenderFace6() const { return m_RenderFace6 ; }
    void SetRenderFace6(bool value) { m_RenderFace6 = value ; }

    bool GetRenderX() const { return m_RenderX ; }
    void SetRenderX(bool value) { m_RenderX = value ; }

    bool GetRenderCube() const { return m_RenderCube ; }
    void SetRenderCube(bool value) { m_RenderCube = value ; }

    bool GetInit() const { return m_init ; }
    void SetInit(bool value) { m_init = value ; }

    wxPoint GetLastP() const { return lastP ; }
    void SetLastP(wxPoint value) { lastP = value ; }

    wxPoint GetP() const { return P ; }
    void SetP(wxPoint value) { P = value ; }

    bool GetRenderAxis() const { return m_RenderAxis ; }
    void SetRenderAxis(bool value) { m_RenderAxis = value ; }

    bool GetUseXYZchip() const { return m_useXchip ; }
    void SetUseXYZchip(bool value) { m_useXchip = value ; }

    bool GetFixMove() const { return m_bFixMove ; }
    void SetFixMove(bool value) { m_bFixMove = value ; }

    bool GetLoadColor() const { return m_LoadColor ; }
    void SetLoadColor(bool value) { m_LoadColor = value ; }

    bool GetUseYchip() const { return m_useYchip ; }
    void SetUseYchip(bool value) { m_useYchip = value ; }

    bool GetUseZchip() const { return m_useZchip ; }
    void SetUseZchip(bool value) { m_useZchip = value ; }

    bool GetRenderBondingBox() const { return m_RenderBondingBox ; }
    void SetRenderBondingBox(bool value) { m_RenderBondingBox = value ; }

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end FirstMain member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();

	void RenderFrame();

////@begin FirstMain member variables
    wxAuiManager m_auiManager;
    wxGLCanvas* itemGLCanvas;
    wxAuiToolBar* m_FileEditToolbar;
    wxAuiToolBar* m_XYZchipEditToolbar;
    wxTextCtrl* m_text_chipX;
    wxSlider* m_XSlider;
    wxTextCtrl* m_text_chipY;
    wxSlider* m_YSlider;
    wxTextCtrl* m_text_chipZ;
    wxSlider* m_ZSlider;
    wxComboBox* m_ShowTypeCombo;
    wxAuiToolBar* m_BoundEditToolbar;
    wxTextCtrl* m_XminText;
    wxTextCtrl* m_XmaxText;
    wxTextCtrl* m_YminText;
    wxTextCtrl* m_YmaxText;
    wxTextCtrl* m_ZminText;
    wxTextCtrl* m_ZmaxText;
    wxAuiToolBar* m_PositionEditToolbar;
    wxTextCtrl* m_MiddleXText;
    wxTextCtrl* m_MiddleYText;
    wxTextCtrl* m_MiddleZText;
    wxSpinCtrl* m_PreciseSpin;
    wxSpinCtrl* m_MarchCubeSet_spinctrl;
    wxListCtrl* m_ColorList;
    MyTreeCtrl* m_treectrl;
    wxGrid* m_grid;
    bool m_RenderFace1;
    bool m_RenderFace2;
    bool m_RenderFace3;
    bool m_RenderFace4;
    bool m_RenderFace5;
    bool m_RenderFace6;
    bool m_RenderX;
    bool m_RenderCube;
    bool m_init;
    wxPoint lastP;
    wxPoint P;
    bool m_RenderAxis;
    bool m_useXchip;
    bool m_bFixMove;
    bool m_LoadColor;
    bool m_useYchip;
    bool m_useZchip;
    bool m_RenderBondingBox;
////@end FirstMain member variables
    private:
	    HandleEvr*	m_hEvr;
	    ConvertToEvr   m_ConvEvr;
	    Solid	m_Solid;
	    SJCScalarField3d* m_psjcF3d;
};

#endif    // _FIRSTMAIN_H_
 
