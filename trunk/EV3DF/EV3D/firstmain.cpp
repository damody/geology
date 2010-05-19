/////////////////////////////////////////////////////////////////////////////
// Name:        firstmain.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     19/03/2010 13:11:58
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////
#include "stdwx.h"
// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif
#define  WX_PRECOMP 1
#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

////@begin includes
#include "wx/imaglist.h"
////@end includes

#include "firstmain.h"

////@begin XPM images
#include "fileopen.xpm"
#include "filesave.xpm"
#include "copy.xpm"
#include "cut.xpm"
#include "paste.xpm"
////@end XPM images
#include <algorithm>

/*
* FirstMain type definition
*/

IMPLEMENT_CLASS( FirstMain, wxFrame )


/*
* FirstMain event table definition
*/

BEGIN_EVENT_TABLE( FirstMain, wxFrame )

////@begin FirstMain event table entries
    EVT_WINDOW_CREATE( FirstMain::OnCreate )
    EVT_WINDOW_DESTROY( FirstMain::OnDestroy )

    EVT_UPDATE_UI( ID_GLCANVAS, FirstMain::OnGlcanvasUpdate )

    EVT_MENU( ID_MENUOPENFILE, FirstMain::OnMenuopenfileClick )

    EVT_MENU( ID_MENUSaveFile, FirstMain::OnMENUSaveFileClick )

    EVT_MENU( ID_MENUSHOWFILEINFO, FirstMain::OnMenushowfileinfoClick )

    EVT_MENU( ID_MENUEDITFILEINFO, FirstMain::OnMenueditfileinfoClick )

    EVT_MENU( ID_MENUEXIT, FirstMain::OnMenuexitClick )

    EVT_MENU( ID_FileEditToolbar, FirstMain::OnFileEditToolbarClick )

    EVT_MENU( ID_PositionEditToolbar, FirstMain::OnPositionEditToolbarClick )

    EVT_MENU( ID_BoundEditToolbar, FirstMain::OnBoundEditToolbarClick )

    EVT_MENU( ID_XYZchipEditToolbar, FirstMain::OnXYZchipEditToolbarClick )

    EVT_MENU( ID_ColorTable, FirstMain::OnColorTableClick )

    EVT_MENU( ID_MENUPreciseToolbar, FirstMain::OnMENUPreciseToolbarClick )

    EVT_MENU( ID_RenderBondingBox, FirstMain::OnRenderBondingBoxClick )

    EVT_MENU( ID_RenderAxis, FirstMain::OnRenderAxisClick )

    EVT_MENU( ID_RenderCube, FirstMain::OnRenderCubeClick )

    EVT_MENU( ID_RenderFace1, FirstMain::OnRenderFace1Click )

    EVT_MENU( ID_RenderFace2, FirstMain::OnRenderFace2Click )

    EVT_MENU( ID_RenderFace3, FirstMain::OnRenderFace3Click )

    EVT_MENU( ID_RenderFace4, FirstMain::OnRenderFace4Click )

    EVT_MENU( ID_RenderFace5, FirstMain::OnRenderFace5Click )

    EVT_MENU( ID_RenderFace6, FirstMain::OnRenderFace6Click )

    EVT_MENU( ID_RenderXchip, FirstMain::OnRenderXchipClick )

    EVT_MENU( ID_BTNOPENFILE, FirstMain::OnBtnopenfileClick )

    EVT_MENU( ID_BTNSAVEFILE, FirstMain::OnBtnsavefileClick )

    EVT_MENU( ID_BTNCOPY, FirstMain::OnBtncopyClick )

    EVT_MENU( ID_BTNCUT, FirstMain::OnBtncutClick )

    EVT_MENU( ID_BTNPASTE, FirstMain::OnBtnpasteClick )

    EVT_CHECKBOX( ID_USE_XCHIP, FirstMain::OnUseXchipClick )

    EVT_TEXT( ID_text_chipX, FirstMain::OnTextChipXTextUpdated )

    EVT_SLIDER( ID_SLIDER, FirstMain::OnSliderUpdated )

    EVT_CHECKBOX( ID_USE_YCHIP, FirstMain::OnUseYchipClick )

    EVT_TEXT( ID_text_chipY, FirstMain::OnTextChipYTextUpdated )

    EVT_SLIDER( ID_SLIDER1, FirstMain::OnSlider1Updated )

    EVT_CHECKBOX( ID_USE_ZCHIP, FirstMain::OnUseZchipClick )

    EVT_TEXT( ID_text_chipZ, FirstMain::OnTextChipZTextUpdated )

    EVT_SLIDER( ID_SLIDER2, FirstMain::OnSlider2Updated )

    EVT_COMBOBOX( ID_ShowTypeCombo, FirstMain::OnShowTypeComboSelected )

    EVT_TEXT( ID_XminText, FirstMain::OnXminTextTextUpdated )

    EVT_TEXT( ID_XmaxText, FirstMain::OnXmaxTextTextUpdated )

    EVT_TEXT( ID_YminText, FirstMain::OnYminTextTextUpdated )

    EVT_TEXT( ID_YmaxText, FirstMain::OnYmaxTextTextUpdated )

    EVT_TEXT( ID_ZminText, FirstMain::OnZminTextTextUpdated )

    EVT_TEXT( ID_ZmaxText, FirstMain::OnZmaxTextTextUpdated )

    EVT_SPINCTRL( ID_SPINCTRL, FirstMain::OnSpinctrlUpdated )

    EVT_SPINCTRL( ID_MarchCubeSet_SPINCTRL, FirstMain::OnMarchCubeSetSPINCTRLUpdated )

////@end FirstMain event table entries

END_EVENT_TABLE()


/*
* FirstMain constructors
*/

FirstMain::FirstMain()
{
	Init();
}

FirstMain::FirstMain( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
	Init();
	Create( parent, id, caption, pos, size, style );
}


/*
* FirstMain creator
*/

bool FirstMain::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
	////@begin FirstMain creation
    wxFrame::Create( parent, id, caption, pos, size, style );

    CreateControls();
    Centre();
	////@end FirstMain creation
	return true;
}


/*
* FirstMain destructor
*/

FirstMain::~FirstMain()
{
	////@begin FirstMain destruction
    GetAuiManager().UnInit();
	////@end FirstMain destruction
	if (m_hEvr)
		delete m_hEvr;
	m_hEvr = NULL;
}


/*
* Member initialisation
*/

void FirstMain::Init()
{
	////@begin FirstMain member initialisation
    m_RenderFace1 = false;
    m_RenderFace2 = false;
    m_RenderFace3 = false;
    m_RenderFace4 = false;
    m_RenderFace5 = false;
    m_RenderFace6 = false;
    m_RenderX = false;
    m_RenderCube = false;
    m_init = false;
    m_RenderAxis = true;
    m_useXchip = false;
    m_bFixMove = false;
    m_LoadColor = false;
    m_useYchip = false;
    m_useZchip = false;
    m_RenderBondingBox = true;
    itemGLCanvas = NULL;
    m_FileEditToolbar = NULL;
    m_XYZchipEditToolbar = NULL;
    m_text_chipX = NULL;
    m_XSlider = NULL;
    m_text_chipY = NULL;
    m_YSlider = NULL;
    m_text_chipZ = NULL;
    m_ZSlider = NULL;
    m_ShowTypeCombo = NULL;
    m_BoundEditToolbar = NULL;
    m_XminText = NULL;
    m_XmaxText = NULL;
    m_YminText = NULL;
    m_YmaxText = NULL;
    m_ZminText = NULL;
    m_ZmaxText = NULL;
    m_PositionEditToolbar = NULL;
    m_MiddleXText = NULL;
    m_MiddleYText = NULL;
    m_MiddleZText = NULL;
    m_PreciseSpin = NULL;
    m_MarchCubeSet_spinctrl = NULL;
    m_ColorList = NULL;
	////@end FirstMain member initialisation
	m_hEvr = NULL;
	m_psjcF3d = NULL;
}


/*
* Control creation for FirstMain
*/

void FirstMain::CreateControls()
{    
	////@begin FirstMain content construction
    FirstMain* itemFrame1 = this;

    GetAuiManager().SetManagedWindow(this);

    wxMenuBar* menuBar = new wxMenuBar;
    wxMenu* itemMenu4 = new wxMenu;
    itemMenu4->Append(ID_MENUOPENFILE, _("OpenFile"), wxEmptyString, wxITEM_NORMAL);
    itemMenu4->Append(ID_MENUSaveFile, _("SaveFile"), wxEmptyString, wxITEM_NORMAL);
    itemMenu4->Append(ID_MENUSHOWFILEINFO, _("ShowFileInfomation"), wxEmptyString, wxITEM_NORMAL);
    itemMenu4->Append(ID_MENUEDITFILEINFO, _("EditFileInformation"), wxEmptyString, wxITEM_NORMAL);
    itemMenu4->Append(ID_MENUEXIT, _("Exit"), wxEmptyString, wxITEM_NORMAL);
    menuBar->Append(itemMenu4, _("File"));
    wxMenu* itemMenu10 = new wxMenu;
    itemMenu10->Append(ID_FileEditToolbar, _("FileEdit Toolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu10->Check(ID_FileEditToolbar, true);
    itemMenu10->Append(ID_PositionEditToolbar, _("Position Edit Toolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu10->Check(ID_PositionEditToolbar, true);
    itemMenu10->Append(ID_BoundEditToolbar, _("Bound Edit Toolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu10->Check(ID_BoundEditToolbar, true);
    itemMenu10->Append(ID_XYZchipEditToolbar, _("XYZchip Edit Toolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu10->Check(ID_XYZchipEditToolbar, true);
    itemMenu10->Append(ID_ColorTable, _("Color Table"), wxEmptyString, wxITEM_CHECK);
    itemMenu10->Check(ID_ColorTable, true);
    itemMenu10->Append(ID_MENUPreciseToolbar, _("PreciseToolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu10->Check(ID_MENUPreciseToolbar, true);
    menuBar->Append(itemMenu10, _("View"));
    wxMenu* itemMenu17 = new wxMenu;
    itemMenu17->Append(ID_RenderBondingBox, _("RenderBondingBox"), wxEmptyString, wxITEM_CHECK);
    itemMenu17->Check(ID_RenderBondingBox, true);
    itemMenu17->Append(ID_RenderAxis, _("RenderAxis"), wxEmptyString, wxITEM_CHECK);
    itemMenu17->Check(ID_RenderAxis, true);
    itemMenu17->Append(ID_RenderCube, _("RenderCube"), wxEmptyString, wxITEM_CHECK);
    itemMenu17->Append(ID_RenderFace1, _("RenderFace1"), wxEmptyString, wxITEM_CHECK);
    itemMenu17->Append(ID_RenderFace2, _("RenderFace2"), wxEmptyString, wxITEM_CHECK);
    itemMenu17->Append(ID_RenderFace3, _("RenderFace3"), wxEmptyString, wxITEM_CHECK);
    itemMenu17->Append(ID_RenderFace4, _("RenderFace4"), wxEmptyString, wxITEM_CHECK);
    itemMenu17->Append(ID_RenderFace5, _("RenderFace5"), wxEmptyString, wxITEM_CHECK);
    itemMenu17->Append(ID_RenderFace6, _("RenderFace6"), wxEmptyString, wxITEM_CHECK);
    itemMenu17->Append(ID_RenderXchip, _("RenderXchip"), wxEmptyString, wxITEM_CHECK);
    menuBar->Append(itemMenu17, _("RenderMethod"));
    itemFrame1->SetMenuBar(menuBar);

    itemGLCanvas = new wxGLCanvas( itemFrame1, ID_GLCANVAS, wxDefaultPosition, wxDefaultSize, 0 );
    itemFrame1->GetAuiManager().AddPane(itemGLCanvas, wxAuiPaneInfo()
        .Name(_T("ID_GLCANVAS")).Caption(_("layout")).Centre().TopDockable(false).BottomDockable(false).CloseButton(false).DestroyOnClose(false).Resizable(true).Floatable(false).FloatingSize(wxSize(800, 600)));

    m_FileEditToolbar = new wxAuiToolBar( itemFrame1, ID_TOOLBAR, wxDefaultPosition, wxDefaultSize, wxAUI_TB_GRIPPER );
    wxBitmap itemtool29Bitmap(itemFrame1->GetBitmapResource(wxT("fileopen.xpm")));
    wxBitmap itemtool29BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNOPENFILE, wxEmptyString, itemtool29Bitmap, itemtool29BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool30Bitmap(itemFrame1->GetBitmapResource(wxT("filesave.xpm")));
    wxBitmap itemtool30BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNSAVEFILE, wxEmptyString, itemtool30Bitmap, itemtool30BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool31Bitmap(itemFrame1->GetBitmapResource(wxT("copy.xpm")));
    wxBitmap itemtool31BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNCOPY, wxEmptyString, itemtool31Bitmap, itemtool31BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool32Bitmap(itemFrame1->GetBitmapResource(wxT("cut.xpm")));
    wxBitmap itemtool32BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNCUT, wxEmptyString, itemtool32Bitmap, itemtool32BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool33Bitmap(itemFrame1->GetBitmapResource(wxT("paste.xpm")));
    wxBitmap itemtool33BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNPASTE, wxEmptyString, itemtool33Bitmap, itemtool33BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    m_FileEditToolbar->Realize();
    itemFrame1->GetAuiManager().AddPane(m_FileEditToolbar, wxAuiPaneInfo()
        .ToolbarPane().Name(_T("FileEditToolbar")).Top().Layer(10).CaptionVisible(false).CloseButton(false).DestroyOnClose(false).Resizable(false).Gripper(true));

    m_XYZchipEditToolbar = new wxAuiToolBar( itemFrame1, ID_AUITOOLBAR, wxDefaultPosition, wxDefaultSize, wxAUI_TB_GRIPPER );
    wxCheckBox* itemCheckBox35 = new wxCheckBox( m_XYZchipEditToolbar, ID_USE_XCHIP, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemCheckBox35->SetValue(false);
    m_XYZchipEditToolbar->AddControl(itemCheckBox35);
    m_XYZchipEditToolbar->AddLabel(ID_LABEL, _("X"), 10);
    m_text_chipX = new wxTextCtrl( m_XYZchipEditToolbar, ID_text_chipX, _("0"), wxDefaultPosition, wxSize(40, -1), 0 );
    m_XYZchipEditToolbar->AddControl(m_text_chipX);
    m_XSlider = new wxSlider( m_XYZchipEditToolbar, ID_SLIDER, 0, 0, 1000, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
    m_XYZchipEditToolbar->AddControl(m_XSlider);
    wxCheckBox* itemCheckBox39 = new wxCheckBox( m_XYZchipEditToolbar, ID_USE_YCHIP, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
    itemCheckBox39->SetValue(false);
    m_XYZchipEditToolbar->AddControl(itemCheckBox39);
    m_XYZchipEditToolbar->AddLabel(ID_LABEL1, _("Y"), 10);
    m_text_chipY = new wxTextCtrl( m_XYZchipEditToolbar, ID_text_chipY, _("0"), wxDefaultPosition, wxSize(40, -1), 0 );
    m_XYZchipEditToolbar->AddControl(m_text_chipY);
    m_YSlider = new wxSlider( m_XYZchipEditToolbar, ID_SLIDER1, 0, 0, 1000, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
    m_XYZchipEditToolbar->AddControl(m_YSlider);
    wxCheckBox* itemCheckBox43 = new wxCheckBox( m_XYZchipEditToolbar, ID_USE_ZCHIP, _("Checkbox"), wxDefaultPosition, wxDefaultSize, 0 );
    itemCheckBox43->SetValue(false);
    m_XYZchipEditToolbar->AddControl(itemCheckBox43);
    m_XYZchipEditToolbar->AddLabel(ID_LABEL2, _("Z"), 10);
    m_text_chipZ = new wxTextCtrl( m_XYZchipEditToolbar, ID_text_chipZ, _("0"), wxDefaultPosition, wxSize(40, -1), 0 );
    m_XYZchipEditToolbar->AddControl(m_text_chipZ);
    m_ZSlider = new wxSlider( m_XYZchipEditToolbar, ID_SLIDER2, 0, 0, 1000, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
    m_XYZchipEditToolbar->AddControl(m_ZSlider);
    wxArrayString m_ShowTypeComboStrings;
    m_ShowTypeCombo = new wxComboBox( m_XYZchipEditToolbar, ID_ShowTypeCombo, wxEmptyString, wxDefaultPosition, wxDefaultSize, m_ShowTypeComboStrings, wxCB_READONLY );
    m_XYZchipEditToolbar->AddControl(m_ShowTypeCombo);
    m_XYZchipEditToolbar->Realize();
    itemFrame1->GetAuiManager().AddPane(m_XYZchipEditToolbar, wxAuiPaneInfo()
        .ToolbarPane().Name(_T("XYZchipEditToolbar")).Top().Row(1).Layer(10).CaptionVisible(false).CloseButton(false).DestroyOnClose(false).Resizable(false).Gripper(true));

    m_BoundEditToolbar = new wxAuiToolBar( itemFrame1, ID_AUITOOLBAR1, wxDefaultPosition, wxDefaultSize, wxAUI_TB_GRIPPER );
    m_BoundEditToolbar->AddLabel(ID_LABEL3, _("Xmin"), 30);
    m_XminText = new wxTextCtrl( m_BoundEditToolbar, ID_XminText, wxEmptyString, wxDefaultPosition, wxSize(50, -1), 0 );
    m_BoundEditToolbar->AddControl(m_XminText);
    m_BoundEditToolbar->AddLabel(ID_LABEL4, _("Xmax"), 30);
    m_XmaxText = new wxTextCtrl( m_BoundEditToolbar, ID_XmaxText, wxEmptyString, wxDefaultPosition, wxSize(50, -1), 0 );
    m_BoundEditToolbar->AddControl(m_XmaxText);
    m_BoundEditToolbar->AddLabel(ID_LABEL6, _("Ymin"), 30);
    m_YminText = new wxTextCtrl( m_BoundEditToolbar, ID_YminText, wxEmptyString, wxDefaultPosition, wxSize(50, -1), 0 );
    m_BoundEditToolbar->AddControl(m_YminText);
    m_BoundEditToolbar->AddLabel(ID_LABEL5, _("Ymax"), 30);
    m_YmaxText = new wxTextCtrl( m_BoundEditToolbar, ID_YmaxText, wxEmptyString, wxDefaultPosition, wxSize(50, -1), 0 );
    m_BoundEditToolbar->AddControl(m_YmaxText);
    m_BoundEditToolbar->AddLabel(ID_LABEL8, _("Zmin"), 30);
    m_ZminText = new wxTextCtrl( m_BoundEditToolbar, ID_ZminText, wxEmptyString, wxDefaultPosition, wxSize(50, -1), 0 );
    m_BoundEditToolbar->AddControl(m_ZminText);
    m_BoundEditToolbar->AddLabel(ID_LABEL7, _("Zmax"), 30);
    m_ZmaxText = new wxTextCtrl( m_BoundEditToolbar, ID_ZmaxText, wxEmptyString, wxDefaultPosition, wxSize(50, -1), 0 );
    m_BoundEditToolbar->AddControl(m_ZmaxText);
    m_BoundEditToolbar->Realize();
    itemFrame1->GetAuiManager().AddPane(m_BoundEditToolbar, wxAuiPaneInfo()
        .ToolbarPane().Name(_T("BoundEditToolbar")).Top().Layer(10).CaptionVisible(false).CloseButton(false).DestroyOnClose(false).Resizable(false).Gripper(true));

    m_PositionEditToolbar = new wxAuiToolBar( itemFrame1, ID_AUITOOLBAR2, wxDefaultPosition, wxDefaultSize, wxAUI_TB_GRIPPER );
    m_PositionEditToolbar->AddLabel(ID_LABEL9, _("MiddleX"), 40);
    m_MiddleXText = new wxTextCtrl( m_PositionEditToolbar, ID_MiddleXText, wxEmptyString, wxDefaultPosition, wxSize(50, -1), 0 );
    m_PositionEditToolbar->AddControl(m_MiddleXText);
    m_PositionEditToolbar->AddLabel(ID_LABEL10, _("MiddleY"), 40);
    m_MiddleYText = new wxTextCtrl( m_PositionEditToolbar, ID_MiddleYText, wxEmptyString, wxDefaultPosition, wxSize(50, -1), 0 );
    m_PositionEditToolbar->AddControl(m_MiddleYText);
    m_PositionEditToolbar->AddLabel(ID_LABEL11, _("MiddleZ"), 40);
    m_MiddleZText = new wxTextCtrl( m_PositionEditToolbar, ID_MiddleZText, wxEmptyString, wxDefaultPosition, wxSize(50, -1), 0 );
    m_PositionEditToolbar->AddControl(m_MiddleZText);
    m_PositionEditToolbar->Realize();
    itemFrame1->GetAuiManager().AddPane(m_PositionEditToolbar, wxAuiPaneInfo()
        .ToolbarPane().Name(_T("PositionEditToolbar")).Top().Layer(10).CaptionVisible(false).CloseButton(false).DestroyOnClose(false).Resizable(false).Gripper(true));

    wxAuiToolBar* itemAuiToolBar68 = new wxAuiToolBar( itemFrame1, ID_AUITOOLBAR3, wxDefaultPosition, wxSize(80, -1), wxAUI_TB_GRIPPER );
    itemAuiToolBar68->AddLabel(ID_LABEL12, _("Precise"), 40);
    m_PreciseSpin = new wxSpinCtrl( itemAuiToolBar68, ID_SPINCTRL, _T("1"), wxDefaultPosition, wxSize(40, -1), wxSP_ARROW_KEYS, 1, 9, 1 );
    itemAuiToolBar68->AddControl(m_PreciseSpin);
    itemAuiToolBar68->AddLabel(ID_LABEL13, _("MarchCubeSet"), 80);
    m_MarchCubeSet_spinctrl = new wxSpinCtrl( itemAuiToolBar68, ID_MarchCubeSet_SPINCTRL, _T("0"), wxDefaultPosition, wxSize(50, -1), wxSP_ARROW_KEYS, 0, 100, 0 );
    itemAuiToolBar68->AddControl(m_MarchCubeSet_spinctrl);
    itemAuiToolBar68->Realize();
    itemFrame1->GetAuiManager().AddPane(itemAuiToolBar68, wxAuiPaneInfo()
        .ToolbarPane().Name(_T("PreciseToolbar")).Top().Row(1).Layer(10).CaptionVisible(false).CloseButton(false).DestroyOnClose(false).Resizable(false).Floatable(false).Gripper(true));

    m_ColorList = new wxListCtrl( itemFrame1, ID_LISTCTRL, wxDefaultPosition, wxDefaultSize, wxLC_REPORT );
    itemFrame1->GetAuiManager().AddPane(m_ColorList, wxAuiPaneInfo()
        .Name(_T("ColorList")).Caption(_("ColorTable")).BestSize(wxSize(200, 200)).CloseButton(false).DestroyOnClose(false).Resizable(false).FloatingSize(wxSize(200, 400)));

    wxStatusBar* itemStatusBar74 = new wxStatusBar( itemFrame1, ID_STATUSBAR, wxST_SIZEGRIP|wxNO_BORDER );
    itemStatusBar74->SetFieldsCount(2);
    itemFrame1->SetStatusBar(itemStatusBar74);

    wxNotebook* itemNotebook75 = new wxNotebook( itemFrame1, ID_NOTEBOOK, wxDefaultPosition, wxDefaultSize, wxBK_DEFAULT );

    wxTreeCtrl* itemTreeCtrl76 = new wxTreeCtrl( itemNotebook75, ID_TREECTRL, wxDefaultPosition, wxDefaultSize, wxTR_SINGLE );

    itemNotebook75->AddPage(itemTreeCtrl76, _("Tab"));

    wxGrid* itemGrid77 = new wxGrid( itemNotebook75, ID_GRID, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL );
    itemGrid77->SetDefaultColSize(50);
    itemGrid77->SetDefaultRowSize(25);
    itemGrid77->SetColLabelSize(25);
    itemGrid77->SetRowLabelSize(50);
    itemGrid77->CreateGrid(5, 5, wxGrid::wxGridSelectCells);

    itemNotebook75->AddPage(itemGrid77, _("Tab"));

    itemFrame1->GetAuiManager().AddPane(itemNotebook75, wxAuiPaneInfo()
        .Name(_T("Pane1")).MinSize(wxSize(200, 400)).BestSize(wxSize(200, 600)).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingSize(wxSize(300, 600)));

    GetAuiManager().Update();

    // Connect events and objects
    itemFrame1->Connect(ID_FIRSTMAIN, wxEVT_DESTROY, wxWindowDestroyEventHandler(FirstMain::OnDestroy), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_CREATE, wxWindowCreateEventHandler(FirstMain::OnCreate), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_DESTROY, wxWindowDestroyEventHandler(FirstMain::OnDestroy), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_SIZE, wxSizeEventHandler(FirstMain::OnSize), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_PAINT, wxPaintEventHandler(FirstMain::OnPaint), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_LEFT_DOWN, wxMouseEventHandler(FirstMain::OnLeftDown), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_LEFT_UP, wxMouseEventHandler(FirstMain::OnLeftUp), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_MIDDLE_DOWN, wxMouseEventHandler(FirstMain::OnMiddleDown), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_MIDDLE_UP, wxMouseEventHandler(FirstMain::OnMiddleUp), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_RIGHT_DOWN, wxMouseEventHandler(FirstMain::OnRightDown), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_RIGHT_UP, wxMouseEventHandler(FirstMain::OnRightUp), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_MOTION, wxMouseEventHandler(FirstMain::OnMotion), NULL, this);
    itemGLCanvas->Connect(ID_GLCANVAS, wxEVT_MOUSEWHEEL, wxMouseEventHandler(FirstMain::OnMouseWheel), NULL, this);
    m_ColorList->Connect(ID_LISTCTRL, wxEVT_CREATE, wxWindowCreateEventHandler(FirstMain::OnCreate), NULL, this);
	////@end FirstMain content construction
} 

/*
* wxEVT_CREATE event handler for ID_FIRSTMAIN
*/

void FirstMain::OnCreate( wxWindowCreateEvent& event )
{
	////@begin wxEVT_CREATE event handler for ID_FIRSTMAIN in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_CREATE event handler for ID_FIRSTMAIN in FirstMain. 
}


/*
* wxEVT_DESTROY event handler for ID_FIRSTMAIN
*/

void FirstMain::OnDestroy( wxWindowDestroyEvent& event )
{
	////@begin wxEVT_DESTROY event handler for ID_FIRSTMAIN in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_DESTROY event handler for ID_FIRSTMAIN in FirstMain. 
}


/*
* Should we show tooltips?
*/

bool FirstMain::ShowToolTips()
{
	return true;
}

/*
* Get bitmap resources
*/

wxBitmap FirstMain::GetBitmapResource( const wxString& name )
{
	// Bitmap retrieval
	////@begin FirstMain bitmap retrieval
    wxUnusedVar(name);
    if (name == _T("fileopen.xpm"))
    {
        wxBitmap bitmap(fileopen_xpm);
        return bitmap;
    }
    else if (name == _T("filesave.xpm"))
    {
        wxBitmap bitmap(filesave_xpm);
        return bitmap;
    }
    else if (name == _T("copy.xpm"))
    {
        wxBitmap bitmap(copy_xpm);
        return bitmap;
    }
    else if (name == _T("cut.xpm"))
    {
        wxBitmap bitmap(cut_xpm);
        return bitmap;
    }
    else if (name == _T("paste.xpm"))
    {
        wxBitmap bitmap(paste_xpm);
        return bitmap;
    }
    return wxNullBitmap;
	////@end FirstMain bitmap retrieval
}

/*
* Get icon resources
*/

wxIcon FirstMain::GetIconResource( const wxString& name )
{
	// Icon retrieval
	////@begin FirstMain icon retrieval
    wxUnusedVar(name);
    return wxNullIcon;
	////@end FirstMain icon retrieval
}


/*
* wxEVT_SIZE event handler for ID_GLCANVAS
*/

void FirstMain::OnSize( wxSizeEvent& event )
{
	itemGLCanvas->SetCurrent();
	m_mathCube.Resize(event.GetSize().GetWidth(),event.GetSize().GetHeight());
	RenderFrame();
}


/*
* wxEVT_PAINT event handler for ID_GLCANVAS
*/

void FirstMain::OnPaint( wxPaintEvent& event )
{
	itemGLCanvas->SetCurrent();
	if (!m_init && m_hEvr)
	{
		m_mathCube.initWorld();
		m_init = true;
	}
	// Init OpenGL once, but after SetCurrent
	RenderFrame();
	event.Skip();
}


/*
* wxEVT_LEFT_DOWN event handler for ID_GLCANVAS
*/

void FirstMain::OnLeftDown( wxMouseEvent& event )
{
	////@begin wxEVT_LEFT_DOWN event handler for ID_GLCANVAS in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_LEFT_DOWN event handler for ID_GLCANVAS in FirstMain. 
}


/*
* wxEVT_LEFT_UP event handler for ID_GLCANVAS
*/

void FirstMain::OnLeftUp( wxMouseEvent& event )
{
	////@begin wxEVT_LEFT_UP event handler for ID_GLCANVAS in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_LEFT_UP event handler for ID_GLCANVAS in FirstMain. 
}


/*
* wxEVT_MIDDLE_DOWN event handler for ID_GLCANVAS
*/

void FirstMain::OnMiddleDown( wxMouseEvent& event )
{
	////@begin wxEVT_MIDDLE_DOWN event handler for ID_GLCANVAS in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_MIDDLE_DOWN event handler for ID_GLCANVAS in FirstMain. 
}


/*
* wxEVT_MIDDLE_UP event handler for ID_GLCANVAS
*/

void FirstMain::OnMiddleUp( wxMouseEvent& event )
{
	////@begin wxEVT_MIDDLE_UP event handler for ID_GLCANVAS in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_MIDDLE_UP event handler for ID_GLCANVAS in FirstMain. 
}


/*
* wxEVT_RIGHT_DOWN event handler for ID_GLCANVAS
*/

void FirstMain::OnRightDown( wxMouseEvent& event )
{
	////@begin wxEVT_RIGHT_DOWN event handler for ID_GLCANVAS in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_RIGHT_DOWN event handler for ID_GLCANVAS in FirstMain. 
}


/*
* wxEVT_RIGHT_UP event handler for ID_GLCANVAS
*/

void FirstMain::OnRightUp( wxMouseEvent& event )
{
	////@begin wxEVT_RIGHT_UP event handler for ID_GLCANVAS in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_RIGHT_UP event handler for ID_GLCANVAS in FirstMain. 
}


/*
* wxEVT_MOTION event handler for ID_GLCANVAS
*/

void FirstMain::OnMotion( wxMouseEvent& event )
{
	P= event.GetPosition();
	if (m_bFixMove)
	{
		lastP = P;
		m_bFixMove = false;
	}
	if (P != lastP)
	{
		wxPoint move = P-lastP;
		if (event.ButtonIsDown(wxMOUSE_BTN_RIGHT))
		{
			if (event.ButtonIsDown(wxMOUSE_BTN_LEFT))
				m_mathCube.SetEyeMove(move.x,move.y, true);
			else
				m_mathCube.SetEyeMove(move.x,move.y, false);
		}
		else if (event.ButtonIsDown(wxMOUSE_BTN_LEFT))
		{
			m_mathCube.SetRotate(move.x,move.y);
			RenderFrame();
		}
	}
	lastP = P;
}


/*
* wxEVT_MOUSEWHEEL event handler for ID_GLCANVAS
*/

void FirstMain::OnMouseWheel( wxMouseEvent& event )
{
	m_mathCube.SetDistance(event.GetWheelRotation());
	RenderFrame();
}


/*
* wxEVT_UPDATE_UI event handler for ID_GLCANVAS
*/

void FirstMain::OnGlcanvasUpdate( wxUpdateUIEvent& event )
{
	RenderFrame();
	event.Skip(false);
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_OPENFILE
*/

void FirstMain::OnBtnopenfileClick( wxCommandEvent& event )
{
	OpenFile();
	event.Skip(false);
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_SAVEFILE
*/

void FirstMain::OnBtnsavefileClick( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_MENU_SELECTED event handler for ID_SAVEFILE in FirstMain.
	// Before editing this code, remove the block markers.
	event.Skip();
	////@end wxEVT_COMMAND_MENU_SELECTED event handler for ID_SAVEFILE in FirstMain. 
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_COPY
*/

void FirstMain::OnBtncopyClick( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_MENU_SELECTED event handler for ID_COPY in FirstMain.
	// Before editing this code, remove the block markers.
	event.Skip();
	////@end wxEVT_COMMAND_MENU_SELECTED event handler for ID_COPY in FirstMain. 
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_PASTE
*/

void FirstMain::OnBtnpasteClick( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_MENU_SELECTED event handler for ID_PASTE in FirstMain.
	// Before editing this code, remove the block markers.
	event.Skip();
	////@end wxEVT_COMMAND_MENU_SELECTED event handler for ID_PASTE in FirstMain. 
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_CUT
*/

void FirstMain::OnBtncutClick( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_MENU_SELECTED event handler for ID_CUT in FirstMain.
	// Before editing this code, remove the block markers.
	event.Skip();
	////@end wxEVT_COMMAND_MENU_SELECTED event handler for ID_CUT in FirstMain. 
}




/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderAxis
*/

void FirstMain::OnRenderAxisClick( wxCommandEvent& event )
{
	m_RenderAxis = event.IsChecked();
	RenderFrame();
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderCube
*/

void FirstMain::OnRenderCubeClick( wxCommandEvent& event )
{
	m_RenderCube = event.IsChecked();
	RenderFrame();
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace1
*/

void FirstMain::OnRenderFace1Click( wxCommandEvent& event )
{
	m_RenderFace1 = event.IsChecked();
	RenderFrame();
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUITEM3
*/

void FirstMain::OnRenderFace2Click( wxCommandEvent& event )
{
	m_RenderFace2 = event.IsChecked();
	RenderFrame();
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace3
*/

void FirstMain::OnRenderFace3Click( wxCommandEvent& event )
{
	m_RenderFace3 = event.IsChecked();
	RenderFrame();
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace4
*/

void FirstMain::OnRenderFace4Click( wxCommandEvent& event )
{
	m_RenderFace4 = event.IsChecked();
	RenderFrame();
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace5
*/

void FirstMain::OnRenderFace5Click( wxCommandEvent& event )
{
	m_RenderFace5 = event.IsChecked();
	RenderFrame();
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderFace6
*/

void FirstMain::OnRenderFace6Click( wxCommandEvent& event )
{
	m_RenderFace6 = event.IsChecked();
	RenderFrame();
}

/*
* ToOpenFile
*/

void FirstMain::OpenFile()
{
	m_bFixMove = true;
	wxFileDialog dialog
		(
		this,
		_T("Open file"),
		wxEmptyString,
		wxEmptyString,
		_T("Lua files (*.Lua)|*.Lua|Dat files (*.dat)|*.dat")
		);
	dialog.CentreOnParent();
	//dialog.SetDirectory(wxGetHomeDir());
	if (dialog.ShowModal() == wxID_OK)
	{
		itemGLCanvas->SetCurrent();       
		switch(dialog.GetFilterIndex())
		{
		case 0:
			if (m_hEvr)
				delete m_hEvr;
			m_hEvr = new HandleEvr(SConvStr::GetChar(dialog.GetPath().c_str()));
			m_hEvr->InitLoad(dialog.GetDirectory().wc_str());
			m_ShowTypeCombo->Clear();
			{
				int i=0;
				for (HandleEvr::strVector::iterator it = m_hEvr->m_format_name.begin();
					it != m_hEvr->m_format_name.end(); it++)
				{
					i++;
					if (i>3)
						m_ShowTypeCombo->Append(wxString::FromAscii(it->c_str()));
				}
			}
			m_ShowTypeCombo->SetSelection(0);
			m_mathCube.ReleaseResources();
			m_psjcF3d = m_hEvr->m_SJCSF3dMap[3].second;
			m_MarchCubeSet_spinctrl->SetMax(*std::max_element(m_psjcF3d->begin(),m_psjcF3d->end()));
			m_MarchCubeSet_spinctrl->SetMin(*std::min_element(m_psjcF3d->begin(),m_psjcF3d->end()));
			m_mathCube.SetData(m_psjcF3d, m_PreciseSpin->GetValue(), m_MarchCubeSet_spinctrl->GetValue()); // 設定第一順位的資料來顯示
			m_XminText->SetValue(wxString::FromAscii(SConvStr::GetChar(m_hEvr->Xmin)));
			m_XmaxText->SetValue(wxString::FromAscii(SConvStr::GetChar(m_hEvr->Xmax)));
			m_YminText->SetValue(wxString::FromAscii(SConvStr::GetChar(m_hEvr->Ymin)));
			m_YmaxText->SetValue(wxString::FromAscii(SConvStr::GetChar(m_hEvr->Ymax)));
			m_ZminText->SetValue(wxString::FromAscii(SConvStr::GetChar(m_hEvr->Zmin)));
			m_ZmaxText->SetValue(wxString::FromAscii(SConvStr::GetChar(m_hEvr->Zmax)));

			m_MiddleXText->SetValue(wxString::FromAscii(SConvStr::GetChar(m_hEvr->Xmin)));
			m_MiddleYText->SetValue(wxString::FromAscii(SConvStr::GetChar(m_hEvr->Ymin)));
			m_MiddleZText->SetValue(wxString::FromAscii(SConvStr::GetChar(m_hEvr->Zmin)));
			break;
		case 1:
			{
				m_ConvEvr = ConvertToEvr();
				m_ConvEvr.Load_Dat(dialog.GetPath().c_str());
				wxFileDialog dialog(this,
					_T("Save file"),
					wxEmptyString,
					_T("newdata"),
					_T("Evr binary (*.lua;*.evr)|*|Evr ascii files (*.lua;*.evr)|*"),
					wxFD_SAVE|wxFD_OVERWRITE_PROMPT);
				dialog.SetFilterIndex(1);
				if (dialog.ShowModal() == wxID_OK)
				{
					switch(dialog.GetFilterIndex())
					{
					case 0:
						m_ConvEvr.Save_Evr(dialog.GetPath().c_str(), dialog.GetFilename().c_str());
						break;
					case 1:
						m_ConvEvr.Save_EvrA(dialog.GetPath().c_str(), dialog.GetFilename().c_str());
						break;
					}
				}
			}
			break;
		}
	}
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUOPENFILE
*/

void FirstMain::OnMenuopenfileClick( wxCommandEvent& event )
{
	OpenFile();
	event.Skip(false);
}

/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUSaveFile
*/

void FirstMain::OnMENUSaveFileClick( wxCommandEvent& event )
{
	if (m_hEvr)
	{
		wxFileDialog dialog(this,
			_T("Save file"),
			wxEmptyString,
			_T("newdata"),
			_T("Evr binary (*.lua;*.evr)|*|Evr ascii files (*.lua;*.evr)|*"),
			wxFD_SAVE|wxFD_OVERWRITE_PROMPT);
		dialog.SetFilterIndex(1);
		if (dialog.ShowModal() == wxID_OK)
		{
			switch(dialog.GetFilterIndex())
			{
			case 0:
				m_hEvr->Save_Evr(dialog.GetPath().c_str(), dialog.GetFilename().c_str());
				break;
			case 1:
				m_hEvr->Save_EvrA(dialog.GetPath().c_str(), dialog.GetFilename().c_str());
				break;
			}
		}
	}
	event.Skip(false);
}

/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUSHOWFILEINFO
*/

void FirstMain::OnMenushowfileinfoClick( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUSHOWFILEINFO in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUSHOWFILEINFO in FirstMain. 
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUEDITFILEINFO
*/

void FirstMain::OnMenueditfileinfoClick( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUEDITFILEINFO in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUEDITFILEINFO in FirstMain. 
}


/*
* wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUEXIT
*/

void FirstMain::OnMenuexitClick( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUEXIT in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUEXIT in FirstMain. 
}


/*
* wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL
*/

void FirstMain::OnXminTextTextUpdated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL in FirstMain. 
}


/*
* wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL1
*/

void FirstMain::OnXmaxTextTextUpdated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL1 in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL1 in FirstMain. 
}


/*
* wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL3
*/

void FirstMain::OnYminTextTextUpdated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL3 in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL3 in FirstMain. 
}


/*
* wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL2
*/

void FirstMain::OnYmaxTextTextUpdated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL2 in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL2 in FirstMain. 
}


/*
* wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL4
*/

void FirstMain::OnZminTextTextUpdated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL4 in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL4 in FirstMain. 
}


/*
* wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL5
*/

void FirstMain::OnZmaxTextTextUpdated( wxCommandEvent& event )
{
	////@begin wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL5 in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
	////@end wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL5 in FirstMain. 
}

/*
 * wxEVT_COMMAND_SLIDER_UPDATED event handler for ID_SLIDER
 */

void FirstMain::OnSliderUpdated( wxCommandEvent& event )
{
	m_text_chipX->SetValue(wxString::FromAscii(SConvStr::GetChar(m_XSlider->GetValue())));
	RenderFrame();
	event.Skip(false); 
}


/*
 * wxEVT_COMMAND_SLIDER_UPDATED event handler for ID_SLIDER1
 */

void FirstMain::OnSlider1Updated( wxCommandEvent& event )
{
	m_text_chipY->SetValue(wxString::FromAscii(SConvStr::GetChar(m_YSlider->GetValue())));
	RenderFrame();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_SLIDER_UPDATED event handler for ID_SLIDER2
 */

void FirstMain::OnSlider2Updated( wxCommandEvent& event )
{
	m_text_chipZ->SetValue(wxString::FromAscii(SConvStr::GetChar(m_ZSlider->GetValue())));
	RenderFrame();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL9
 */
//text_chipX
void FirstMain::OnTextChipXTextUpdated( wxCommandEvent& event )
{
	if (!m_init) return; // 確定有初始化了才執行
	long n;
	m_text_chipX->GetValue().ToLong(&n);
	if (n<0) n = 0;
	if (n>1000) n = 1000;
	m_XSlider->SetValue(n);
	RenderFrame();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL10
 */
//text_chipY
void FirstMain::OnTextChipYTextUpdated( wxCommandEvent& event )
{
	if (!m_init) return; // 確定有初始化了才執行
	long n;
	m_text_chipY->GetValue().ToLong(&n);
	if (n<0) n = 0;
	if (n>1000) n = 1000;
	m_YSlider->SetValue(n);
	RenderFrame();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_TEXT_UPDATED event handler for ID_TEXTCTRL11
 */
//text_chipZ
void FirstMain::OnTextChipZTextUpdated( wxCommandEvent& event )
{
	if (!m_init) return; // 確定有初始化了才執行
	long n;
	m_text_chipZ->GetValue().ToLong(&n);
	if (n<0) n = 0;
	if (n>1000) n = 1000;
	m_ZSlider->SetValue(n);
	RenderFrame();
	event.Skip(false);
}


void FirstMain::RenderFrame()
{
	itemGLCanvas->SetCurrent();
	if (!m_init && m_hEvr)
	{
		m_mathCube.initWorld();
		m_LoadColor = true;
		ShowColorTable(m_mathCube.GetColorTable());
		m_init = true;
	}
	if (!m_init) return;
// 	if (!m_LoadColor)
// 	{
// 		m_LoadColor = true;
// 		ShowColorTable(m_mathCube.GetColorTable());
// 	}
	// Init OpenGL once, but after SetCurrent
	m_mathCube.RenderStart();
	if (m_RenderAxis)
		m_mathCube.RenderAxis();
	
	if (m_hEvr)
	{
		if (m_useXchip)
			m_mathCube.RenderChip(MathCube::USE_X, m_XSlider->GetValue());
		if (m_useYchip)
			m_mathCube.RenderChip(MathCube::USE_Y, m_YSlider->GetValue());
		if (m_useZchip)
			m_mathCube.RenderChip(MathCube::USE_Z, m_ZSlider->GetValue());
		if (m_RenderCube)
			m_mathCube.RenderCube();
		else
		{
			if (m_RenderFace1)
				m_mathCube.RenderFace(1);
			if (m_RenderFace2)
				m_mathCube.RenderFace(2);
			if (m_RenderFace3)
				m_mathCube.RenderFace(3);
			if (m_RenderFace4)
				m_mathCube.RenderFace(4);
			if (m_RenderFace5)
				m_mathCube.RenderFace(5);
			if (m_RenderFace6)
				m_mathCube.RenderFace(6);
		}
		if (m_RenderBondingBox)
			m_mathCube.RenderBondingBox();
	}
	itemGLCanvas->SwapBuffers();
}


/*
 * ToShowColorList
 */

void FirstMain::ShowColorTable(ColorTable* ict)
{
	m_ColorList->ClearAll();
	wxListItem itemCol;
	itemCol.SetText(_T("Color"));
	m_ColorList->InsertColumn(0, itemCol);
	itemCol.SetText(_T("Red"));
	m_ColorList->InsertColumn(1, itemCol);
	itemCol.SetText(_T("Green"));
	m_ColorList->InsertColumn(2, itemCol);
	itemCol.SetText(_T("Blue"));
	m_ColorList->InsertColumn(3, itemCol);

	m_ColorList->SetColumnWidth( 0, 70 );
	m_ColorList->SetColumnWidth( 1, 40 );
	m_ColorList->SetColumnWidth( 2, 40 );
	m_ColorList->SetColumnWidth( 3, 40 );
	int i=0;
	for (ColorTable::ctMap::iterator it = ict->vMapping.begin(); it != ict->vMapping.end();it++)
	{
		wxColour wc(it->second.r, it->second.g, it->second.b);
		InsertColor(i++, it->first, wc);
	}
}

/*
 * InsertItem to ColorList
 */

void FirstMain::InsertColor( int i, double val, wxColour& iwc )
{
	wxString buf;
	buf.Printf(_T("●%f"), val);
	long tmp = m_ColorList->InsertItem(i, buf, 0);
	m_ColorList->SetItemData(tmp, i);

	buf.Printf(_T("%d"), int(iwc.Red()));
	m_ColorList->SetItem(tmp, 1, buf);
	buf.Printf(_T("%d"), int(iwc.Green()));
	m_ColorList->SetItem(tmp, 2, buf);
	buf.Printf(_T("%d"), int(iwc.Blue()));
	m_ColorList->SetItem(tmp, 3, buf);

	wxListItem item;
	item.m_itemId = i;
	item.SetTextColour(iwc);
	m_ColorList->SetItem( item );
}


/*
 * wxEVT_COMMAND_COMBOBOX_SELECTED event handler for ID_ShowTypeCombo
 */

void FirstMain::OnShowTypeComboSelected( wxCommandEvent& event )
{
	std::string str = SConvStr::GetChar(m_ShowTypeCombo->GetValue().c_str());
	uint i=0;
	for (;i<m_hEvr->m_SJCSF3dMap.size();i++)
		if (m_hEvr->m_SJCSF3dMap[i].first == str)
	assert(i!=m_hEvr->m_SJCSF3dMap.size());
	if (i==m_hEvr->m_SJCSF3dMap.size())  return;
	m_psjcF3d = m_hEvr->m_SJCSF3dMap[i].second;
	m_MarchCubeSet_spinctrl->SetMax(*std::max_element(m_psjcF3d->begin(),m_psjcF3d->end()));
	m_MarchCubeSet_spinctrl->SetMin(*std::min_element(m_psjcF3d->begin(),m_psjcF3d->end()));
	m_mathCube.SetData(m_psjcF3d, m_PreciseSpin->GetValue(), m_MarchCubeSet_spinctrl->GetValue()); // 設定第目前的資料來顯示
	RenderFrame();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_SPINCTRL_UPDATED event handler for ID_SPINCTRL
 */

void FirstMain::OnSpinctrlUpdated( wxSpinEvent& event )
{
	m_mathCube.SetData(m_psjcF3d, m_PreciseSpin->GetValue(), m_MarchCubeSet_spinctrl->GetValue());
	RenderFrame();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_MENU_SELECTED event handler for ID_FileEditToolbar
 */

void FirstMain::OnFileEditToolbarClick( wxCommandEvent& event )
{
	if (event.IsChecked())
		this->GetAuiManager().GetPane(_T("FileEditToolbar")).Show();
	else
		this->GetAuiManager().GetPane(_T("FileEditToolbar")).Hide();
	this->GetAuiManager().Update();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_MENU_SELECTED event handler for ID_PositionEditToolbar
 */

void FirstMain::OnPositionEditToolbarClick( wxCommandEvent& event )
{
	if (event.IsChecked())
		this->GetAuiManager().GetPane(_T("PositionEditToolbar")).Show();
	else
		this->GetAuiManager().GetPane(_T("PositionEditToolbar")).Hide();
	this->GetAuiManager().Update();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_MENU_SELECTED event handler for ID_BoundEditToolbar
 */

void FirstMain::OnBoundEditToolbarClick( wxCommandEvent& event )
{
	if (event.IsChecked())
		this->GetAuiManager().GetPane(_T("BoundEditToolbar")).Show();
	else
		this->GetAuiManager().GetPane(_T("BoundEditToolbar")).Hide();
	this->GetAuiManager().Update();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_MENU_SELECTED event handler for ID_XYZchipEditToolbar
 */

void FirstMain::OnXYZchipEditToolbarClick( wxCommandEvent& event )
{
	if (event.IsChecked())
		this->GetAuiManager().GetPane(_T("XYZchipEditToolbar")).Show();
	else
		this->GetAuiManager().GetPane(_T("XYZchipEditToolbar")).Hide();
	this->GetAuiManager().Update();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_MENU_SELECTED event handler for ID_ColorTable
 */

void FirstMain::OnColorTableClick( wxCommandEvent& event )
{
	if (event.IsChecked())
		this->GetAuiManager().GetPane(_T("ColorList")).Show();
	else
		this->GetAuiManager().GetPane(_T("ColorList")).Hide();
	this->GetAuiManager().Update();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_MENU_SELECTED event handler for ID_MENUITEM
 */

void FirstMain::OnMENUPreciseToolbarClick( wxCommandEvent& event )
{
	if (event.IsChecked())
		this->GetAuiManager().GetPane(_T("PreciseToolbar")).Show();
	else
		this->GetAuiManager().GetPane(_T("PreciseToolbar")).Hide();
	this->GetAuiManager().Update();
	event.Skip(false);
}




/*
 * wxEVT_COMMAND_SPINCTRL_UPDATED event handler for ID_MarchCubeSet_SPINCTRL
 */

void FirstMain::OnMarchCubeSetSPINCTRLUpdated( wxSpinEvent& event )
{
	m_mathCube.ResetMarchCubeLevel(m_MarchCubeSet_spinctrl->GetValue());
	event.Skip(false);
}

/*
* wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_CHECKBOX
*/

void FirstMain::OnUseXchipClick( wxCommandEvent& event )
{
	m_useXchip = event.IsChecked();
	event.Skip(false);
}

/*
 * wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_USE_YCHIP
 */

void FirstMain::OnUseYchipClick( wxCommandEvent& event )
{
	m_useYchip = event.IsChecked();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_CHECKBOX_CLICKED event handler for ID_USE_ZCHIP
 */

void FirstMain::OnUseZchipClick( wxCommandEvent& event )
{
	m_useZchip = event.IsChecked();
	event.Skip(false);
}


/*
 * wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderXchip
 */

void FirstMain::OnRenderXchipClick( wxCommandEvent& event )
{
////@begin wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderXchip in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderXchip in FirstMain. 
}


/*
 * wxEVT_COMMAND_MENU_SELECTED event handler for ID_RenderBondingBox
 */

void FirstMain::OnRenderBondingBoxClick( wxCommandEvent& event )
{
	m_RenderBondingBox = event.IsChecked();
	event.Skip(false); 
}

