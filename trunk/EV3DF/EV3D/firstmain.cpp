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
#include "StdWxVtk.h"
// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#ifdef __BORLANDC__
#pragma hdrstop
#endif
#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif
////@begin includes
#include "wx/imaglist.h"
#include "Mytreectrl.h"
#include "mygrid.h"
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
    EVT_WINDOW_DESTROY( FirstMain::OnDestroy )

    EVT_UPDATE_UI( ID_GLCANVAS, FirstMain::OnGlcanvasUpdate )

    EVT_MENU( ID_MENUOPENFILE, FirstMain::OnMenuopenfileClick )

    EVT_MENU( ID_MENUSaveFile, FirstMain::OnMENUSaveFileClick )

    EVT_MENU( ID_MENU_CONVERT_FILE, FirstMain::OnMenuConvertFileClick )

    EVT_MENU( ID_MENUEXIT, FirstMain::OnMenuexitClick )

    EVT_MENU( ID_FileEditToolbar, FirstMain::OnFileEditToolbarClick )

    EVT_MENU( ID_PositionEditToolbar, FirstMain::OnPositionEditToolbarClick )

    EVT_MENU( ID_BoundEditToolbar, FirstMain::OnBoundEditToolbarClick )

    EVT_MENU( ID_XYZchipEditToolbar, FirstMain::OnXYZchipEditToolbarClick )

    EVT_MENU( ID_ColorTable, FirstMain::OnColorTableClick )

    EVT_MENU( ID_MENUPreciseToolbar, FirstMain::OnMENUPreciseToolbarClick )

    EVT_MENU( ID_BTNOPENFILE, FirstMain::OnBtnopenfileClick )

    EVT_MENU( ID_BTNSAVEFILE, FirstMain::OnBtnsavefileClick )

    EVT_MENU( ID_BTNCOPY, FirstMain::OnBtncopyClick )

    EVT_MENU( ID_BTNCUT, FirstMain::OnBtncutClick )

    EVT_MENU( ID_BTNPASTE, FirstMain::OnBtnpasteClick )

    EVT_TEXT( ID_XminText, FirstMain::OnXminTextTextUpdated )

    EVT_TEXT( ID_XmaxText, FirstMain::OnXmaxTextTextUpdated )

    EVT_TEXT( ID_YminText, FirstMain::OnYminTextTextUpdated )

    EVT_TEXT( ID_YmaxText, FirstMain::OnYmaxTextTextUpdated )

    EVT_TEXT( ID_ZminText, FirstMain::OnZminTextTextUpdated )

    EVT_TEXT( ID_ZmaxText, FirstMain::OnZmaxTextTextUpdated )

    EVT_COMBOBOX( ID_ShowTypeCombo, FirstMain::OnShowTypeComboSelected )

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
    m_init = false;
    m_bFixMove = false;
    m_LoadColor = false;
    itemGLCanvas = NULL;
    m_FileEditToolbar = NULL;
    m_BoundEditToolbar = NULL;
    m_XminText = NULL;
    m_XmaxText = NULL;
    m_YminText = NULL;
    m_YmaxText = NULL;
    m_ZminText = NULL;
    m_ZmaxText = NULL;
    m_ShowTypeCombo = NULL;
    m_ColorList = NULL;
    m_treectrl = NULL;
    m_grid = NULL;
	////@end FirstMain member initialisation
	m_hEvr = NULL;
	m_psjcF3d = NULL;
	m_convertdialog = NULL;
	m_SolidCtrl = SharePtrNew;
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
    itemMenu4->Append(ID_MENU_CONVERT_FILE, _("ConvertFile"), wxEmptyString, wxITEM_NORMAL);
    itemMenu4->Append(ID_MENUEXIT, _("Exit"), wxEmptyString, wxITEM_NORMAL);
    menuBar->Append(itemMenu4, _("File"));
    wxMenu* itemMenu9 = new wxMenu;
    itemMenu9->Append(ID_FileEditToolbar, _("FileEdit Toolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu9->Check(ID_FileEditToolbar, true);
    itemMenu9->Append(ID_PositionEditToolbar, _("Position Edit Toolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu9->Check(ID_PositionEditToolbar, true);
    itemMenu9->Append(ID_BoundEditToolbar, _("Bound Edit Toolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu9->Check(ID_BoundEditToolbar, true);
    itemMenu9->Append(ID_XYZchipEditToolbar, _("XYZchip Edit Toolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu9->Check(ID_XYZchipEditToolbar, true);
    itemMenu9->Append(ID_ColorTable, _("Color Table"), wxEmptyString, wxITEM_CHECK);
    itemMenu9->Check(ID_ColorTable, true);
    itemMenu9->Append(ID_MENUPreciseToolbar, _("PreciseToolbar"), wxEmptyString, wxITEM_CHECK);
    itemMenu9->Check(ID_MENUPreciseToolbar, true);
    menuBar->Append(itemMenu9, _("View"));
    itemFrame1->SetMenuBar(menuBar);

    itemGLCanvas = new wxGLCanvas( itemFrame1, ID_GLCANVAS, wxDefaultPosition, wxDefaultSize, 0 );
    itemFrame1->GetAuiManager().AddPane(itemGLCanvas, wxAuiPaneInfo()
        .Name(_T("ID_GLCANVAS")).Caption(_("layout")).Centre().TopDockable(false).BottomDockable(false).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingSize(wxSize(800, 800)));

    m_FileEditToolbar = new wxAuiToolBar( itemFrame1, ID_TOOLBAR, wxDefaultPosition, wxDefaultSize, wxAUI_TB_GRIPPER );
    wxBitmap itemtool17Bitmap(itemFrame1->GetBitmapResource(wxT("fileopen.xpm")));
    wxBitmap itemtool17BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNOPENFILE, wxEmptyString, itemtool17Bitmap, itemtool17BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool18Bitmap(itemFrame1->GetBitmapResource(wxT("filesave.xpm")));
    wxBitmap itemtool18BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNSAVEFILE, wxEmptyString, itemtool18Bitmap, itemtool18BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool19Bitmap(itemFrame1->GetBitmapResource(wxT("copy.xpm")));
    wxBitmap itemtool19BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNCOPY, wxEmptyString, itemtool19Bitmap, itemtool19BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool20Bitmap(itemFrame1->GetBitmapResource(wxT("cut.xpm")));
    wxBitmap itemtool20BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNCUT, wxEmptyString, itemtool20Bitmap, itemtool20BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    wxBitmap itemtool21Bitmap(itemFrame1->GetBitmapResource(wxT("paste.xpm")));
    wxBitmap itemtool21BitmapDisabled;
    m_FileEditToolbar->AddTool(ID_BTNPASTE, wxEmptyString, itemtool21Bitmap, itemtool21BitmapDisabled, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL);
    m_FileEditToolbar->Realize();
    itemFrame1->GetAuiManager().AddPane(m_FileEditToolbar, wxAuiPaneInfo()
        .ToolbarPane().Name(_T("FileEditToolbar")).Top().Layer(10).CaptionVisible(false).CloseButton(false).DestroyOnClose(false).Resizable(false).Floatable(false).Gripper(true));

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
    wxArrayString m_ShowTypeComboStrings;
    m_ShowTypeCombo = new wxComboBox( m_BoundEditToolbar, ID_ShowTypeCombo, wxEmptyString, wxDefaultPosition, wxDefaultSize, m_ShowTypeComboStrings, wxCB_READONLY );
    m_BoundEditToolbar->AddControl(m_ShowTypeCombo);
    m_BoundEditToolbar->Realize();
    itemFrame1->GetAuiManager().AddPane(m_BoundEditToolbar, wxAuiPaneInfo()
        .ToolbarPane().Name(_T("BoundEditToolbar")).Top().Layer(10).CaptionVisible(false).CloseButton(false).DestroyOnClose(false).Resizable(false).Floatable(false).Gripper(true));

    m_ColorList = new wxListCtrl( itemFrame1, ID_LISTCTRL, wxDefaultPosition, wxDefaultSize, wxLC_REPORT );
    itemFrame1->GetAuiManager().AddPane(m_ColorList, wxAuiPaneInfo()
        .Name(_T("ColorList")).Caption(_("ColorTable")).BestSize(wxSize(200, 200)).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingSize(wxSize(200, 200)));

    wxStatusBar* itemStatusBar37 = new wxStatusBar( itemFrame1, ID_STATUSBAR, wxST_SIZEGRIP|wxNO_BORDER );
    itemStatusBar37->SetFieldsCount(2);
    itemFrame1->SetStatusBar(itemStatusBar37);

    m_treectrl = new MyTreeCtrl( itemFrame1, ID_TREECTRL, wxDefaultPosition, wxDefaultSize, wxTR_EDIT_LABELS|wxTR_SINGLE );
    itemFrame1->GetAuiManager().AddPane(m_treectrl, wxAuiPaneInfo()
        .Name(_T("ID_TREECTRL")).Caption(_("Effect Tree")).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingSize(wxSize(200, 200)));

    m_grid = new MyGrid( itemFrame1, ID_GRID, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL );
    m_grid->SetDefaultColSize(75);
    m_grid->SetDefaultRowSize(25);
    m_grid->SetColLabelSize(25);
    m_grid->SetRowLabelSize(100);
    m_grid->CreateGrid(1, 1, wxGrid::wxGridSelectCells);
    itemFrame1->GetAuiManager().AddPane(m_grid, wxAuiPaneInfo()
        .Name(_T("ID_GRID")).Caption(_("Attribute")).CloseButton(false).DestroyOnClose(false).Resizable(true).FloatingSize(wxSize(200, -1)));

    GetAuiManager().Update();

    // Connect events and objects
    itemFrame1->Connect(ID_FIRSTMAIN, wxEVT_DESTROY, wxWindowDestroyEventHandler(FirstMain::OnDestroy), NULL, this);
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
	////@end FirstMain content construction
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
	if (m_SolidCtrl.get() != NULL)
		m_SolidCtrl->ReSize(event.GetSize().GetWidth(),event.GetSize().GetHeight());
	RenderFrame();
}

/*
* wxEVT_PAINT event handler for ID_GLCANVAS
*/
void FirstMain::OnPaint( wxPaintEvent& event )
{
	itemGLCanvas->SetCurrent();
	// Init OpenGL once, but after SetCurrent
	RenderFrame();
	event.Skip();
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
	
	lastP = P;
}

/*
* wxEVT_MOUSEWHEEL event handler for ID_GLCANVAS
*/
void FirstMain::OnMouseWheel( wxMouseEvent& event )
{
	RenderFrame();
	event.Skip(false);
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
			m_hEvr = new HandleEvr(VarStr(dialog.GetPath().c_str()));
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
			m_treectrl->RmAllAddItem();
			m_psjcF3d = m_hEvr->m_SJCSF3dMap[3].second;
			if (m_psjcF3d)
				m_SolidCtrl->SetGridedData(m_psjcF3d); // 設定第一順位的資料來顯示
			else
				m_SolidCtrl->SetUnGridData(m_hEvr->m_SJCSF3dMap[3].second.m_polydata);
			m_XminText->SetValue(wxString::FromAscii(ConvStr::GetStr(m_hEvr->Xmin).c_str()));
			m_XmaxText->SetValue(wxString::FromAscii(ConvStr::GetStr(m_hEvr->Xmax).c_str()));
			m_YminText->SetValue(wxString::FromAscii(ConvStr::GetStr(m_hEvr->Ymin).c_str()));
			m_YmaxText->SetValue(wxString::FromAscii(ConvStr::GetStr(m_hEvr->Ymax).c_str()));
			m_ZminText->SetValue(wxString::FromAscii(ConvStr::GetStr(m_hEvr->Zmin).c_str()));
			m_ZmaxText->SetValue(wxString::FromAscii(ConvStr::GetStr(m_hEvr->Zmax).c_str()));
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
void FirstMain::OnMenuConvertFileClick( wxCommandEvent& event )
{
	m_convertdialog = new ConvertDialog(this);
	m_convertdialog->Show(true);
	event.Skip(false);
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

void FirstMain::RenderFrame()
{
	if (m_hEvr)
	{
		m_SolidCtrl->SetHwnd((HWND)itemGLCanvas->GetHandle());
		m_SolidCtrl->Render(); 
	}
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
	std::string str = ConvStr::GetStr(m_ShowTypeCombo->GetValue().c_str());
	uint i=0;
	for (;i<m_hEvr->m_SJCSF3dMap.size();i++)
	{
		if (m_hEvr->m_SJCSF3dMap[i].first == str)
		{
			// 保證取得的資料是對的，不對就不選擇
			assert(i!=m_hEvr->m_SJCSF3dMap.size());
			if (i==m_hEvr->m_SJCSF3dMap.size())  return;
			m_psjcF3d = m_hEvr->m_SJCSF3dMap[i].second;
		}
	}
	if (m_psjcF3d)
		m_SolidCtrl->SetGridedData(m_psjcF3d); // 設定第一順位的資料來顯示
	else
		m_SolidCtrl->SetUnGridData(m_hEvr->m_SJCSF3dMap[i].second.m_polydata); // 需要griding一下
	m_treectrl->RmAllAddItem();
// 	m_MarchCubeSet_spinctrl->SetMax(*std::max_element(m_psjcF3d->begin(),m_psjcF3d->end()));
// 	m_MarchCubeSet_spinctrl->SetMin(*std::min_element(m_psjcF3d->begin(),m_psjcF3d->end()));
	//m_SolidCtrl.Set(m_PreciseSpin->GetValue());
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
 * wxEVT_LEFT_DOWN event handler for ID_GLCANVAS
 */
void FirstMain::OnLeftDown( wxMouseEvent& event )
{
////@begin wxEVT_LEFT_DOWN event handler for ID_GLCANVAS in FirstMain.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_LEFT_DOWN event handler for ID_GLCANVAS in FirstMain. 
}
