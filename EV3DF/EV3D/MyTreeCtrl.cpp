/////////////////////////////////////////////////////////////////////////////
// Name:        MyTreeCtrl.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     04/10/2010 04:59:20
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

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

////@begin includes
#include "wx/imaglist.h"
////@end includes
#include <algorithm>
#include "mytreectrl.h"
#include "firstmain.h"
#include "mygrid.h"
#include "DW/SEffect.h"
#include "DW/SolidView.h"
#include "DW/SolidDefine.h"

////@begin XPM images
////@end XPM images


/*
 * MyTreeCtrl type definition
 */

IMPLEMENT_DYNAMIC_CLASS( MyTreeCtrl, wxTreeCtrl )


/*
 * MyTreeCtrl event table definition
 */

BEGIN_EVENT_TABLE( MyTreeCtrl, wxTreeCtrl )

////@begin MyTreeCtrl event table entries
    EVT_TREE_SEL_CHANGED( ID_TREECTRL, MyTreeCtrl::OnTreectrlSelChanged )
    EVT_TREE_SEL_CHANGING( ID_TREECTRL, MyTreeCtrl::OnTreectrlSelChanging )
    EVT_TREE_DELETE_ITEM( ID_TREECTRL, MyTreeCtrl::OnTreectrlDeleteItem )
    EVT_TREE_ITEM_ACTIVATED( ID_TREECTRL, MyTreeCtrl::OnTreectrlItemActivated )
    EVT_TREE_ITEM_COLLAPSED( ID_TREECTRL, MyTreeCtrl::OnTreectrlItemCollapsed )
    EVT_TREE_ITEM_COLLAPSING( ID_TREECTRL, MyTreeCtrl::OnTreectrlItemCollapsing )
    EVT_TREE_ITEM_EXPANDED( ID_TREECTRL, MyTreeCtrl::OnTreectrlItemExpanded )
    EVT_TREE_ITEM_EXPANDING( ID_TREECTRL, MyTreeCtrl::OnTreectrlItemExpanding )
    EVT_TREE_KEY_DOWN( ID_TREECTRL, MyTreeCtrl::OnTreectrlKeyDown )
    EVT_TREE_ITEM_MENU( ID_TREECTRL, MyTreeCtrl::OnTreectrlItemMenu )
    EVT_TREE_ITEM_RIGHT_CLICK( ID_TREECTRL, MyTreeCtrl::OnTreectrlItemRightClick )

////@end MyTreeCtrl event table entries
    EVT_MENU(TreeCtrlMenu_AddItem,   MyTreeCtrl::OnMenu_AddItem)
    EVT_MENU(TreeCtrlMenu_DeleteItem,   MyTreeCtrl::OnMenu_DeleteItem)
    
END_EVENT_TABLE()


/*
 * MyTreeCtrl constructors
 */

MyTreeCtrl::MyTreeCtrl()
{
    Init();
}

MyTreeCtrl::MyTreeCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
    Init();
    Create(parent, id, pos, size, style);
}


/*
 * MyTreeCtrl creator
 */

bool MyTreeCtrl::Create(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
////@begin MyTreeCtrl creation
    wxTreeCtrl::Create(parent, id, pos, size, style);
    CreateControls();
////@end MyTreeCtrl creation
    return true;
}


/*
 * MyTreeCtrl destructor
 */

MyTreeCtrl::~MyTreeCtrl()
{
////@begin MyTreeCtrl destruction
////@end MyTreeCtrl destruction
}


/*
 * Member initialisation
 */

void MyTreeCtrl::Init()
{
////@begin MyTreeCtrl member initialisation
////@end MyTreeCtrl member initialisation
}


/*
 * Control creation for MyTreeCtrl
 */

void MyTreeCtrl::CreateControls()
{    

	CreateImageList();
	int image = MyTreeCtrl::TreeCtrlIcon_Folder;
	wxTreeItemId rootId = AddRoot(wxT("Effect"),
		image, image,
		new MyTreeItemData(wxT("Root item")));
	if ( !HasFlag(wxTR_HIDE_ROOT) && image != -1 )
	{
		SetItemImage(rootId, TreeCtrlIcon_FolderOpened, wxTreeItemIcon_Expanded);
	}
	if ( !HasFlag(wxTR_HIDE_ROOT) )
		SetItemFont(rootId, *wxITALIC_FONT);
	wxString item_str;
	wxTreeItemId id;
	item_str = wxT("Bounding Box");
	id = AppendItem(rootId, item_str, TreeCtrlIcon_Folder, TreeCtrlIcon_Folder+1,
		new MyTreeItemData(item_str));
	SetItemImage(id, TreeCtrlIcon_FolderOpened,
		wxTreeItemIcon_Expanded);
	item_str = wxT("Vertex");
	id = AppendItem(rootId, item_str, TreeCtrlIcon_Folder, TreeCtrlIcon_Folder+1,
		new MyTreeItemData(item_str));
	SetItemImage(id, TreeCtrlIcon_FolderOpened,
		wxTreeItemIcon_Expanded);
	item_str = wxT("Isosurface Contour");
	id = AppendItem(rootId, item_str, TreeCtrlIcon_Folder, TreeCtrlIcon_Folder+1,
		new MyTreeItemData(item_str));
	SetItemImage(id, TreeCtrlIcon_FolderOpened,
		wxTreeItemIcon_Expanded);
	item_str = wxT("Axes");
	id = AppendItem(rootId, item_str, TreeCtrlIcon_Folder, TreeCtrlIcon_Folder+1,
		new MyTreeItemData(item_str));
	SetItemImage(id, TreeCtrlIcon_FolderOpened,
		wxTreeItemIcon_Expanded);
	item_str = wxT("Ruler");
	id = AppendItem(rootId, item_str, TreeCtrlIcon_Folder, TreeCtrlIcon_Folder+1,
		new MyTreeItemData(item_str));
	SetItemImage(id, TreeCtrlIcon_FolderOpened,
		wxTreeItemIcon_Expanded);
	item_str = wxT("Plane Chip");
	id = AppendItem(rootId, item_str, TreeCtrlIcon_Folder, TreeCtrlIcon_Folder+1,
		new MyTreeItemData(item_str));
	SetItemImage(id, TreeCtrlIcon_FolderOpened,
		wxTreeItemIcon_Expanded);
	item_str = wxT("Contour Chip");
	id = AppendItem(rootId, item_str, TreeCtrlIcon_Folder, TreeCtrlIcon_Folder+1,
		new MyTreeItemData(item_str));
	SetItemImage(id, TreeCtrlIcon_FolderOpened,
		wxTreeItemIcon_Expanded);
	item_str = wxT("Volume Render");
	id = AppendItem(rootId, item_str, TreeCtrlIcon_Folder, TreeCtrlIcon_Folder+1,
		new MyTreeItemData(item_str));
	SetItemImage(id, TreeCtrlIcon_FolderOpened,
		wxTreeItemIcon_Expanded);
	Expand(rootId);

}

/*
 * wxEVT_COMMAND_TREE_DELETE_ITEM event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlDeleteItem( wxTreeEvent& event )
{
////@begin wxEVT_COMMAND_TREE_DELETE_ITEM event handler for ID_TREECTRL in MyTreeCtrl.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_TREE_DELETE_ITEM event handler for ID_TREECTRL in MyTreeCtrl. 
}


/*
 * wxEVT_COMMAND_TREE_KEY_DOWN event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlKeyDown( wxTreeEvent& event )
{
////@begin wxEVT_COMMAND_TREE_KEY_DOWN event handler for ID_TREECTRL in MyTreeCtrl.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_TREE_KEY_DOWN event handler for ID_TREECTRL in MyTreeCtrl. 
}


/*
 * wxEVT_COMMAND_TREE_ITEM_RIGHT_CLICK event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlItemRightClick( wxTreeEvent& event )
{
	SetLastItem(GetSelection());
	event.Skip();
}

void MyTreeCtrl::CreateImageList( int size /*= 16*/ )
{
	if ( size == -1 )
	{
		SetImageList(NULL);
		return;
	}
	if ( size == 0 )
		size = m_imageSize;
	else
		m_imageSize = size;

	// Make an image list containing small icons
	wxImageList *images = new wxImageList(size, size, true);

	// should correspond to TreeCtrlIcon_xxx enum
	wxBusyCursor wait;
	wxIcon icons[5];
	icons[0] = wxIcon(icon1_xpm);
	icons[1] = wxIcon(icon2_xpm);
	icons[2] = wxIcon(icon3_xpm);
	icons[3] = wxIcon(icon4_xpm);
	icons[4] = wxIcon(icon5_xpm);

	int sizeOrig = icons[0].GetWidth();
	for ( size_t i = 0; i < WXSIZEOF(icons); i++ )
	{
		if ( size == sizeOrig )
		{
			images->Add(icons[i]);
		}
		else
		{
			images->Add(wxBitmap(wxBitmap(icons[i]).ConvertToImage().Rescale(size, size)));
		}
	}

	AssignImageList(images);
}


/*
 * wxEVT_COMMAND_TREE_SEL_CHANGING event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlSelChanging( wxTreeEvent& event )
{
	SetLastItem(event.GetItem());
	event.Skip();
}


/*
 * wxEVT_COMMAND_TREE_ITEM_COLLAPSED event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlItemCollapsed( wxTreeEvent& event )
{
////@begin wxEVT_COMMAND_TREE_ITEM_COLLAPSED event handler for ID_TREECTRL in MyTreeCtrl.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_TREE_ITEM_COLLAPSED event handler for ID_TREECTRL in MyTreeCtrl. 
}


/*
 * wxEVT_COMMAND_TREE_ITEM_COLLAPSING event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlItemCollapsing( wxTreeEvent& event )
{
    event.Veto();
}


/*
 * wxEVT_COMMAND_TREE_ITEM_EXPANDED event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlItemExpanded( wxTreeEvent& event )
{
////@begin wxEVT_COMMAND_TREE_ITEM_EXPANDED event handler for ID_TREECTRL in MyTreeCtrl.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_TREE_ITEM_EXPANDED event handler for ID_TREECTRL in MyTreeCtrl. 
}


/*
 * wxEVT_COMMAND_TREE_ITEM_EXPANDING event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlItemExpanding( wxTreeEvent& event )
{
////@begin wxEVT_COMMAND_TREE_ITEM_EXPANDING event handler for ID_TREECTRL in MyTreeCtrl.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_COMMAND_TREE_ITEM_EXPANDING event handler for ID_TREECTRL in MyTreeCtrl. 
}


/*
 * wxEVT_COMMAND_TREE_ITEM_MENU event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlItemMenu( wxTreeEvent& event )
{
	SetLastItem(event.GetItem());
	wxTreeItemId itemId = event.GetItem();
	//MyTreeItemData *item = itemId.IsOk() ? (MyTreeItemData *)GetItemData(itemId): NULL;
	wxPoint clientpt = event.GetPoint();
	wxPoint screenpt = ClientToScreen(clientpt);
	ShowMenu(itemId, clientpt);
	event.Skip();
}

void MyTreeCtrl::ShowMenu( wxTreeItemId id, const wxPoint& pt )
{
	wxString title;
	if ( id.IsOk() )
	{
		title << wxT("Menu for ") << GetItemText(id);
	}
	else
	{
		title = wxT("Menu for no particular item");
	}

#if wxUSE_MENUS
	wxMenu menu;
	menu.Append(TreeCtrlMenu_AddItem, wxT("&AddItem"));
	menu.AppendSeparator();
	menu.Append(TreeCtrlMenu_DeleteItem, wxT("&DeleteItem"));

	PopupMenu(&menu, pt);
#endif // wxUSE_MENUS
}

void MyTreeCtrl::OnMenu_AddItem( wxCommandEvent& event )
{
	if (GetRootItem() == m_lastItem)
	{
		wxMessageDialog add_dialog(NULL, wxT("Prime item can't add"), wxT("Failed"));
		add_dialog.ShowModal();
		return;
	}
	wxTreeItemId tmpid = GetItemParent(m_lastItem);
	tmpid = GetItemParent(tmpid);
	if (GetRootItem() == tmpid)
	{
		wxMessageDialog add_dialog(NULL, wxT("Set item can't add"), wxT("Failed"));
		add_dialog.ShowModal();
		return;
	}
	wxString item_str = wxT("item");
	int i = GetChildrenCount( m_lastItem, false );
	item_str << wxT(" ") << i;
	const wxTreeItemId id = AppendItem(m_lastItem, item_str, TreeCtrlIcon_File, TreeCtrlIcon_File+1,
		new MyTreeItemData(item_str));
	m_newIds.push_back(id);
	MyTreeItemData* mti_data = (MyTreeItemData*)GetItemData(m_lastItem);
	SolidCtrl_Sptr& sc = ((FirstMain*)GetParent())->m_SolidCtrl;
	wxString wxstr = mti_data->GetDesc();
	SolidView_Sptr spView;
	if (wxstr == wxT("Bounding Box"))
	{
		SEffect_Sptr Setting = SEffect::New(SEffect::BOUNDING_BOX);
		spView = sc->NewSEffect(Setting);
	}
	else if (wxstr == wxT("Vertex"))
	{
		SEffect_Sptr Setting = SEffect::New(SEffect::VERTEX);
		spView = sc->NewSEffect(Setting);
	}
	else if (wxstr == wxT("Isosurface Contour"))
	{
		SEffect_Sptr Setting = SEffect::New(SEffect::CONTOUR);
		spView = sc->NewSEffect(Setting);
	}
	else if (wxstr == wxT("Axes"))
	{
		SEffect_Sptr Setting = SEffect::New(SEffect::CONTOUR);
		spView = sc->NewSEffect(Setting);
	}
	else if (wxstr == wxT("Ruler"))
	{
		SEffect_Sptr Setting = SEffect::New(SEffect::AXES);
		spView = sc->NewSEffect(Setting);
	}
	else if (wxstr == wxT("Plane Chip"))
	{
		SEffect_Sptr Setting = SEffect::New(SEffect::PLANE_CHIP);
		spView = sc->NewSEffect(Setting);
	}
	else if (wxstr == wxT("Contour Chip"))
	{
		SEffect_Sptr Setting = SEffect::New(SEffect::CONTOUR_CHIP);
		spView = sc->NewSEffect(Setting);
	}
	else if (wxstr == wxT("Volume Render"))
	{
		SEffect_Sptr Setting = SEffect::New(SEffect::VOLUME_RENDERING);
		spView = sc->NewSEffect(Setting);
	}
	MyTreeItemData* new_data = (MyTreeItemData*)GetItemData(id);
	new_data->SetView(spView);
	((FirstMain*)GetParent())->m_ActiveView = spView;
	ChangeGrid(mti_data->GetDesc());
	Expand(m_lastItem);
	event.Skip();
}

void MyTreeCtrl::OnMenu_DeleteItem( wxCommandEvent& event )
{
	if (GetRootItem() == GetItemParent(m_lastItem) || GetRootItem() == m_lastItem)
	{
		wxMessageDialog add_dialog(NULL, wxT("Prime item can't Delete"), wxT("Failed"));
		add_dialog.ShowModal();
	}
	else
	{
		wxTreeItemIds::iterator it =  find(m_newIds.begin(), m_newIds.end(), m_lastItem);
		m_newIds.erase(it);
		MyTreeItemData* mti_data = (MyTreeItemData*)GetItemData(m_lastItem);
		((FirstMain*)GetParent())->m_SolidCtrl->RmView(mti_data->GetView());
		mti_data->RmView();
		Delete(m_lastItem);
	}
	event.Skip();
}


/*
 * wxEVT_COMMAND_TREE_SEL_CHANGED event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlSelChanged( wxTreeEvent& event )
{
	SetLastItem(event.GetItem());
	wxTreeItemId wxid = m_lastItem;
	MyTreeItemData* mti_data = (MyTreeItemData*)GetItemData(wxid);
	while (!ChangeGrid(mti_data->GetDesc()) && wxid != GetRootItem())
	{
		wxid = GetItemParent(wxid);
		mti_data = (MyTreeItemData*)GetItemData(wxid);
		((FirstMain*)GetParent())->m_ActiveView = mti_data->GetView();
	}
	event.Skip();
}

bool MyTreeCtrl::ChangeGrid( const wxString& wxstr )
{
	if (wxstr == wxT("Bounding Box"))
	{
		((FirstMain*)GetParent())->m_grid->ConvertTo_BoundingBox();
		return true;
	}
	else if (wxstr == wxT("Vertex"))
	{
		((FirstMain*)GetParent())->m_grid->ConvertTo_Vertex();
		return true;
	}
	else if (wxstr == wxT("Isosurface Contour"))
	{
		((FirstMain*)GetParent())->m_grid->ConvertTo_IsosurfaceContour();
		return true;
	}
	else if (wxstr == wxT("Axes"))
	{
		((FirstMain*)GetParent())->m_grid->ConvertTo_Axes();
		return true;
	}
	else if (wxstr == wxT("Ruler"))
	{
		((FirstMain*)GetParent())->m_grid->ConvertTo_Ruler();
		return true;
	}
	else if (wxstr == wxT("Plane Chip"))
	{
		((FirstMain*)GetParent())->m_grid->ConvertTo_PlaneChip();
		return true;
	}
	else if (wxstr == wxT("Contour Chip"))
	{
		((FirstMain*)GetParent())->m_grid->ConvertTo_ContourChip();
		return true;
	}
	else if (wxstr == wxT("Volume Render"))
	{
		((FirstMain*)GetParent())->m_grid->ConvertTo_VolumeRender();
		return true;
	}
	else
		return false;
}


/*
 * wxEVT_COMMAND_TREE_ITEM_ACTIVATED event handler for ID_TREECTRL
 */

void MyTreeCtrl::OnTreectrlItemActivated( wxTreeEvent& event )
{
	wxTreeItemId wxid = m_lastItem;
	MyTreeItemData* mti_data = (MyTreeItemData*)GetItemData(wxid);
	while (!ChangeGrid(mti_data->GetDesc()) && wxid != GetRootItem())
	{
		wxid = GetItemParent(wxid);
		mti_data = (MyTreeItemData*)GetItemData(wxid);
		((FirstMain*)GetParent())->m_ActiveView = mti_data->GetView();
	}
	event.Skip();
}

void MyTreeCtrl::RmAllAddItem()
{
	for (wxTreeItemIds::iterator it = m_newIds.begin();
		it != m_newIds.end();
		it++)
	{
		MyTreeItemData* mti_data = (MyTreeItemData*)GetItemData(*it);
		mti_data->GetView()->SetVisable(false);
		Delete(*it);
	}
	m_newIds.clear();
}


void MyTreeItemData::RmView()
{
	m_View->SetVisable(false);
	SolidView_Sptr tmp;
	tmp.swap(m_View);
}
