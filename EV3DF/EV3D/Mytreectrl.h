/////////////////////////////////////////////////////////////////////////////
// Name:        mytreectrl.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     04/10/2010 04:59:20
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _MYTREECTRL_H_
#define _MYTREECTRL_H_


/*!
 * Includes
 */

////@begin includes
#include "wx/treectrl.h"
////@end includes

/*!
 * Forward declarations
 */

////@begin forward declarations
class MyTreeCtrl;
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_TREECTRL 10067
#define SYMBOL_MYTREECTRL_STYLE wxTR_EDIT_LABELS|wxTR_SINGLE
#define SYMBOL_MYTREECTRL_IDNAME ID_TREECTRL
#define SYMBOL_MYTREECTRL_SIZE wxDefaultSize
#define SYMBOL_MYTREECTRL_POSITION wxDefaultPosition
////@end control identifiers
enum
{
	TreeCtrlMenu_AddItem,
	TreeCtrlMenu_DeleteItem
};

/*!
 * MyTreeCtrl class declaration
 */

class MyTreeCtrl: public wxTreeCtrl
{    
    DECLARE_DYNAMIC_CLASS( MyTreeCtrl )
    DECLARE_EVENT_TABLE()

public:
	enum
	{
		TreeCtrlIcon_File,
		TreeCtrlIcon_FileSelected,
		TreeCtrlIcon_Folder,
		TreeCtrlIcon_FolderSelected,
		TreeCtrlIcon_FolderOpened
	};
    /// Constructors
    MyTreeCtrl();
    MyTreeCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTR_HAS_BUTTONS);

    /// Creation
    bool Create(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTR_HAS_BUTTONS);

    /// Destructor
    ~MyTreeCtrl();

    /// Initialises member variables
    void Init();

    /// Creates the controls and sizers
    void CreateControls();

////@begin MyTreeCtrl event handler declarations

    /// wxEVT_COMMAND_TREE_SEL_CHANGED event handler for ID_TREECTRL
    void OnTreectrlSelChanged( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_SEL_CHANGING event handler for ID_TREECTRL
    void OnTreectrlSelChanging( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_DELETE_ITEM event handler for ID_TREECTRL
    void OnTreectrlDeleteItem( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_ITEM_ACTIVATED event handler for ID_TREECTRL
    void OnTreectrlItemActivated( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_ITEM_COLLAPSED event handler for ID_TREECTRL
    void OnTreectrlItemCollapsed( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_ITEM_COLLAPSING event handler for ID_TREECTRL
    void OnTreectrlItemCollapsing( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_ITEM_EXPANDED event handler for ID_TREECTRL
    void OnTreectrlItemExpanded( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_ITEM_EXPANDING event handler for ID_TREECTRL
    void OnTreectrlItemExpanding( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_KEY_DOWN event handler for ID_TREECTRL
    void OnTreectrlKeyDown( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_ITEM_MENU event handler for ID_TREECTRL
    void OnTreectrlItemMenu( wxTreeEvent& event );

    /// wxEVT_COMMAND_TREE_ITEM_RIGHT_CLICK event handler for ID_TREECTRL
    void OnTreectrlItemRightClick( wxTreeEvent& event );

////@end MyTreeCtrl event handler declarations

////@begin MyTreeCtrl member function declarations

////@end MyTreeCtrl member function declarations
	void SetLastItem(wxTreeItemId id) { m_lastItem = id; }
	void OnMenu_AddItem(wxCommandEvent& event);
	void OnMenu_DeleteItem(wxCommandEvent& event);
	void CreateImageList(int size = 16);
	void ShowMenu(wxTreeItemId id, const wxPoint& pt);
	bool ChangeGrid(const wxString& wxstr);
////@begin MyTreeCtrl member variables
////@end MyTreeCtrl member variables
	int          m_imageSize;               // current size of images
	bool         m_reverseSort;             // flag for OnCompareItems
	wxTreeItemId m_lastItem,                // for OnEnsureVisible()
		m_draggedItem;             // item being dragged right now
};

class MyTreeItemData : public wxTreeItemData
{
public:
	MyTreeItemData(const wxString& desc) : m_desc(desc) { }

	void ShowInfo(wxTreeCtrl *tree);
	const wxString& GetDesc() const { return m_desc; }

private:
	wxString m_desc;
};

/* XPM */
static const char * icon1_xpm[] = {
	/* columns rows colors chars-per-pixel */
	"32 32 41 1",
	"> c #97C4E7",
	"# c #4381AA",
	"d c #FFFFFF",
	"< c #71B2DE",
	"+ c #538BB1",
	"& c #D1E5F5",
	"q c #63B3DE",
	"6 c #F1F4F7",
	"* c #CAE1F3",
	"y c #7AC4E5",
	"= c #C3DDF1",
	"X c #74A1BD",
	"- c #BCD9EF",
	"5 c #619BC4",
	"3 c #E6EAF1",
	"2 c #4B8EBF",
	"o c #6B97B6",
	". c #4B82A8",
	"  c None",
	"w c #54A6D8",
	"1 c #71A8D1",
	", c #85BBE2",
	"t c #EFF6FC",
	"7 c #DEEDF8",
	"@ c #4388B4",
	"a c #F7FBFD",
	"$ c #D7E0E9",
	"r c #FAFCFE",
	"4 c #DAEAF7",
	"e c #E9F3FA",
	"0 c #76BAE2",
	"% c #7FA6C0",
	"s c #FDFDFE",
	"O c #5896BE",
	"p c #B6D5EE",
	"8 c #87ABC3",
	": c #A5CCEA",
	"9 c #E5F0F9",
	"; c #AFD1EC",
	"i c #F4F9FD",
	"u c #8FB0C3",
	/* pixels */
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"        .XXXooOO++@#$           ",
	"        %&*=-;:>>,<123          ",
	"        %4&*=-;:>>,1>56         ",
	"        %74&*=-;:>>1*>56        ",
	"        89700qqqqwq1e*>X        ",
	"        8e974&*=-;:1re*>8       ",
	"        8te974&*=-;11111#       ",
	"        8tty000qqqqqww>,+       ",
	"        uitte974&*=-p:>>+       ",
	"        uaitte974&*=-p:>O       ",
	"        uaayyyy000qqqqp:O       ",
	"        uraaitte974&*=-po       ",
	"        urraaitte974&*=-o       ",
	"        usryyyyyyy000q*=X       ",
	"        ussrraaitte974&*X       ",
	"        udssrraaitte974&X       ",
	"        uddyyyyyyyyyy074%       ",
	"        udddssrraaitte97%       ",
	"        uddddssrraaitte9%       ",
	"        udddddssrraaitte8       ",
	"        uddddddssrraaitt8       ",
	"        uuuuuuuuuuuuuu88u       ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                "
};
/* XPM */
static const char * icon2_xpm[] = {
	/* columns rows colors chars-per-pixel */
	"32 32 15 1",
	". c Black",
	"O c #97C4E7",
	"$ c #63B3DE",
	"@ c #CAE1F3",
	"; c #7AC4E5",
	"* c #74A1BD",
	"+ c #619BC4",
	"o c #4B8EBF",
	"  c None",
	"% c #54A6D8",
	"= c #FAFCFE",
	"& c #E9F3FA",
	"# c #76BAE2",
	"X c #C00000",
	"- c #87ABC3",
	/* pixels */
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"        .............           ",
	"        .XXXXXXXXXX.o.          ",
	"        .XXXXXXXXXX.O+.         ",
	"        .XXXXXXXXXX.@O+.        ",
	"        .XX##$$$$%$.&@O*        ",
	"        .XXXXXXXXXX.=&@O-       ",
	"        .XXXXXXXXXX......       ",
	"        .XX;###$$$$$%%XX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .XX;;;;###$$$$XX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .XX;;;;;;;###$XX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .XX;;;;;;;;;;#XX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .XXXXXXXXXXXXXXX.       ",
	"        .................       ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                "
};
/* XPM */
static const char * icon3_xpm[] = {
	/* columns rows colors chars-per-pixel */
	"32 32 41 1",
	"6 c #EDF2FB",
	"- c #AAC1E8",
	": c #B9CDED",
	"X c #295193",
	", c #C6D6F0",
	"a c #4A7CCE",
	"u c #779DDB",
	"y c #7FA2DD",
	"$ c #3263B4",
	"5 c #EAF0FA",
	". c #2D59A3",
	"o c #6E96D8",
	"* c #356AC1",
	"r c #F7F9FD",
	"> c #BED0EE",
	"3 c #E1E9F7",
	"7 c #F0F5FC",
	"< c #CBD9F1",
	"2 c #DAE5F6",
	"# c #3161B1",
	"  c None",
	"0 c #FDFEFF",
	"= c #9FB9E5",
	"e c #AEC5EA",
	"t c #89A9DF",
	"q c #98B5E4",
	"p c #5584D1",
	"d c #3A70CA",
	"@ c #305FAC",
	"i c #5D89D3",
	"1 c #D2DFF4",
	"% c #3366B9",
	"9 c #FAFCFE",
	"8 c #F5F8FD",
	"s c #4075CC",
	"O c #638ED5",
	"w c #90AFE2",
	"& c #3467BC",
	"+ c #2F5DA9",
	"; c #B3C8EB",
	"4 c #E5EDF9",
	/* pixels */
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"      ......X                   ",
	"      .oooooO+                  ",
	"      .ooooooo.                 ",
	"      .+@@@##$%%&&&&&****.      ",
	"      .=-;:>,<12345678900.      ",
	"      .q=-;:>,<1234567890.      ",
	"      .wq=-e:>,<12345678r.      ",
	"      .twq=-e:>,<12345678.      ",
	"      .ytwq=-e:>,<1234567.      ",
	"      .uytwq=-e:>,<123456.      ",
	"      .ouytwq=-e:>,<12345.      ",
	"      .Oouytwq=-e;>,<1234.      ",
	"      .iOouytwq=-e;>,<123.      ",
	"      .piOouytwq=-e;>,<12.      ",
	"      .apiOouytwq=-e;>,<1.      ",
	"      .sapiOouytwq=-e;>,<.      ",
	"      .dsapiOouytwq=-e;>,.      ",
	"      ...................#      ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                "
};
/* XPM */
static const char * icon4_xpm[] = {
	/* columns rows colors chars-per-pixel */
	"32 32 5 1",
	". c Black",
	"o c #8399B4",
	"X c #8DA0B9",
	"  c None",
	"O c #800000",
	/* pixels */
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"      .......                   ",
	"      .XXXXXo.                  ",
	"      .XXXXXXX.                 ",
	"      ....................      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      .OOOOOOOOOOOOOOOOOO.      ",
	"      ....................      ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                "
};
/* XPM */
static const char * icon5_xpm[] = {
	/* columns rows colors chars-per-pixel */
	"32 32 41 1",
	"0 c #AAC1E8",
	"q c #B9CDED",
	"X c #295193",
	"e c #C6D6F0",
	"a c #4A7CCE",
	"& c #779DDB",
	"* c #7FA2DD",
	"2 c #EAF0FA",
	"@ c #2D59A3",
	"o c #6E96D8",
	"y c #356AC1",
	"d c #214279",
	"w c #BED0EE",
	"= c #85A7DF",
	"< c #E1E9F7",
	"3 c #F0F5FC",
	"s c #CBD9F1",
	", c #DAE5F6",
	"7 c #3161B1",
	"  c None",
	". c #274D8B",
	"6 c #FDFEFF",
	"i c #E7EEF9",
	"9 c #9FB9E5",
	"- c #89A9DF",
	"8 c #98B5E4",
	"$ c #5584D1",
	"+ c #3569BF",
	"% c #305FAC",
	"O c #5D89D3",
	"> c #D2DFF4",
	"p c #3366B9",
	"5 c #FAFCFE",
	"4 c #F5F8FD",
	"t c #4075CC",
	"u c #638ED5",
	"r c #CEDCF2",
	"; c #90AFE2",
	"# c #2F5DA9",
	": c #B3C8EB",
	"1 c #E5EDF9",
	/* pixels */
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"     ......X                    ",
	"     XoooooO.                   ",
	"     Xoooooo+.                  ",
	"     Xooooooo@XXXXXXXXXX#       ",
	"     Xoooooooooooooooooo#       ",
	"     Xoooooooooooooooooo#       ",
	"     Xoo$###################    ",
	"     Xoo%O&*=-;:>,<123445667    ",
	"     XooX890:qwer>,<123445q#    ",
	"     Xoty;890:qwer>,<12344#     ",
	"     Xo%u-;890:qwer>,<i234#     ",
	"     XoX&*-;890:qwer>,<i2r#     ",
	"     Xtpo&*-;890:qwer>,<i#      ",
	"     X%auo&*-;890:qwer>,<#      ",
	"     XX$Ouo&*-;890:qwer>s#      ",
	"     d%a$Ouo&*-;890:qwer#       ",
	"     d+ta$Ouo&*-;890:qwe#       ",
	"     d..................#       ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                ",
	"                                "
};


#endif
    // _MYTREECTRL_H_
