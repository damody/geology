﻿// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
// Created:     04/10/2010 14:37:18

#ifndef _MYGRID_H_
#define _MYGRID_H_


/*!
 * Includes
 */

////@begin includes
#include "wx/grid.h"
////@end includes

/*!
 * Forward declarations
 */

////@begin forward declarations
class MyGrid;
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_GRID 10068
#define SYMBOL_MYGRID_STYLE wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL
#define SYMBOL_MYGRID_IDNAME ID_GRID
#define SYMBOL_MYGRID_SIZE wxDefaultSize
#define SYMBOL_MYGRID_POSITION wxDefaultPosition
////@end control identifiers


/*!
 * MyGrid class declaration
 */

class MyGrid: public wxGrid
{    
    DECLARE_DYNAMIC_CLASS( MyGrid )
    DECLARE_EVENT_TABLE()

public:
    /// Constructors
    MyGrid();
    MyGrid(wxWindow* parent, wxWindowID id = ID_GRID, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL);

    /// Creation
    bool Create(wxWindow* parent, wxWindowID id = ID_GRID, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL);

    /// Destructor
    ~MyGrid();

    /// Initialises member variables
    void Init();

    /// Creates the controls and sizers
    void CreateControls();

////@begin MyGrid event handler declarations

    /// wxEVT_GRID_CELL_CHANGE event handler for ID_GRID
    void OnCellChange( wxGridEvent& event );

    /// wxEVT_GRID_CMD_CELL_CHANGE event handler for ID_GRID
    void OnGridCellChange( wxGridEvent& event );

    /// wxEVT_KEY_DOWN event handler for ID_GRID
    void OnKeyDown( wxKeyEvent& event );

////@end MyGrid event handler declarations
	void ConvertTo_BoundingBox();
	void ConvertTo_Vertex();
	void ConvertTo_Contour();
	void ConvertTo_Axes();
	void ConvertTo_Ruler();
	void ConvertTo_ClipPlane();
	void ConvertTo_ClipContour();
	void ConvertTo_VolumeRender();
	void DeleteGrid();
	void AppendGrid(int Cols, int Rows);
	void ReCreateGrid(int Cols, int Rows);
	void ChangeToView(int col, int row, const wxString& data);
	bool ChangeGrid(SolidView_Sptr& view);
////@begin MyGrid member function declarations

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end MyGrid member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();

////@begin MyGrid member variables
////@end MyGrid member variables
};

#endif
    // _MYGRID_H_
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
