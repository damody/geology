/////////////////////////////////////////////////////////////////////////////
// Name:        colorgrid.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     19/09/2011 17:09:25
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _COLORGRID_H_
#define _COLORGRID_H_


/*!
 * Includes
 */

#include "DW/SolidDefine.h"
#include "DW/SEffect.h"
#include "DW/SolidView.h"

////@begin includes
#include "wx/grid.h"
// #include "mygrid.h"
////@end includes

/*!
 * Forward declarations
 */

////@begin forward declarations
class ColorGrid;
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
#define ID_GRID2 10051
#define SYMBOL_COLORGRID_STYLE wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL
#define SYMBOL_COLORGRID_IDNAME ID_GRID2
#define SYMBOL_COLORGRID_SIZE wxSize(200, 150)
#define SYMBOL_COLORGRID_POSITION wxDefaultPosition
////@end control identifiers


/*!
 * ColorGrid class declaration
 */

class ColorGrid: public wxGrid
{    
    DECLARE_DYNAMIC_CLASS( ColorGrid )
    DECLARE_EVENT_TABLE()

public:
    /// Constructors
    ColorGrid();
    ColorGrid(wxWindow* parent, wxWindowID id = ID_GRID2, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(200, 150), long style = wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL);

    /// Creation
    bool Create(wxWindow* parent, wxWindowID id = ID_GRID2, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(200, 150), long style = wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL);

    /// Destructor
    ~ColorGrid();

    /// Initialises member variables
    void Init();
    void InitGrid();

    /// Creates the controls and sizers
    void CreateControls();

////@begin ColorGrid event handler declarations

    /// wxEVT_GRID_CELL_CHANGE event handler for ID_GRID2
    void OnCellChange( wxGridEvent& event );

////@end ColorGrid event handler declarations

////@begin ColorGrid member function declarations

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end ColorGrid member function declarations

    /// Should we show tooltips?
    static bool ShowToolTips();


	void DeleteGrid();
	void AppendGrid(int Cols, int Rows);
	void ReCreateGrid(int Cols, int Rows);

	void ChangeToView(int col, int row, const wxString& data);

	bool ChangeColorGrid(SolidView_Sptr& view);
	bool ChangeGrid(SolidView_Sptr& view);

////@begin ColorGrid member variables
////@end ColorGrid member variables
};

#endif
    // _COLORGRID_H_
