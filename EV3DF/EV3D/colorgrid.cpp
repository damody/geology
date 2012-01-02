// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
// Created:     19/09/2011 17:09:25

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

////@begin includes
////@end includes

#include "DW/SolidDefine.h"
#include "DW/SEffect.h"
#include "DW/SolidView.h"
#include "colorgrid.h"
////@begin XPM images
////@end XPM images


/*
 * ColorGrid type definition
 */

IMPLEMENT_DYNAMIC_CLASS( ColorGrid, wxGrid )


/*
 * ColorGrid event table definition
 */

BEGIN_EVENT_TABLE( ColorGrid, wxGrid )

////@begin ColorGrid event table entries
    EVT_GRID_CELL_CHANGE( ColorGrid::OnCellChange )

////@end ColorGrid event table entries

END_EVENT_TABLE()


/*
 * ColorGrid constructors
 */

ColorGrid::ColorGrid()
{
    Init();
}

ColorGrid::ColorGrid(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
    Init();
    Create(parent, id, pos, size, style);
}


/*
 * ColorGrid creator
 */

bool ColorGrid::Create(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
{
////@begin ColorGrid creation
    wxGrid::Create(parent, id, pos, size, style);
    CreateControls();

    ReCreateGrid(1, 4);
    SetColLabelValue(0, wxT("R"));
    SetColLabelValue(1, wxT("G"));
    SetColLabelValue(2, wxT("B"));
    SetColLabelValue(3, wxT("Value"));

    SetRowLabelValue(0, wxT("1"));
////@end ColorGrid creation
    return true;
}


/*
 * ColorGrid destructor
 */

ColorGrid::~ColorGrid()
{
////@begin ColorGrid destruction
////@end ColorGrid destruction
}


/*
 * Member initialisation
 */

void ColorGrid::Init()
{
////@begin ColorGrid member initialisation
////@end ColorGrid member initialisation
}


/*
 * Control creation for ColorGrid
 */

void ColorGrid::CreateControls()
{    
////@begin ColorGrid content construction
////@end ColorGrid content construction
}


/*
 * Should we show tooltips?
 */

bool ColorGrid::ShowToolTips()
{
    return true;
}

/*
 * Get bitmap resources
 */

wxBitmap ColorGrid::GetBitmapResource( const wxString& name )
{
    // Bitmap retrieval
////@begin ColorGrid bitmap retrieval
    wxUnusedVar(name);
    return wxNullBitmap;
////@end ColorGrid bitmap retrieval
}

/*
 * Get icon resources
 */

wxIcon ColorGrid::GetIconResource( const wxString& name )
{
    // Icon retrieval
////@begin ColorGrid icon retrieval
    wxUnusedVar(name);
    return wxNullIcon;
////@end ColorGrid icon retrieval
}




/*
 * wxEVT_GRID_CELL_CHANGED event handler for ID_GRID2
 */

void ColorGrid::OnCellChange( wxGridEvent& event )
{
////@begin wxEVT_GRID_CELL_CHANGED event handler for ID_GRID2 in ColorGrid.
    // Before editing this code, remove the block markers.
    event.Skip();
////@end wxEVT_GRID_CELL_CHANGED event handler for ID_GRID2 in ColorGrid. 
}

void ColorGrid::DeleteGrid()
{
	while (GetNumberRows())
	{
		DeleteRows();
	}
	while (GetNumberCols())
	{
		DeleteCols();
	}
}

void ColorGrid::AppendGrid( int Cols, int Rows )
{
	while (Cols--)
		AppendCols();
	while (Rows--)
		AppendRows();
}

void ColorGrid::ReCreateGrid( int Cols, int Rows )
{
	DeleteGrid();
	//AppendGrid( Rows, Cols );
}

void ColorGrid::ChangeToView( int col, int row, const wxString& data )
{

	return;
}

bool ColorGrid::ChangeColorGrid( SolidView_Sptr& view )
{
	if (view.get() == 0) // ÂI¿ù¿ï¶µ
		return false;
// 	switch (view->GetType())
// 	{
// 	case SEffect::BOUNDING_BOX:
// 		{
// 			ConvertTo_BoundingBox();
// 			return true;
// 		}
// 	case SEffect::CLIP_PLANE:
// 		{
// 			ConvertTo_ClipPlane();
// 			const ClipPlane_Setting* seffect = (ClipPlane_Setting*)(view->GetEffect().get());
// 			switch (seffect->m_Axes)
// 			{
// 			case 0:
// 				SetCellValue(0, 0, _T("X Axes"));
// 				break;
// 			case 1:
// 				SetCellValue(0, 0, _T("Y Axes"));
// 				break;
// 			case 2:
// 				SetCellValue(0, 0, _T("Z Axes"));
// 				break;
// 			}
// 			wxString setvalue;
// 			setvalue << seffect->m_Percent;
// 			SetCellValue(1, 0, setvalue);
// 			return true;
// 		}
// 	}
// 	ReCreateGrid(1,1);
// 	SetColLabelValue(0, wxT("Value"));
// 	int i=0;
// 	SetRowLabelValue(i++, wxT("Color"));


	std::vector<Color3Val> colortable = view->GetEffect()->m_ColorPoints;
	int cpnum = colortable.size();
	ReCreateGrid(cpnum+1, 4);

	wxString setvalue;
	for (int i=0; i<cpnum; i++)
	{
		SetRowLabelValue(i,  wxString::Format(wxT("%i"), i+1));
		setvalue = wxString::Format(wxT("%i"), (int)colortable[i].r);
		SetCellValue(i, 0, setvalue);
		setvalue = wxString::Format(wxT("%i"), (int)colortable[i].g);
		SetCellValue(i, 1, setvalue);
		setvalue = wxString::Format(wxT("%i"), (int)colortable[i].b);
		SetCellValue(i, 2, setvalue);

		setvalue = wxString::Format(wxT("%f"), (int)colortable[i].val);
		SetCellValue(i, 3, setvalue);
	}


	return true;
}

void ColorGrid::InitGrid()
{
	ReCreateGrid(7, 1);

	int i=0;
	SetRowLabelValue(i++, _T("¬õ"));
	SetRowLabelValue(i++, _T("¼á"));
	SetRowLabelValue(i++, _T("¶À"));
	SetRowLabelValue(i++, _T("ºñ"));
	SetRowLabelValue(i++, _T("ÂÅ"));
	SetRowLabelValue(i++, _T("ÀQ"));
	SetRowLabelValue(i++, _T("µµ"));

	SetColLabelValue(0, wxT("Value"));	// ¶b¦Vªº²Ê²Ó
// 	wxString setvalue;
// 	for (int i=0; i<7; i++)
// 	{
// 		SetRowLabelValue(i,  wxString::Format(wxT("%i"), i+1));
// 		setvalue = wxString::Format(wxT("%i"), (int)colortable[i].r);
// 		SetCellValue(i, 0, setvalue);
// 		setvalue = wxString::Format(wxT("%i"), (int)colortable[i].g);
// 		SetCellValue(i, 1, setvalue);
// 		setvalue = wxString::Format(wxT("%i"), (int)colortable[i].b);
// 		SetCellValue(i, 2, setvalue);
// 
// 		setvalue = wxString::Format(wxT("%f"), (int)colortable[i].val);
// 		SetCellValue(i, 3, setvalue);
// 	}
}

bool ColorGrid::ChangeGrid( SolidView_Sptr& view )
{
	InitGrid();
	return true;
}

// author: t1238142000@gmail.com Liang-Shiuan Huang ¶À«G°a
// author: a910000@gmail.com Kuang-Yi Chen ³¯¥ú«³
// In academic purposes only(2012/1/12)
