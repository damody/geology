/////////////////////////////////////////////////////////////////////////////
// Name:        app.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     19/11/2010 20:43:16
// RCS-ID:      
// Copyright:   ntust
// Licence:     
/////////////////////////////////////////////////////////////////////////////

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

#include "app.h"

////@begin XPM images
////@end XPM images


/*
 * Application instance implementation
 */

////@begin implement app
IMPLEMENT_APP( FishfindingApp )
////@end implement app


/*
 * FishfindingApp type definition
 */

IMPLEMENT_CLASS( FishfindingApp, wxApp )


/*
 * FishfindingApp event table definition
 */

BEGIN_EVENT_TABLE( FishfindingApp, wxApp )

////@begin FishfindingApp event table entries
////@end FishfindingApp event table entries

END_EVENT_TABLE()


/*
 * Constructor for FishfindingApp
 */

FishfindingApp::FishfindingApp()
{
    Init();
}


/*
 * Member initialisation
 */

void FishfindingApp::Init()
{
////@begin FishfindingApp member initialisation
////@end FishfindingApp member initialisation
}

/*
 * Initialisation for FishfindingApp
 */

bool FishfindingApp::OnInit()
{    
////@begin FishfindingApp initialisation
	// Remove the comment markers above and below this block
	// to make permanent changes to the code.

#if wxUSE_XPM
	wxImage::AddHandler(new wxXPMHandler);
#endif
#if wxUSE_LIBPNG
	wxImage::AddHandler(new wxPNGHandler);
#endif
#if wxUSE_LIBJPEG
	wxImage::AddHandler(new wxJPEGHandler);
#endif
#if wxUSE_GIF
	wxImage::AddHandler(new wxGIFHandler);
#endif
	mainframe* mainWindow = new mainframe( NULL );
	mainWindow->Show(true);
////@end FishfindingApp initialisation

    return true;
}


/*
 * Cleanup for FishfindingApp
 */

int FishfindingApp::OnExit()
{    
////@begin FishfindingApp cleanup
	return wxApp::OnExit();
////@end FishfindingApp cleanup
}

