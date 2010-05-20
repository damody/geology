/////////////////////////////////////////////////////////////////////////////
// Name:        earthviewer3dapp.cpp
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     03/03/2010 16:25:54
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

// For compilers that support precompilation, includes "wx/wx.h".
#include "stdwx.h"
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

////@begin includes
////@end includes

#include "earthviewer3dapp.h"

////@begin XPM images
////@end XPM images


/*
 * Application instance implementation
 */

////@begin implement app
IMPLEMENT_APP( EarthViewer3DApp )
////@end implement app


/*
 * EarthViewer3DApp type definition
 */

IMPLEMENT_CLASS( EarthViewer3DApp, wxApp )


/*
 * EarthViewer3DApp event table definition
 */

BEGIN_EVENT_TABLE( EarthViewer3DApp, wxApp )

////@begin EarthViewer3DApp event table entries
////@end EarthViewer3DApp event table entries

END_EVENT_TABLE()


/*
 * Constructor for EarthViewer3DApp
 */

EarthViewer3DApp::EarthViewer3DApp()
{
    Init();
}


/*
 * Member initialisation
 */

void EarthViewer3DApp::Init()
{
////@begin EarthViewer3DApp member initialisation
////@end EarthViewer3DApp member initialisation
}

/*
 * Initialisation for EarthViewer3DApp
 */

bool EarthViewer3DApp::OnInit()
{    
////@begin EarthViewer3DApp initialisation
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
	FirstMain* mainWindow = new FirstMain( NULL );
	mainWindow->Show(true);
////@end EarthViewer3DApp initialisation

    return true;
}


/*
 * Cleanup for EarthViewer3DApp
 */

int EarthViewer3DApp::OnExit()
{    
////@begin EarthViewer3DApp cleanup
	return wxApp::OnExit();
////@end EarthViewer3DApp cleanup
}

