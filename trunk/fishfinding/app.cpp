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
#include "StdVtkWx.h"
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

#include <windows.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <fcntl.h>
#include <io.h>

void RedirectIOToConsole()
{
	using namespace std;
	int hConHandle;
	long lStdHandle;
	CONSOLE_SCREEN_BUFFER_INFO coninfo;
	FILE *fp;
	// allocate a console for this app
	AllocConsole();
	// set the screen buffer to be big enough to let us scroll text
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &coninfo);
	coninfo.dwSize.Y = 1024;
	SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE),
		coninfo.dwSize);
	// redirect unbuffered STDOUT to the console
	lStdHandle = (long)GetStdHandle(STD_OUTPUT_HANDLE);
	hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
	fp = _fdopen( hConHandle, "w" );
	*stdout = *fp;
	setvbuf( stdout, NULL, _IONBF, 0 );
	// redirect unbuffered STDIN to the console
	lStdHandle = (long)GetStdHandle(STD_INPUT_HANDLE);
	hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
	fp = _fdopen( hConHandle, "r" );
	*stdin = *fp;
	setvbuf( stdin, NULL, _IONBF, 0 );
	// redirect unbuffered STDERR to the console
	lStdHandle = (long)GetStdHandle(STD_ERROR_HANDLE);
	hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
	fp = _fdopen( hConHandle, "w" );
	*stderr = *fp;
	setvbuf( stderr, NULL, _IONBF, 0 );
	// make cout, wcout, cin, wcin, wcerr, cerr, wclog and clog
	// point to console as well
	ios::sync_with_stdio();
}

/*
 * Constructor for FishfindingApp
 */

FishfindingApp::FishfindingApp()
{
    Init();
    RedirectIOToConsole();
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

