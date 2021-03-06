﻿// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
// Created:     03/03/2010 16:25:54

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
#ifdef _WIN32

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
#endif
/*
 * Constructor for EarthViewer3DApp
 */

EarthViewer3DApp::EarthViewer3DApp()
{
    Init();
#ifdef _WIN32 
    RedirectIOToConsole();
#endif
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
	Taiwan* mainWindow = new Taiwan( NULL );
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

// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
