/////////////////////////////////////////////////////////////////////////////
// Name:        earthviewer3dapp.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     03/03/2010 16:25:54
// RCS-ID:      
// Copyright:   NTUST
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _EARTHVIEWER3DAPP_H_
#define _EARTHVIEWER3DAPP_H_


/*!
 * Includes
 */

////@begin includes
#include "wx/image.h"
#include "firstmain.h"
////@end includes

/*!
 * Forward declarations
 */

////@begin forward declarations
////@end forward declarations

/*!
 * Control identifiers
 */

////@begin control identifiers
////@end control identifiers

/*!
 * EarthViewer3DApp class declaration
 */

class EarthViewer3DApp: public wxApp
{    
    DECLARE_CLASS( EarthViewer3DApp )
    DECLARE_EVENT_TABLE()

public:
    /// Constructor
    EarthViewer3DApp();

    void Init();

    /// Initialises the application
    virtual bool OnInit();

    /// Called on exit
    virtual int OnExit();

////@begin EarthViewer3DApp event handler declarations

////@end EarthViewer3DApp event handler declarations

////@begin EarthViewer3DApp member function declarations

////@end EarthViewer3DApp member function declarations

////@begin EarthViewer3DApp member variables
////@end EarthViewer3DApp member variables
};

/*!
 * Application instance declaration 
 */

////@begin declare app
DECLARE_APP(EarthViewer3DApp)
////@end declare app

#endif
    // _EARTHVIEWER3DAPP_H_
