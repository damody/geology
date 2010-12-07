/////////////////////////////////////////////////////////////////////////////
// Name:        app.h
// Purpose:     
// Author:      damody
// Modified by: 
// Created:     19/11/2010 20:43:16
// RCS-ID:      
// Copyright:   ntust
// Licence:     
/////////////////////////////////////////////////////////////////////////////

#ifndef _APP_H_
#define _APP_H_


/*!
 * Includes
 */

////@begin includes
#include "wx/image.h"
#include "mainframe.h"
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
 * FishfindingApp class declaration
 */

class FishfindingApp: public wxApp
{    
    DECLARE_CLASS( FishfindingApp )
    DECLARE_EVENT_TABLE()

public:
    /// Constructor
    FishfindingApp();

    void Init();

    /// Initialises the application
    virtual bool OnInit();

    /// Called on exit
    virtual int OnExit();

////@begin FishfindingApp event handler declarations

////@end FishfindingApp event handler declarations

////@begin FishfindingApp member function declarations

////@end FishfindingApp member function declarations

////@begin FishfindingApp member variables
////@end FishfindingApp member variables
};

/*!
 * Application instance declaration 
 */

////@begin declare app
DECLARE_APP(FishfindingApp)
////@end declare app

#endif
    // _APP_H_
