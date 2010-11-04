#include "StdVtk.h"
#include "SolidCtrl.h"
#include "SolidDoc.h"
#include "SolidView.h"

SolidCtrl::SolidCtrl(void)
{
}

SolidCtrl::~SolidCtrl(void)
{
}

SolidDoc_Sptr	SolidCtrl::NewDoc( const BoxArea* area )
{
	SolidDoc_Sptr tmpPtr(new SolidDoc(area));
	m_SolidDocPtrs.push_back(tmpPtr);
	return tmpPtr;
}

SolidView_Sptr	SolidCtrl::NewView( const SEffect_Setting* effect, SolidDoc* Doc )
{
	SolidView_Sptr tmpPtr(new SolidView(Doc));
	m_SolidViewPtrs.push_back(tmpPtr);
	return tmpPtr;
}
