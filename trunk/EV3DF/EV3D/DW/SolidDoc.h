#pragma once
#include "SolidDefine.h"
/**

*/
class SolidDoc
{
public:
	SolidDoc*	GetParentDoc();
	SolidCtrl*	GetParentCtrl();
	virtual ~SolidDoc(void);
private:
	SolidDoc(const BoxArea* area);
	SolidDoc*	m_ParentDoc;
	SolidCtrl*	m_ParentCtrl;
	friend SolidCtrl;
	friend SolidView;
};
