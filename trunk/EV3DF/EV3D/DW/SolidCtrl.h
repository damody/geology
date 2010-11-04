#pragma once
#include <vector>
#include <boost/shared_ptr.hpp>
#include "BoxArea.h"
#include "SolidDefine.h"
/**
control unit
*/
class SolidCtrl
{
public:
	
	SolidViewPtrs	m_SolidViewPtrs;
	SolidDocPtrs	m_SolidDocPtrs;
	enum {
		SET_OK,
		SET_FAIL
	};
	int SetData();
	SolidDoc_Sptr	NewDoc(const BoxArea* area);
	SolidView_Sptr	NewView(const SEffect_Setting* area, SolidDoc* Doc);
	SolidCtrl(void);
	virtual ~SolidCtrl(void);
};
