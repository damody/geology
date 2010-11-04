#pragma once
class Solid;
class SolidDoc;
class SolidView;
class SolidCtrl;
class SEffect;
class SEffect_Setting;
class BoxArea;
#include <boost/shared_ptr.hpp>
typedef boost::shared_ptr<SolidView>	SolidView_Sptr;
typedef boost::shared_ptr<SolidDoc>	SolidDoc_Sptr;
typedef std::vector<SolidView_Sptr>	SolidViewPtrs;
typedef std::vector<SolidDoc_Sptr>	SolidDocPtrs;