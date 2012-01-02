// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
//  @ File Name : ColorTable.h
//  @ Date : 2010/3/18
//  @ Author : 


#if !defined(_COLORTABLE_H)
#define _COLORTABLE_H
#include <map>

// for color use
struct Color4 {
public:
	union  {
		struct  {
			unsigned char r,g,b,a;
		};
		unsigned char c[4];
	};
	Color4(){}
	Color4(unsigned char ir,unsigned char ig,unsigned char ib,unsigned char ia):a(ia),r(ir),g(ig),b(ib){}
	// for mix color
	Color4 GetMixColor(const Color4 & ic, float percent ) // this*(1-percent) + input*percent
	{
		Color4 res = Color4((unsigned char)(ic.r*percent+r*(1-percent)),
			unsigned char(ic.g*percent+g*(1-percent)),
			unsigned char(ic.b*percent+b*(1-percent)),
			unsigned char(ic.a*percent+a*(1-percent)));
		return res;
	}
};
// for color table use
class ColorTable 
{
public:
	typedef std::map<float,Color4> ctMap;
	int size;
	ctMap vMapping;
	void push_back(const float num, const Color4 c)
	{
		vMapping.insert(std::make_pair(num, c));
	}
	void clear()
	{
		vMapping.clear();
	}
	inline Color4 GetColor4(float c)
	{
		int i = 0;
		ctMap::iterator it, lit;
		for (it = vMapping.begin();it != vMapping.end();it++)
		{
			if (it->first > c)
			{
				if (i == 0)
					return it->second;
				else
				{
					return lit->second.GetMixColor(it->second, (c - lit->first)/(it->first - lit->first));
				}
			}
			lit = it;
			i++;
		}
		return (--it)->second;
	}
};

#endif  //_COLORTABLE_H
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
