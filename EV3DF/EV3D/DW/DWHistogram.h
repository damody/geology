﻿// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
//  @ Date : 2010/4/26
// for use Histogram
// you will want know middle number of this set
// or what is number of 97% ?

#if !defined(_DWHISTOGRAM_H)
#define _DWHISTOGRAM_H
#include <vector>
#include <algorithm>

// you need use sort when you sort done
// TODO: use auto sort
template <class T>
class DWHistogram {
public:
	typedef std::vector<T> Tvector;
	std::vector<T> m_data;
	DWHistogram() {}
	DWHistogram(T* data, int num)
	{
		m_data = Tvector(data, data+num);
		std::sort(m_data.begin(), m_data.end());
	}
	template <class iter>
	DWHistogram(iter start, iter end)
	{
		m_data = Tvector(start, end);
		std::sort(m_data.begin(), m_data.end());
	}
	double Find(T key);
	T GetPersentValue(double p)
	{
		if (p<=0) return *(m_data.begin());
		if (p>=1) return *(m_data.end()-1);
		unsigned int i = (m_data.size()-1)*p;
		return m_data[i];
	}
	unsigned int size();
	void AddData(T* dst, int size, int step=1);
	void Append(T data) {m_data.push_back(data);}
	void Sort()
	{
		std::sort(m_data.begin(), m_data.end());
	}
	void Clear();
};
typedef DWHistogram<double> Histogramd;
typedef DWHistogram<float> Histogramf;
typedef DWHistogram<int> Histogrami;

#endif  //_DWHISTOGRAM_H
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
