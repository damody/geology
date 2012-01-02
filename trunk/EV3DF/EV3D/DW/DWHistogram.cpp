// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
//  @ File Name : DWHistogram.cpp
//  @ Date : 2010/4/26

#include "DWHistogram.h"

template <class T>
void DWHistogram<T>::AddData( T* dst, int size, int step )
{
	for (int i=0;i<size;i+=step)
	{
		m_data->push_back(*(dst+i));
	}
	sort(m_data.begin(), m_data.end());
}
template <class T>
unsigned int DWHistogram<T>::size()
{
	return m_data->size();
}
template <class T>
double DWHistogram<T>::Find( T key )
{
	unsigned int i = 0;
	for (Tvector::iterator it = m_data->begin();it != m_data->end();it++)
	{
		if (*it = key) break;
		i++;
	}
	double p = m_data->size();
	p = i/p;
	return p;
}

template <class T>
void DWHistogram<T>::Clear()
{
	m_data.clear();
}
// author: t1238142000@gmail.com Liang-Shiuan Huang 黃亮軒
// author: a910000@gmail.com Kuang-Yi Chen 陳光奕
// In academic purposes only(2012/1/12)
