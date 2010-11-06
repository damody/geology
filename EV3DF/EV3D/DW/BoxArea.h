#pragma once
#pragma warning(disable:4201)
/**
限制要顯示的範圍用的box
*/
class BoxArea
{
public:
	enum
	{
		UP,
		DOWN,
		LEFT,
		RIGHT,
		AHEAD,
		BACK
	};
	union
	{
		struct 
		{
			float m_up;
			float m_down;
			float m_left;
			float m_right;
			float m_ahead;
			float m_back;
		};
		struct 
		{
			float m_limits[6];
		};
	};
	float	m_rangeX,
		m_rangeY,
		m_rangeZ;
	int	m_numX,
		m_numY,
		m_numZ;
};
