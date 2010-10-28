#pragma once
#pragma warning(disable:4201)
/**
����n��ܪ��d��Ϊ�box
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
	BoxArea(void);
	~BoxArea(void);
	union
	{
		float m_up;
		float m_down;
		float m_left;
		float m_right;
		float m_ahead;
		float m_back;
		struct 
		{
			float limits[6];
		};
	};
	
};