/*! @brief �o�O²��гy���������O
*/

#ifndef Win32Window_Im40eda407m1229c278a93mm7e5f_H
#define Win32Window_Im40eda407m1229c278a93mm7e5f_H

/** 
*@file Windows ���Y��
*
* Windows �ϥΰʧ@�禡�p�U
*/ 
#include <windows.h>

class Win32Window
{	
public:
	Win32Window();
	~Win32Window();
	/**@brief 
	*�гy�@�ӵ���
	*/
	HWND	ToCreateWindow(int x, int y, int width, int height, const wchar_t *title, LRESULT (_stdcall *wndporc)(HWND, UINT, WPARAM, LPARAM) = &Win32Window::Proc);
	HWND	ToFullScreen(int width, int height, WNDPROC = &Win32Window::Proc);
	MSG	HandlePeekMessage();
	/**
	* @brief
	* Handle �����
	*/
	HWND		GetHandle();
	HINSTANCE	GetInstance();
	RECT		GetRect();
	/**
	* @brief
	* �o��ƹ����A
	*/
	void	GetMouseState(int &x, int &y, int button[3]);
	/**
	* @brief
	* ��ܵ���
	*/
	void	ToShow();
	/**
	* @brief
	* ���õ���
	*/
	void	ToHide();
	/**
	* @brief
	* ���ʪ�����
	*/
	void	ToMoveCenter();
	/**
	* @brief
	* ���ʨ�ϰ�
	*/
	void	ToMove(UINT x, UINT y, UINT nWidth, UINT nHeight, bool bRepaint = true);
	
	/*! �o��window��Style */
	DWORD	GetStyle() {return m_style;}
	/*! �]�wwindow��Style */
	void	SetStyle(DWORD style) {m_style = style;}
	/*! �o��window��ExStyle */
	DWORD	GetExStyle() {return m_EX_style;}
	/*! �]�wwindow��ExStyle */
	void	SetExStyle(DWORD style) {m_EX_style = style;}
	/*! �o����������e */
	void	Get_W_H(int &w, int &h) {w = m_Width; h = m_Height;}
	/*! �w�]���ƥ�B�z��� */
	static LRESULT CALLBACK Proc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
	{
		switch (message)
		{
		case WM_DESTROY:
			PostQuitMessage(WM_QUIT);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		return 0;
	}
private:
	int m_Height,m_Width;
	HWND m_hWnd;
	HINSTANCE m_hInstance;
	DWORD m_style;
	DWORD m_EX_style;
};


#endif // Win32Window_Im40eda407m1229c278a93mm7e5f_H

