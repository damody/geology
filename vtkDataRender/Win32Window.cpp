/**
 * @file Win32Window �Բӻ���
 *
 * �o�O�@�ӱM�����͵��������O�A���U���o�ˡA�B�z�T������ơA���ݱq�~�����ѵ�ToCreateWindow�A�_�h�|�ϥιw�]����ơC
 *LRESULT CALLBACK Proc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
 *		{
 *			switch (message)
 *				{
 *					case WM_DESTROY:
 *					PostQuitMessage(0);
 *					break;
 *					default:
 *					return DefWindowProc(hWnd, message, wParam, lParam);
 *				}
 *			return 0;
 *		}
 *ToCreateWindow���e������ƬO�ΨӴy�z�������A�Ĥ����N�O�n���W�������, �O�o�n�x�s�^�Ǫ�HWND����X�C
 *GetHandle�u��o��̫�@�ӵ���������X�C
 *�`�N�I�I�I�b�L���j�餤���ݦۦ�I�sHandlePeekMessage�ӳB�z�T���C
 *GetMouseState�i�H��o�ƹ����A�C
 *
 * 
 */

#include "Win32Window.h"

Win32Window::Win32Window(void):m_style(WS_OVERLAPPEDWINDOW|WS_CAPTION),m_EX_style(WS_EX_OVERLAPPEDWINDOW)
{
	m_hInstance = GetModuleHandle(NULL);
}

HWND Win32Window::ToCreateWindow(int x, int y, int width, int height, const wchar_t *title, 
				 LRESULT (_stdcall *wndporc)(HWND, UINT, WPARAM, LPARAM)/*= &Win32Window::Proc*/)
{
	if (x < 0 || y < 0)
	{
		int w = GetSystemMetrics(SM_CXSCREEN);
		x = w/2-width/2;
		int h = GetSystemMetrics(SM_CYSCREEN);
		y = h/2-height/2;
	}
	m_Height = height;
	m_Width = width;
	WNDCLASS	window_class;
	memset(&window_class, 0, sizeof(WNDCLASS));///<�]�w�������A�C

	window_class.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;///<�]�w�T���B�z�禡��WndProc�C

	window_class.lpfnWndProc = wndporc;///<���o�ثe���檺�{���C

	window_class.hInstance = m_hInstance;///<�ϥΤ��w���ƹ���СC

	window_class.hCursor = LoadCursor(NULL, IDC_ARROW);///<�I���]���¦�, ��꦳�S���]���S�t.�C

	window_class.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);///<���w�@�ӵ��U�����ɨϥΪ��W�r�C
	window_class.hIcon = NULL;
	//window_class.hIcon = LoadIcon(m_hInstance, MAKEINTRESOURCE(IDI_ICON1));
	window_class.lpszClassName = title;///<��@�~�t�ε��U�@�ӷs�������C

	if (RegisterClass(&window_class) == 0)
	{
		return false;
	}
	///<�p���������u�|�������I��, ���n�ɭn�W�[�����j�p�C
	RECT window_rect;
	SetRect(&window_rect, x, y, x+width, y+height);
	AdjustWindowRect(&window_rect, m_style, FALSE); ///<WS_POPUPWINDOW�O��ؽu�����C
   /**
     * @brief
     *
     * �ǳƤu�@����, �}�Ҥ@�ӵ���
     */
	HWND window_handle = CreateWindowEx(
		m_EX_style, ///<WS_EX_TOPMOST
		title,
		title,
		m_style,
		window_rect.left,	///< x
		window_rect.top,	///< y
		window_rect.right - window_rect.left,	///< width
		window_rect.bottom - window_rect.top,	///< height
		NULL,
		NULL,
		window_class.hInstance,
		NULL
		);
	
	if (window_handle == NULL)
		return false;
	m_hWnd = window_handle;	
	return window_handle;
}

HWND Win32Window::ToFullScreen(int width, int height, WNDPROC wndproc/*= &Win32Window::Proc*/)
{
	wchar_t *title = L"fullscreen";
	DEVMODE dmScreenSettings;								///< Device Mode
	memset(&dmScreenSettings,0,sizeof(dmScreenSettings));	///< Makes Sure Memory's Cleared
	dmScreenSettings.dmSize=sizeof(dmScreenSettings);		///< Size Of The Devmode Structure
	dmScreenSettings.dmPelsWidth	= width;				///< Selected Screen Width
	dmScreenSettings.dmPelsHeight	= height;				///< Selected Screen Height
	dmScreenSettings.dmBitsPerPel	= 32;					///< Selected Bits Per Pixel
	dmScreenSettings.dmFields=DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT;

	///< Try To Set Selected Mode And Get Results.  NOTE: CDS_FULLSCREEN Gets Rid Of Start Bar.
	if (ChangeDisplaySettings(&dmScreenSettings,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL)
	{
		///< If The Mode Fails, Offer Two Options.  Quit Or Use Windowed Mode.
		return false;
	}


	WNDCLASS	window_class;

	memset(&window_class, 0, sizeof(WNDCLASS));

	///< �]�w�������A
	window_class.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
	///< �]�w�T���B�z�禡��WndProc
	window_class.lpfnWndProc = wndproc;
	///< ���o�ثe���檺�{��
	window_class.hInstance = GetModuleHandle(NULL);
	///< �ϥΤ��w���ƹ����
	window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
	///< �I���]���¦�, ��꦳�S���]���S�t
	window_class.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	///< ���w�@�ӵ��U�����ɨϥΪ��W�r
	window_class.lpszClassName = title;
	///< ��@�~�t�ε��U�@�ӷs������
	if (RegisterClass(&window_class) == 0)
		return 0;
#ifdef _DEBUG
	DWORD dwExStyle=WS_EX_APPWINDOW;	// Window Extended Style
#else
	DWORD dwExStyle=WS_EX_APPWINDOW;	// can't see other windows
#endif	
	DWORD dwStyle=WS_POPUP;			///< Windows Style
	ShowCursor(FALSE);			///< Hide Mouse Pointer

	RECT	WindowRect;			///< Grabs Rectangle Upper Left / Lower Right Values

	WindowRect.left=(long)0;		///< Set Left Value To 0
	WindowRect.right=(long)width;		///< Set Right Value To Requested Width
	WindowRect.top=(long)0;			///< Set Top Value To 0
	WindowRect.bottom=(long)height;		///< Set Bottom Value To Requested Height

	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);		///< Adjust Window To True Requested Size

	HWND window_handle = CreateWindowEx(	
		dwExStyle,
		L"fullscreen",
		title,
		dwStyle |
		WS_CLIPSIBLINGS |
		WS_CLIPCHILDREN,
		0, 0,
		WindowRect.right-WindowRect.left,
		WindowRect.bottom-WindowRect.top,
		NULL,
		NULL,
		window_class.hInstance,
		NULL
		);

	if (window_handle == NULL)
		return false;

	
	m_hWnd = window_handle;///< �O���U�ΨӥN�������ثe����{��������, �H��i�H�A�Ψ�
	m_hInstance = window_class.hInstance;
	ShowWindow(window_handle, SW_SHOWNORMAL);///< ��ܵ���
	SetActiveWindow(window_handle);
	SetForegroundWindow(window_handle);	
	SetFocus(window_handle);

	return m_hWnd;
}

MSG Win32Window::HandlePeekMessage()
{
	MSG msg;
	// �D�T���j��:
	if ( PeekMessage(&msg, m_hWnd, 0, 0, PM_REMOVE) )
	{	
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}///< ���T������, �N�⥦�e���T���B�z�禡
	return msg;
}

HWND Win32Window::GetHandle(void)
{
	return m_hWnd;
}
HINSTANCE Win32Window::GetInstance(void)
{
	return m_hInstance;
}
void Win32Window::GetMouseState(int &x, int &y, int button[3])
{
	POINT p;
	GetCursorPos(&p);
	ScreenToClient(m_hWnd, &p);
	x = p.x;
	y = p.y;
	button[0] = GetKeyState(VK_LBUTTON) & 0x80 ? 1 : 0;
	button[1] = GetKeyState(VK_MBUTTON) & 0x80 ? 1 : 0;
	button[2] = GetKeyState(VK_RBUTTON) & 0x80 ? 1 : 0;
}

void Win32Window::ToShow(void)
{
	ShowWindow(m_hWnd, SW_SHOWNORMAL);///< ��ܵ���
	SetActiveWindow(m_hWnd);
}
void Win32Window::ToHide(void)
{
	
	ShowWindow(m_hWnd, SW_HIDE);///< ��ܵ���
	SetActiveWindow(m_hWnd);
}

void Win32Window::ToMove( UINT x, UINT y, UINT nWidth, UINT nHeight, bool bRepaint /*= true*/ )
{
	MoveWindow(m_hWnd , x, y, nWidth, nHeight, bRepaint);
}

RECT Win32Window::GetRect()
{
	RECT rect;
	GetWindowRect(m_hWnd, &rect);
	return rect;
}

Win32Window::~Win32Window()
{
	DestroyWindow( m_hWnd );
}

void Win32Window::ToMoveCenter()
{
	int w = GetSystemMetrics(SM_CXSCREEN);
	int x = w/2-m_Width/2;
	int h = GetSystemMetrics(SM_CYSCREEN);
	int y = h/2-m_Height/2;
	MoveWindow(m_hWnd , x, y, m_Width, m_Height, true);
}
