#ifndef _GUT_H_
#define _GUT_H_

#include "Matrix4x4.h"


// left hand coord system
Matrix4x4 GutMatrixLookAtLH(Vector4 &eye, Vector4 &lookat, Vector4 &up)
{
	Vector4 up_normalized = VectorNormalize(up);
	Vector4 zaxis = (lookat - eye); zaxis.Normalize();
	Vector4 xaxis = Vector3CrossProduct(up_normalized, zaxis);
	Vector4 yaxis = Vector3CrossProduct(zaxis, xaxis);

	Matrix4x4 matrix; 
	matrix.Identity();

	matrix.SetColumn(0, xaxis);
	matrix.SetColumn(1, yaxis);
	matrix.SetColumn(2, zaxis);
	matrix[3][0] = -Vector3Dot(xaxis, eye)[0];
	matrix[3][1] = -Vector3Dot(yaxis, eye)[0];
	matrix[3][2] = -Vector3Dot(zaxis, eye)[0];

	return matrix;
}

// right hand coord system
// eye = `鏡頭位置`
// lookat = `鏡頭對準的位置`
// up = `鏡頭正上方的方向`
Matrix4x4 GutMatrixLookAtRH(Vector4 &eye, Vector4 &lookat, Vector4 &up)
{
	Vector4 up_normalized = VectorNormalize(up);
	Vector4 zaxis = (eye - lookat); zaxis.Normalize();
	Vector4 xaxis = Vector3CrossProduct(up_normalized, zaxis);
	Vector4 yaxis = Vector3CrossProduct(zaxis, xaxis);

	Matrix4x4 matrix; 
	matrix.Identity();

	matrix.SetColumn(0, xaxis);
	matrix.SetColumn(1, yaxis);
	matrix.SetColumn(2, zaxis);
	matrix[3][0] = -Vector3Dot(xaxis, eye)[0];
	matrix[3][1] = -Vector3Dot(yaxis, eye)[0];
	matrix[3][2] = -Vector3Dot(zaxis, eye)[0];

	return matrix;
}

// Direct3D native left hand system
// fovy = 垂直方向的視角
// aspect = 水平方向視角對重直方向視角的比值
// z_hear = 鏡頭可以看到的最近距離
// z_far = 鏡頭可以看到的最遠距離
Matrix4x4 GutMatrixPerspective_DirectX(float fovy, float aspect, float z_near, float z_far)
{
	Matrix4x4 matrix;
	matrix.Identity();

	float fovy_radian = FastMath::DegreeToRadian(fovy);
	float yscale = FastMath::Cot(fovy_radian/2.0f);
	float xscale = yscale * aspect;

	matrix[0][0] = xscale;
	matrix[1][1] = yscale;
	matrix[2][2] = z_far / (z_far - z_near);
	matrix[2][3] = 1.0f;
	matrix[3][2] = -(z_near * z_far) /(z_far - z_near);
	matrix[3][3] = 0.0f;

	return matrix;
}

// w = `鏡頭水平方向可以看到的範圍`
// h = `鏡頭垂直方向可以看到的範圍`
// z_hear = `鏡頭可以看到的最近距離`
// z_far  = `鏡頭可以看到的最遠距離`
Matrix4x4 GutMatrixOrthoRH_DirectX(float w, float h, float z_near, float z_far)
{
	Matrix4x4 matrix;
	matrix.Identity();

	matrix[0][0] = 2.0f/w;
	matrix[1][1] = 2.0f/h;
	matrix[2][2] = 1.0f/(z_near - z_far);
	matrix[3][2] = z_near / (z_near - z_far);

	return matrix;
}

// fovy = 垂直方向的視角
// aspect = `水平方向視角對重直方向視角的比值`
// z_hear = `鏡頭可以看到的最近距離`
// z_far = `鏡頭可以看到的最遠距離`
Matrix4x4 GutMatrixPerspectiveRH_DirectX(float fovy, float aspect, 
					 float z_near, float z_far)
{
	Matrix4x4 matrix;
	matrix.Identity();

	float fovy_radian = FastMath::DegreeToRadian(fovy);
	float yscale = FastMath::Cot(fovy_radian/2.0f);
	float xscale = yscale * aspect;

	matrix[0][0] = xscale;
	matrix[1][1] = yscale;
	matrix[2][2] = z_far / (z_near - z_far);
	matrix[2][3] = -1.0f;
	matrix[3][2] = (z_near * z_far) /(z_near - z_far);
	matrix[3][3] = 0.0f;

	return matrix;
}

// OpenGL native right hand system
// fovy = `垂直方向的視角`
// aspect = `水平方向視角對重直方向視角的比值`
// z_hear = `鏡頭可以看到的最近距離`
// z_far = `鏡頭可以看到的最遠距離`
Matrix4x4 GutMatrixOrthoRH_OpenGL(float w, float h, float z_near, float z_far)
{
	Matrix4x4 matrix;
	matrix.Identity();

	matrix[0][0] = 2.0f/w;
	matrix[1][1] = 2.0f/h;
	matrix[2][2] = 2.0f/(z_near - z_far);
	matrix[3][2] = (z_far + z_near)/(z_near - z_far);
	matrix[3][3] = 1.0f;

	return matrix;
}

Matrix4x4 GutMatrixOrtho_OpenGL(float w, float h, float z_near, float z_far)
{
	return GutMatrixOrthoRH_OpenGL(w, h, z_near, z_far);
}

// OpenGL native right hand system
// fovy = `垂直方向的視角`
// aspect = `水平方向視角對重直方向視角的比值`
// z_hear = `鏡頭可以看到的最近距離`
// z_far = `鏡頭可以看到的最遠距離`
Matrix4x4 GutMatrixPerspectiveRH_OpenGL(float fovy, float aspect, 
					float z_near, float z_far)
{
	Matrix4x4 matrix;
	matrix.Identity();

	float fovy_radian = FastMath::DegreeToRadian(fovy);
	float yscale =  FastMath::Cot(fovy_radian/2.0f);
	float xscale = yscale * aspect;

	matrix[0][0] = xscale;
	matrix[1][1] = yscale;
	matrix[2][2] = (z_far + z_near)/(z_near - z_far);
	matrix[2][3] = -1.0f;
	matrix[3][2] = 2.0f * z_far * z_near / (z_near - z_far);
	matrix[3][3] = 0.0f;

	return matrix;
}

Matrix4x4 GutMatrixPerspective_OpenGL(float fovy, float aspect, float z_near, float z_far)
{
	return GutMatrixPerspectiveRH_OpenGL(fovy, aspect, z_near, z_far);
}

#endif
