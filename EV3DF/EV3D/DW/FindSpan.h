﻿#pragma once
bool IsEqual(double x1, double x2, double sigma);
double FindDeltaAndSpan ( double *values, int num, int step, double sigma, double& vmin, double& vmax, double& delta, int& span);
