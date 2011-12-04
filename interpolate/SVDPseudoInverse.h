#pragma once

float* SVDPseudoInverseHost(float *inmat, int M, int N);

float* SVDPseudoInverseDevice(float *inmat, int M, int N);

float* SVDPseudoInverseAllDevice(float *inmat, int M, int N);