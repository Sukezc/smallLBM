#include"macrography.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

namespace lbm
{
	namespace macro
	{

		__global__ void computeDensitybase(double* __restrict__ density,
			const double* __restrict__ distri1, const double* __restrict__ distri2, const double* __restrict__ distri3,
			const double* __restrict__ distri4, const double* __restrict__ distri5, const double* __restrict__ distri6,
			const double* __restrict__ distri7, const double* __restrict__ distri8, const double* __restrict__ distri9,
			int Xwidth, int x1, int y1,int x2,int y2)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int ind = x + y * Xwidth;
			if (x > x1 && x < x2 && y > y1 && y < y2)
			{
				density[ind] =
					distri1[ind] + distri2[ind] + distri3[ind] +
					distri4[ind] + distri5[ind] + distri6[ind] +
					distri7[ind] + distri8[ind] + distri9[ind];
			}
		}
		
		__global__ void computeVelocitybase(double* __restrict__ Umacro1,double* __restrict__ Umacro2,const double* __restrict__ density,
			const double* __restrict__ distri1, const double* __restrict__ distri2, const double* __restrict__ distri3,
			const double* __restrict__ distri4, const double* __restrict__ distri5, const double* __restrict__ distri6,
			const double* __restrict__ distri7, const double* __restrict__ distri8,
			int Xwidth, int x1, int y1, int x2, int y2)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int ind = x + y * Xwidth;
			if (x > x1 && x < x2 && y > y1 && y < y2)
			{
				Umacro1[ind] = (distri1[ind] + distri5[ind] + distri8[ind] - distri3[ind] - distri6[ind] - distri7[ind])/density[ind];
				Umacro2[ind] = (distri2[ind] + distri5[ind] + distri6[ind] - distri4[ind] - distri7[ind] - distri8[ind])/density[ind];
			}

		}

		constexpr int delta1 = 32;

		extern "C"
		{
			void MacroVelocityComputeBlockd2(field::Field& Umacro,field::Field& density,field::Field& distri)
			{
				int X = (density[0]->Xwidth() + delta1 - 1) / delta1, Y = (density[0]->Ywidth() + delta1 - 1) / delta1;
				computeVelocitybase << <dim3(X, Y), dim3(delta1, delta1) >> >
				((double*)Umacro[0]->Dptr(), (double*)Umacro[1]->Dptr(), (double*)density[0]->Dptr(),
					(double*)distri[1]->Dptr(), (double*)distri[2]->Dptr(), (double*)distri[3]->Dptr(),
					(double*)distri[4]->Dptr(), (double*)distri[5]->Dptr(), (double*)distri[6]->Dptr(),
					(double*)distri[7]->Dptr(), (double*)distri[8]->Dptr(),
					Umacro[0]->Xwidth(), -1, -1, Umacro[0]->Xwidth(), Umacro[0]->Ywidth());
			}

			void MacroDensityComputeBlockd2(field::Field& density,field::Field& distri)
			{
				int X = (density[0]->Xwidth() + delta1 - 1) / delta1, Y = (density[0]->Ywidth() + delta1 - 1) / delta1;
				computeDensitybase << <dim3(X, Y), dim3(delta1, delta1) >> >
				((double*)density[0]->Dptr(),
					(double*)distri[0]->Dptr(), (double*)distri[1]->Dptr(), (double*)distri[2]->Dptr(),
					(double*)distri[3]->Dptr(), (double*)distri[4]->Dptr(), (double*)distri[5]->Dptr(),
					(double*)distri[6]->Dptr(), (double*)distri[7]->Dptr(), (double*)distri[8]->Dptr(),
					density[0]->Xwidth(), -1, -1, density[0]->Xwidth(), density[0]->Ywidth());
				
				
			}
		}
	}
}