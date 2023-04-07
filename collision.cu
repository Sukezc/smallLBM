#include"collision.h"
#include"physicsfield.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>


namespace lbm
{
	namespace collision
	{
		__host__ __device__ double feqbase(const double omega,const int e1, const int e2, const double rho, const double u1, const double u2, const double c_reciprocal)
		{
			double eu, uv, feq;
			eu = e1 * u1 + e2 * u2;
			uv = u1 * u1 + u2 * u2;
			feq = omega * rho * (1.0 + 3.0 * eu * c_reciprocal + 4.5 * eu * eu * c_reciprocal * c_reciprocal - 1.5 * uv * c_reciprocal * c_reciprocal);
			return feq;
		}

		__global__ void collisionbase(double* __restrict__ distri,const double* __restrict__ Rho, 
			const double* __restrict__ u1, const double* __restrict__ u2,
			int e1, int e2,double omega,double tau_re,double c_re,int Xwidth, int x1, int y1, int x2, int y2)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int ind = x + y * Xwidth;
			if (x > x1 && x < x2 && y > y1 && y < y2)
			{
				distri[ind] = distri[ind] * (1.0 - tau_re) + feqbase(omega, e1, e2, Rho[ind], u1[ind], u2[ind],c_re) * tau_re;
			}
		}

		constexpr int delta1 = 32;

		extern "C"
		{
			void collisionBlockd2(field::Field& distri,field::Field& Rho,field::Field& uMacro,AbstractVelocitySet * VelocitySet, field::Weightfunc& Weight,double dx = 1.0, double dt = 1.0)
			{
				double c_reciprocal = dt / dx;
				for (int i = 0; i < distri.size(); i++)
				{
					int e1 = VelocitySet->getV1(i), e2 = VelocitySet->getV2(i);
					double omega = Weight[i];
					int X = (distri[i]->Xwidth() + delta1 - 1) / delta1, Y = (distri[i]->Ywidth() + delta1 - 1) / delta1;
					collisionbase << <dim3(X, Y), dim3(delta1, delta1) >> >
						((double*)distri[i]->Dptr(), (double*)Rho[i]->Dptr(),(double*)uMacro[0]->Dptr(), (double*)uMacro[1]->Dptr(),
							VelocitySet->getV1(i),VelocitySet->getV2(i),
							Weight[i],1.0,c_reciprocal,distri[i]->Xwidth(),
							0, 0,
							distri[i]->Xwidth() - 1, distri[i]->Ywidth() - 1
							);//
				}
			}


		}
	}
}