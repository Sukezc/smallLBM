#pragma once
#include"cuda_runtime.h"
#include"physicsfield.h"
namespace lbm
{
	namespace collision
	{

		__host__ __device__ double feqbase(double omega, int e1, int e2, double rho, double u1, double u2, double c_reciprocal = 1.0);

		extern "C"
		{
			void collisionBlockd2(field::Field& distri, field::Field& Rho, field::Field& uMacro, AbstractVelocitySet* VelocitySet, field::Weightfunc& Weight, double dx, double dt);
		}
	}
}