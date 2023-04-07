#pragma once
#include"Interface.h"
#include"physicsfield.h"
#include<cstdio>

namespace lbm
{
	namespace stream
	{
		extern "C"
		{
			void streamBlockd2(field::Field& dstDistri, field::Field& srcDistri, AbstractVelocitySet* VelocitySet);

			void streamEdged2(field::Field& dstDistri, field::Field& srcDistri, AbstractVelocitySet* velocitySet, int position);
		}

	}// end of namespace stream
}//end of namespace lbm