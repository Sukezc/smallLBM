#pragma once
#include"physicsfield.h"

namespace lbm
{
	namespace macro
	{

		extern "C"
		{
			void MacroVelocityComputeBlockd2(field::Field& Umacro, field::Field& density, field::Field& distri);

			void MacroDensityComputeBlockd2(field::Field& density, field::Field& distri);
		}
	}
}