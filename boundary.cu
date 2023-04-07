#include"boundary.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

namespace lbm
{
	namespace boundary
	{
		__global__ void Reversebase2(double* __restrict__ dst, const double* __restrict__ src, int ratio,int points)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			if (x < points)
			{
				dst[x*ratio] = src[x*ratio];
			}
		}

		constexpr int delta1 = 32;

		extern "C"
		{
			void BoundaryReversed2(field::Field& distri,AbstractVelocitySet* velocitySet,int position)
			{
				int Xwidth = distri[0]->Xwidth(), Ywidth = distri[0]->Ywidth(),X = 0;
				if (position % 2) X = (Ywidth + delta1 - 1) / delta1;
				else X = (Xwidth + delta1 - 1) / delta1;

				auto lt = std::less<int>(); 
				auto gt = std::greater<int>();
				
				std::vector<int> hash(distri.size(), 0);
				using member = int (AbstractVelocitySet::*) (int);
				
				auto conditional = [&](int _i,int _j,auto compare)
				{
					member checkV = & AbstractVelocitySet::getV2;
					
					if (position%2)checkV = &AbstractVelocitySet::getV1;

					if (compare((velocitySet->*checkV)(_i), 0))
						if (velocitySet->getV1(_j) == -velocitySet->getV1(_i) && velocitySet->getV2(_j) == -velocitySet->getV2(_i))
							hash[_i] = _j;
				};

				for (int i = 1; i < velocitySet->size(); i++)
				{
					for (int j = 1; j < velocitySet->size(); j++)
					{
						if (position < 3)
						{
							conditional(i, j, gt);
						}
						else
						{
							conditional(i, j, lt);
						}
					}
				}

				int start = 0;
				switch (position)
				{
				case 1:
					start = Xwidth * 2 - 1; break;
				case 2:
					start = Xwidth * Ywidth - Xwidth + 1; break;
				case 3:
					start = Xwidth; break;
				case 4:
					start = 1; break;
				default:
					throw; break;
				}

				for (int i = 1; i < hash.size(); i++)
				{
					if (!hash[i])
					{
						Reversebase2 << <X, delta1 >> >
						((double*)distri[hash[i]]->Dptr() + start, (double*)distri[i]->Dptr() + start,
							(position % 2 ? Xwidth : 1),(position % 2 ? Ywidth : Xwidth) - 2);

					}
				}

			}
		}
	}
}