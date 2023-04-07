#include"stream.h"
#include"physicsfield.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

namespace lbm
{
	namespace stream
	{

		__global__ void streambase2(double* __restrict__ dst,const double* __restrict__ src, int Xwidth, int x1, int y1, int x2, int y2, int e1, int e2)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			if (x > x1 && x < x2 && y > y1 && y < y2)
			{
				dst[x + e1 + y * Xwidth + e2 * Xwidth] = src[x + y * Xwidth];
			}
		}

		__global__ void streamEdgeBKbased2(double* __restrict__ dst, const double* __restrict__ src, int e1, int e2, int ratio, int xratio,int yratio,int points)
		{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			if (x < points)
			{
				dst[x * ratio + e1 * xratio + e2 * yratio] = src[x * ratio];
			}
		}

		constexpr int delta1 = 32;

		extern "C"
		{
			void streamBlockd2(field::Field& dstDistri,field::Field &srcDistri, AbstractVelocitySet* VelocitySet)
			{

				for (int i = 1; i < dstDistri.size(); i++)
				{
					int e1, e2;
					e1 = VelocitySet->getV1(i); e2 = VelocitySet->getV2(i);
					int X = (srcDistri[i]->Xwidth() + delta1 - 1) / delta1, Y = (srcDistri[i]->Ywidth() + delta1 - 1) / delta1;
					streambase2 << <dim3(X,Y),dim3(delta1,delta1) >> >
						((double*)dstDistri[i]->Dptr(),(double*)srcDistri[i]->Dptr()
						,srcDistri[i]->Xwidth(),
						0,0,
						srcDistri[i]->Xwidth()-1,srcDistri[i]->Ywidth()-1,
						e1,e2);
				}
			}

			void streamEdged2(field::Field& dstDistri, field::Field& srcDistri,AbstractVelocitySet* velocitySet, int position)
			{
				int Xwidth = dstDistri[0]->Xwidth(), Ywidth = dstDistri[0]->Ywidth();
				//int X = (dstDistri[0]->Xwidth() + delta1 - 1) / delta1;
				
				int X;
				if (position % 2) X = (Ywidth + delta1 - 1) / delta1;
				else X = (Xwidth + delta1 - 1) / delta1;
				
				////<<<X,delta1>>>
				std::vector<int> direction;
				std::vector<int> e1;
				std::vector<int> e2;
				direction.reserve((dstDistri.size() + 1) / 2);
				e1.reserve((dstDistri.size() + 1) / 2);
				e2.reserve((dstDistri.size() + 1) / 2);

				int start = 0;
				
				auto statusSave = [&](int i) 
				{
					direction.push_back(i);
					e1.push_back(velocitySet->getV1(i));
					e2.push_back(velocitySet->getV2(i)); 
				};
				
				using member = int (AbstractVelocitySet::*) (int);
				auto gq = std::greater_equal<int>();
				auto lq = std::less_equal<int>();

				auto conditional = [&](int _i,auto compare)
				{
					member checkV = &AbstractVelocitySet::getV2;
					if (position % 2) checkV = &AbstractVelocitySet::getV1;

					if (compare((velocitySet->*checkV)(_i), 0))
						statusSave(_i);
				};

				for (int i = 1; i < velocitySet->size(); i++){
					if (position < 3) conditional(i, lq);
					else conditional(i, gq);
				}
					

				switch (position)
				{
				case 1:
					start = Xwidth * 2 - 1;break;
				case 2:
					start = Xwidth * Ywidth - Xwidth + 1;break;
				case 3:
					start = Xwidth;break;
				case 4:
					start = 1;break;
				default:
					throw; break;
				}

				for (int i = 0; i < direction.size(); i++)
				{
					streamEdgeBKbased2 << <X, delta1 >> >
					((double*)dstDistri[direction[i]]->Dptr() + start, (double*)srcDistri[direction[i]]->Dptr() + start,
						e1[i], e2[i], (position % 2 ? Xwidth : 1), 1, Xwidth,(position % 2 ? Ywidth : Xwidth) - 2);
				}
			}
		}

	}//end of namespace stream
}//end of namespace lbm

