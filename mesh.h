#pragma once
#include"Interface.h"
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
namespace lbm
{
	namespace mesh
	{

		template<typename _Ty>
		class Mesh : public AbstractMesh
		{
			thrust::host_vector<_Ty> Hvec;
			thrust::device_vector<_Ty> Dvec;
			int mX, mY;
		public:
			Mesh() :Hvec(), Dvec() {
				mX = 0, mY = 0;
			};
			~Mesh() override { mX = 0; mY = 0; }

			Mesh(int _X, int _Y) :Hvec(_X* _Y), Dvec(_X* _Y)
			{
				mX = _X;
				mY = _Y;
			}

			Mesh(const Mesh<_Ty>& other) = default;
			Mesh(Mesh<_Ty>&& other) = default;

			Mesh<_Ty>& operator=(const Mesh<_Ty>& other)
			{
				if (&other == this) return *this;
				Hvec = other.Hvec;
				Dvec = other.Dvec;
				mX = other.mX;
				mY = other.mY;
				return *this;
			}

			Mesh<_Ty>& operator=(Mesh<_Ty>&& other)
			{
				if (&other == this)return *this;
				Hvec = std::move(other.Hvec);
				Dvec = std::move(other.Dvec);
				mX = other.mX;
				other.mX = 0;
				mY = other.mY;
				other.mY = 0;
				return *this;
			}

			_Ty operator()(int _X, int _Y) {
				return Hvec[_X * mY + _Y];
			}

			_Ty at(int _X, int _Y) {
				if (_X * mY + _Y >= Hvec.size())
				{
					throw;
				}
				else
					return Hvec[_X * mY + _Y];
			}

			void fetch() override {
				Hvec = Dvec;
			}

			void send() override {
				Dvec = Hvec;
			}

			void* Hptr() override {
				return static_cast<void*>(Hvec.data());
			}

			void* Dptr() override {
				return static_cast<void*>(Dvec.data().get());
			}

			void HValue(void* dst, int offset) override
			{
				*((_Ty*)dst) = this->Hvec[offset];
			}

			void DValue(void* dst, int offset) override
			{
				*((_Ty*)dst) = this->Dvec[offset];
			}

			int Xwidth() override
			{
				return mX;
			}

			int Ywidth() override
			{
				return mY;
			}

			int Datasize() override
			{
				return sizeof(_Ty);
			}

			void* Hend() override
			{
				return Hvec.data() + mX * mY;
			}

			void* Dend() override
			{
				return Dvec.data().get() + mX * mY;
			}
		};
	}//end of namespace mesh

}// end of namespace lbm