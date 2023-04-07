#pragma once
namespace lbm
{
	class AbstractMesh
	{
	public:
		AbstractMesh() {}
		virtual ~AbstractMesh() {}
		virtual void* Hptr() = 0;
		virtual void* Dptr() = 0;
		virtual void fetch() = 0;
		virtual void send() = 0;
		virtual void HValue(void* dst, int offset) = 0;
		virtual void DValue(void* dst, int offset) = 0;
		virtual int Xwidth() = 0;
		virtual int Ywidth() = 0;
		virtual int Datasize() = 0;
		virtual void* Hend() { return nullptr; };
		virtual void* Dend() { return nullptr; };

	};

	class AbstractVelocitySet
	{
	public:
		AbstractVelocitySet(){}
		virtual ~AbstractVelocitySet(){}
		virtual int getV1(int pos) = 0;
		virtual int getV2(int pos) = 0;
		virtual int size() = 0;
		virtual int getV3(int pos) { return 0; };
	};

	class AbstractWeight
	{
	public:
		AbstractWeight(){}
		virtual ~AbstractWeight(){}
		virtual double operator[](int) = 0;
	};
}