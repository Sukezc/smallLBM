#pragma once
#include"mesh.h"
#include<vector>
#include<memory>
#include<thrust/execution_policy.h>

namespace lbm
{
	namespace field
	{
		class Field
		{
		public:
			std::vector<std::unique_ptr<AbstractMesh>> field;

			Field() = default;
			~Field() = default;

			Field(const Field& other) = delete;
			Field(Field&& other) = default;

			Field& operator=(const Field& other) = delete;
			Field& operator=(Field&& other) = default;

			AbstractMesh* operator[](int pos) { return this->field[pos].get(); }
			
			AbstractMesh* at(int pos){
				if (pos<field.size())return this->field[pos].get();
				else throw;
			}

			template<typename ...Args>
			void fillField(AbstractMesh&& mesh, Args&&... args)
			{
				this->field.emplace_back(&mesh);
				if constexpr (sizeof...(args) == 0)return;
				else this->fillField(std::forward<Args>(args)...);
			}

			template<typename Type,typename ...Args>
			void fillField(std::unique_ptr<Type>&& ptr, Args&&... args)
			{
				this->field.emplace_back(std::move(ptr));
				if constexpr (sizeof...(args) == 0)return;
				else this->fillField(std::forward<Args>(args)...);
			}

			template<typename Type>
			void rangefillField(int num, int _X, int _Y)
			{
				for (int i = 0; i < num; i++)
				{
					this->fillField(std::make_unique<lbm::mesh::Mesh<Type>>(_X, _Y));
				}
			}

			Field& copy(const Field& other)
			{
				if (&other == this)return *this;
				else
				{
					if (other.field.size() != this->field.size())throw;
					else
					{
						for (int i = 0; i < this->field.size(); i++)
						{
							if (this->field[i]->Xwidth() != other.field[i]->Xwidth() ||
								this->field[i]->Ywidth() != other.field[i]->Ywidth() ||
								this->field[i]->Datasize() != other.field[i]->Datasize())throw;
							else
							{
								thrust::copy(thrust::host,
									static_cast<char*>(other.field[i]->Hptr()),
									static_cast<char*>(other.field[i]->Hptr()) + other.field[i]->Datasize() * other.field[i]->Xwidth() * other.field[i]->Ywidth(),
									static_cast<char*>(this->field[i]->Hptr()));
								thrust::copy(thrust::device,
									static_cast<char*>(other.field[i]->Dptr()),
									static_cast<char*>(other.field[i]->Dptr()) + other.field[i]->Datasize() * other.field[i]->Xwidth() * other.field[i]->Ywidth(),
									static_cast<char*>(this->field[i]->Dptr()));
							}
						}
						return *this;
					}
				}
			}

			Field& copyfast(const Field& other)
			{
				for (int i = 0; i < this->field.size(); i++)
				{
					thrust::copy(thrust::host, 
						static_cast<char*>(other.field[i]->Hptr()),
						static_cast<char*>(other.field[i]->Hptr()) + other.field[i]->Datasize() * other.field[i]->Xwidth() * other.field[i]->Ywidth(),
						static_cast<char*>(this->field[i]->Hptr()));
					thrust::copy(thrust::device,
						static_cast<char*>(other.field[i]->Dptr()),
						static_cast<char*>(other.field[i]->Dptr()) + other.field[i]->Datasize() * other.field[i]->Xwidth() * other.field[i]->Ywidth(),
						static_cast<char*>(this->field[i]->Dptr()));
				}
				return *this;
			}

			size_t size()
			{
				return this->field.size();
			}

		};// end of the defination of the class Field

		class VelositySet : public AbstractVelocitySet
		{
		public:
			std::vector<int> set;
			thrust::device_vector<int> Dset;
			int dimension;
			VelositySet():dimension(0){}
			~VelositySet(){}
			
			void Initial(const std::string name)
			{
				if (name == "D2Q9")
				{
					dimension = 2;
					set = { 
						0,0,
						1,0,
						0,1,
						-1,0,
						0,-1,
						1,1,
						-1,1,
						-1,-1,
						1,-1 };
					Dset = thrust::device_vector<int>(set.begin(), set.end());
				}
				else if (name == "D2Q5")
				{
					dimension = 2;
					set = {
						0,0,
						1,0,
						0,1,
						-1,0,
						0,-1};
					Dset = thrust::device_vector<int>(set.begin(), set.end());
				}
			}
			
			int getV1(int pos) override
			{
				return set[pos*dimension];
			}

			int getV2(int pos) override
			{
				return set[pos*dimension + 1];
			}

			int size() override
			{
				return set.size() / dimension;
			}

		};// end of class VelocitySet

		class Weightfunc : public AbstractWeight
		{
		public:
			std::vector<double> weight;
			thrust::device_vector<double> Dweight;
			
			Weightfunc(){}
			~Weightfunc(){}

			void Initial(const std::string name)
			{
				if (name == "D2Q9")
				{
					weight =
					{
						4.0/9,
						1.0/9, 1.0/9, 1.0/9, 1.0/9,
						1.0/36, 1.0/36, 1.0/36, 1.0/36
					};
					Dweight = thrust::device_vector<double>(weight.begin(), weight.end());
				}
			}

			double operator[](int pos)
			{
				return weight[pos];
			}
		};

	}// end of namespace field
}// end of namespace lbm
