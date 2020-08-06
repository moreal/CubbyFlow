/*************************************************************************
> File Name: GeometryCacheParser.cpp
> Project Name: CubbyFlow
> This code is based on Jet Framework that was created by Doyub Kim.
> References: https://github.com/doyubkim/fluid-engine-dev
> Purpose: Fluid data parser (pos, xyz)
> Created Time: 2020/06/03
> Copyright (c) 2020, Ji-Hong snowapril
*************************************************************************/
#include <Vox/ParticleLoader.hpp>
#include <Vox/DebugUtils.hpp>
#include <Core/Utils/Logging.h>
#include <Core/Utils/Serialization.h>
#include <Core/Vector/Vector3.h>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstdio>
#include <functional>
#include <limits>

using namespace CubbyFlow;

namespace Vox {
	ParticleLoader::ParticleLoader()
	{
		constexpr float min = std::numeric_limits<float>::min();
		constexpr float max = std::numeric_limits<float>::max();
		_bbMin = Vector3F(max, max, max);
		_bbMax = Vector3F(min, min, min);
	}

	ParticleLoader::~ParticleLoader()
	{
		//! Do nothing.
	}

	void ParticleLoader::LoadParticles(const std::string& format, size_t count)
	{
		using namespace std::placeholders;
		size_t idx = format.find_last_of('.');
		const std::string& extension = format.substr(idx + 1);

		std::function<void(const std::string&)> loadMethod;
		if (extension == "xyz") loadMethod = std::bind(&ParticleLoader::LoadXyzFile, this, std::placeholders::_1);
		else if (extension == "pos") loadMethod = std::bind(&ParticleLoader::LoadPosFile, this, std::placeholders::_1);
		else
		{
			std::cerr << "Unknown Particle File extension [" << extension << "]" << std::endl;
			StackTrace::PrintStack();
			std::abort();
		}

		char baseName[256];
		for (size_t i = 0; i < count; ++i)
		{
			snprintf(baseName, sizeof(baseName), format.c_str(), i);
			loadMethod(baseName);
		}
	}

	size_t ParticleLoader::GetNumberOfFrame() const
	{
		return _frames.size();
	}

    size_t ParticleLoader::GetNumberOfBytes(size_t frameIndex) const
	{
		assert(frameIndex < _frames.size());
		return _frames[frameIndex].size() * sizeof(float);
	}
    void ParticleLoader::CopyParticleData(void* dst, size_t frameIndex) const
	{
		assert(frameIndex < _frames.size());
		const void* src = static_cast<const void*>(_frames[frameIndex].data());
		std::memcpy(dst, src, GetNumberOfBytes(frameIndex));
	}

	void ParticleLoader::LoadPosFile(const std::string& path)
	{
		std::ifstream file(path, std::ifstream::binary);
		if (file.is_open() == false)
	    {
			std::cerr << "Failed to load file [" << path << "]" << std::endl;
			StackTrace::PrintStack();
			std::abort();
	    }
		Array1<Vector3D> tempParticles;
	    const std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
	    Deserialize(buffer, &tempParticles);
		
		const size_t numElements = tempParticles.size() * 3;
		Array1<float> particles(numElements);
		tempParticles.ParallelForEachIndex([&](size_t index){
			const size_t baseIndex = index * 3;
			particles[  baseIndex  ] = static_cast<float>(tempParticles[index].x);
			particles[baseIndex + 1] = static_cast<float>(tempParticles[index].y);
			particles[baseIndex + 2] = static_cast<float>(tempParticles[index].z);

			RenewBoundingBox(tempParticles[index].x, tempParticles[index].y, tempParticles[index].z);
		});
		_frames.push_back(particles);
	    file.close();
	}

	void ParticleLoader::LoadXyzFile(const std::string& path)
	{
		std::ifstream file(path);
		if (file.is_open() == false)
		{
			std::cerr << "Failed to load file [" << path << "]" << std::endl;
			StackTrace::PrintStack();
			std::abort();
		}

		Array1<float> particles;
		std::string line;
		while (std::getline(file, line))
		{
			std::istringstream isstr(line);
			float x, y, z;
			isstr >> x >> y >> z;
			particles.Append(x);
			particles.Append(y);
			particles.Append(z);

			RenewBoundingBox(x, y, z);
		}
		_frames.push_back(particles);
	    file.close();
	}

	void ParticleLoader::RenewBoundingBox(float x, float y, float z)
	{	
		_bbMin.x = std::min(_bbMin.x, x);
		_bbMin.y = std::min(_bbMin.y, y);
		_bbMin.z = std::min(_bbMin.z, z);
		_bbMax.x = std::max(_bbMax.x, x);
		_bbMax.y = std::max(_bbMax.y, y);
		_bbMax.z = std::max(_bbMax.z, z);
	}

	std::tuple<CubbyFlow::Vector3F, CubbyFlow::Vector3F> ParticleLoader::GetBoundingBox() const
	{
		return std::make_tuple(_bbMin, _bbMax);
	}

};


