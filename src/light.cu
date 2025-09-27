#include "light.h"

#include "bsdf.h"

__device__ glm::vec3 DirectSampleAreaLight(glm::vec3 view_point, glm::vec3 view_nor, const AreaLight& chosenLight, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf)
{
	GeomType gt = chosenLight.type;

	switch (gt)
	{
	case CUBE:
		{
			thrust::uniform_real_distribution<float> u01(0, 1);
			glm::vec4 randPosLocal(u01(rng), u01(rng), 0.0f, 0.0f);
			glm::vec4 norLocal(0.0f, 0.0f, 1.0f, 0.0f);

			glm::vec3 randPosWorld(chosenLight.transform * randPosLocal);
			glm::vec3 norWorld(chosenLight.transform * norLocal);

			float surfaceArea = 4.0f; // TODO: This seems wrong?
			float areaPDF = 1.0f / surfaceArea;

			glm::vec3 lightToSurface = view_point - randPosWorld;
			float r = glm::length(lightToSurface);
			
			if (r < FLT_EPSILON)
				return glm::vec3(0.0f);

			lightToSurface /= r;
			float cosTheta = glm::dot(norWorld, lightToSurface); // try normalize norWorld if something looks off

			out_wiW = -lightToSurface;
			if (cosTheta < FLT_EPSILON)
			{
				out_pdf = 0.0f;
				return glm::vec3(0.0f);
			}
			else
				out_pdf = (r*r / cosTheta) * areaPDF;

		}
		break;
	default:
		// TODO
	}

	return chosenLight.Le * numLights * chosenLight.color;
}

__device__ glm::vec3 Sample_Li(glm::vec3 view_point, glm::vec3 nor, AreaLight* areaLights, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf)
{
	thrust::uniform_int_distribution<int> iu0N(0, numLights-1);
	int randomLightIndex = iu0N(rng);
	const AreaLight chosenLight = areaLights[randomLightIndex];

	return DirectSampleAreaLight(view_point, nor, chosenLight, numLights, rng, out_wiW, out_pdf);
}
