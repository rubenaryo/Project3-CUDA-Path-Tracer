#include "light.h"

#include "bsdf.h"
#include <cmath>

__device__ glm::vec3 DirectSampleAreaLight(glm::vec3 view_point, glm::vec3 view_nor, const Light& chosenLight, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf)
{
	GeomType gt = chosenLight.geomType;

	switch (gt)
	{
	case GT_RECT:
		{
			float scaleX = chosenLight.scale.x;
			float scaleZ = chosenLight.scale.z;
			thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);
			glm::vec4 randPosLocal(u01(rng), u01(rng), u01(rng), 0.0f);
			randPosLocal = glm::normalize(randPosLocal);
			glm::vec4 norLocal(0.0f, 0.0f, 1.0f, 0.0f);

			glm::vec3 randPosWorld(chosenLight.transform * randPosLocal);
			glm::vec3 norWorld(chosenLight.transform * norLocal);

			float surfaceArea = (scaleX) * (scaleZ);
			float areaPDF = 1.0f / surfaceArea;

			glm::vec3 lightToSurface = view_point - randPosWorld;
			float r2 = glm::dot(lightToSurface, lightToSurface);
			
			//if (r2 < FLT_EPSILON)
			//{
			//	out_pdf = 0.0f;
			//	out_wiW = glm::normalize(-lightToSurface);
			//	return glm::vec3(0.0f);
			//}

			lightToSurface *= glm::inversesqrt(r2); // normalize
			out_wiW = -lightToSurface;
			float cosTheta = glm::abs(glm::dot(norWorld, lightToSurface));

			//if (cosTheta < FLT_EPSILON)
			//	out_pdf = 0.0f;
			//else
				out_pdf = (r2 / cosTheta) * areaPDF;

			return chosenLight.emittance * cosTheta * chosenLight.color;

		}
		break;
	default:
		// TODO
	}

	float Le = chosenLight.emittance;
	return Le * numLights * chosenLight.color;
}

__device__ glm::vec3 Sample_Li(glm::vec3 view_point, glm::vec3 nor, Light* lights, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf)
{
	thrust::uniform_int_distribution<int> iu0N(0, numLights-1);
	int randomLightIndex = iu0N(rng);
	const Light chosenLight = lights[randomLightIndex];

	return DirectSampleAreaLight(view_point, nor, chosenLight, numLights, rng, out_wiW, out_pdf);
}
