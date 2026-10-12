#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace ohy {

	// Model property (CLIP), not domain: same for cars, flowers, birds.
	struct PreprocessConfig {
		int size = 224;
		float mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
		float std[3] = { 0.26862954f, 0.26130258f, 0.27577711f };
	};

	std::vector<uint8_t> ReadFileBytes(const std::string& path);

	// Decodes JPEG/PNG bytes and writes one normalised [3, size, size] image to dst (device memory).
	// Matches CLIP's Python preprocess: Resize(size, BICUBIC) -> CenterCrop(size) -> ToTensor -> Normalize.
	void PreprocessImage(const std::vector<uint8_t>& encoded, const PreprocessConfig& cfg,
		float* dst, cudaStream_t stream);
}