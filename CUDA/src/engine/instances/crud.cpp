#include "crud.h"

#include "../../kernels/kernel_utility.h"
#include "../utility.h"
#include "log.h"
#include "../preprocess.h"
#include <sstream>
#include <iomanip>
#include <fstream>

namespace ohy {
	void CRUDInstance::InitSelf(const InitArg& initArg) {

	}
	void CRUDInstance::RunSelf(const RunArg& runArg) {
		

		// Check if all image file exists
		auto& imagePathList = runArg.imagePaths;
		for (auto& imagePaths : imagePathList) {
			std::filesystem::path image_abs_path = std::filesystem::absolute(imagePaths);
			if (!std::filesystem::exists(image_abs_path)) {
				throw std::runtime_error("Image file not found: " + image_abs_path.string());
			}
		}


		// Check if the model has input names
		if (input_names.empty()) {
			throw std::runtime_error("No input names found in the model.");
		}

		for (auto& imagePathStr : imagePathList) {
			std::filesystem::path image_abs_path = std::filesystem::absolute(imagePathStr);
			logging::Infof("Processing image: %s", image_abs_path.c_str());

			std::string imagePath = image_abs_path.string();
			OpContext ctx;

			// Input shape comes from the graph. The dynamo export bakes the batch size in, so every dim is fixed.
			const std::vector<int64_t> inShape = InputShape();

			auto input = std::make_shared<Tensor>();
			input->dtype = DataType::FLOAT32;
			input->shape = inShape;
			input->storage = std::make_shared<Buffer>(Location::Device, input->numel() * sizeof(float));

			PreprocessConfig cfg;
			cfg.size = static_cast<int>(inShape[3]);
			PreprocessImage(ReadFileBytes(imagePath), cfg, input->f32(), ctx.stream);

			// Fixed batch: fill the other slots with copies of the image; use row 0 of the output.
			const int64_t slot = inShape[1] * inShape[2] * inShape[3];
			for (int64_t n = 1; n < inShape[0]; ++n) {
				CUDA_CHECK(cudaMemcpyAsync(input->f32() + n * slot, input->f32(), slot * sizeof(float),
					cudaMemcpyDeviceToDevice, ctx.stream));
			}
			env[input_names[0]] = std::move(input);


			// Write to file for checking
			std::filesystem::path path(imagePath);
			std::vector<float> h(slot);
			CUDA_CHECK(cudaMemcpy(h.data(), env[input_names[0]]->f32(), slot * sizeof(float), cudaMemcpyDeviceToHost));
			std::ofstream(path.string() + ".bin", std::ios::binary).write(reinterpret_cast<const char*>(h.data()), h.size() * sizeof(float));


			auto outputMap = RunGraph(runArg, ctx);

			//TODO: do something with output
		}



	}
}