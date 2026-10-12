#include "compare.h"

#include "../../kernels/kernel_utility.h"
#include "../utility.h"
#include "log.h"
#include "../preprocess.h"
#include <sstream>
#include <iomanip>

namespace ohy {
	void CompareInstance::InitSelf(const InitArg& initArg) {
		
	}
	void CompareInstance::RunSelf(const RunArg& runArg) {
		if (runArg.imagePaths.empty()) {
			throw std::runtime_error("EmbedImages: no images");
		}

		auto& imagePaths = runArg.imagePaths;
		const std::vector<int64_t> inShape = InputShape();
		const int64_t batch = inShape[0];
		const int64_t slot = inShape[1] * inShape[2] * inShape[3];

		PreprocessConfig cfg;
		cfg.size = static_cast<int>(inShape[3]);
		OpContext ctx;

		auto input = std::make_shared<Tensor>();
		input->dtype = DataType::FLOAT32;
		input->shape = inShape;
		input->storage = std::make_shared<Buffer>(Location::Device, input->numel() * sizeof(float));

		std::vector<std::vector<float>> embeds;
		for (size_t start = 0; start < imagePaths.size(); start += static_cast<size_t>(batch)) {
			const int64_t used = std::min<int64_t>(batch, static_cast<int64_t>(imagePaths.size() - start));
			for (int64_t n = 0; n < used; ++n) {
				std::cout << "Embedding " << imagePaths[start + n] << std::endl;
				PreprocessImage(ReadFileBytes(imagePaths[start + n]), cfg, input->f32() + n * slot, ctx.stream);
			}
			// Fixed batch: pad unused slots with copies of slot 0 and ignore their outputs.
			for (int64_t n = used; n < batch; ++n) {
				CUDA_CHECK(cudaMemcpyAsync(input->f32() + n * slot, input->f32(), slot * sizeof(float),
					cudaMemcpyDeviceToDevice, ctx.stream));
			}
			env[input_names[0]] = input;

			auto outMap = RunGraph(runArg, ctx);
			TensorPtr out = outMap.at(output_names[0]);                // [batch, 512]
			const int64_t dim = out->numel() / batch;
			std::vector<float> host(out->numel());
			CUDA_CHECK(cudaMemcpyAsync(host.data(), out->f32(), host.size() * sizeof(float),
				cudaMemcpyDeviceToHost, ctx.stream));
			CUDA_CHECK(cudaStreamSynchronize(ctx.stream));       // also makes `input` safe to overwrite next chunk

			for (int64_t n = 0; n < used; ++n) {
				embeds.emplace_back(host.begin() + n * dim, host.begin() + (n + 1) * dim);
			}
		}

		//Print result
        // CLIP's learned temperature, logit_scale.exp(). For ViT-B/32 it is 100
        // (export_clip.py stores it as "logit_scale" in manifest.json).
        constexpr double kLogitScale = 100.0;

        // L2-normalise, so a dot product is the cosine similarity.
        for (std::vector<float>& v : embeds) {
            double sumSq = 0.0;
            for (float x : v) sumSq += static_cast<double>(x) * x;
            const double inv = 1.0 / std::sqrt(sumSq);
            for (float& x : v) x = static_cast<float>(x * inv);
        }

        const size_t n = embeds.size();
        std::vector<std::vector<double>> cos(n, std::vector<double>(n));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double dot = 0.0;
                for (size_t k = 0; k < embeds[i].size(); ++k) dot += static_cast<double>(embeds[i][k]) * embeds[j][k];
                cos[i][j] = dot;
            }
        }

        std::vector<std::string> names;
        for (const std::string& p : runArg.imagePaths) names.push_back(std::filesystem::path(p).filename().string());

        std::stringstream ss;
        ss << "\nImages:\n";
        for (size_t i = 0; i < n; ++i) ss << "  [" << i << "] " << names[i] << "\n";

        ss << "\nCosine similarity:\n      ";
        for (size_t j = 0; j < n; ++j) ss << std::setw(8) << ("[" + std::to_string(j) + "]");
        ss << "\n" << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < n; ++i) {
            ss << std::setw(6) << ("[" + std::to_string(i) + "]");
            for (size_t j = 0; j < n; ++j) ss << std::setw(8) << cos[i][j];
            ss << "\n";
        }

        // For each image: softmax over the OTHER images of logit_scale * cosine (same maths as CLIP's test.py).
        ss << "\nMatch probabilities (softmax of " << kLogitScale << " x cosine over the other images):\n"
            << std::setprecision(2);
        for (size_t i = 0; i < n; ++i) {
            double maxLogit = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < n; ++j) if (j != i) maxLogit = std::max(maxLogit, kLogitScale * cos[i][j]);
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) if (j != i) sum += std::exp(kLogitScale * cos[i][j] - maxLogit);

            ss << "  [" << i << "] " << names[i] << " ->";
            for (size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                const double p = std::exp(kLogitScale * cos[i][j] - maxLogit) / sum;
                ss << "  [" << j << "] " << std::setw(6) << 100.0 * p << "%";
            }
            ss << "\n";
        }

        size_t bestI = 0, bestJ = 1;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (cos[i][j] > cos[bestI][bestJ]) { bestI = i; bestJ = j; }
            }
        }
        ss << "\nClosest pair: [" << bestI << "] " << names[bestI] << " <-> [" << bestJ << "] "
            << names[bestJ] << "  (cosine " << std::setprecision(4) << cos[bestI][bestJ] << ")\n";


        logging::Infof("%s", ss.str().c_str());
	}
}