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

		std::vector<std::vector<float>> embeds = EmbedImages(runArg.imagePaths, runArg);

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