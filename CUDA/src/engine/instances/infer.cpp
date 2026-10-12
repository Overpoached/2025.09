#include "infer.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <stdexcept>

#include "log.h"
#include "../image_files.h"
#include "../../search/embed_store.h"
#include "../../search/file_storage_proxy.h"

namespace ohy {
	namespace {
		// Cosine distance: 0 = same direction, 1 = unrelated, 2 = opposite.
		float Distance(float similarity) { return 1.0f - similarity; }
	}

	void InferInstance::InitSelf(const InitArg& initArg) {
		if (initArg.storageDir.empty()) {
			throw std::runtime_error("INFER: storage directory not set");
		}
		// Opening a missing directory would create an empty store; a typo should fail instead.
		if (!std::filesystem::is_directory(initArg.storageDir)) {
			throw std::runtime_error("INFER: storage directory not found: " +
				std::filesystem::absolute(initArg.storageDir).string());
		}

		std::vector<EmbedRecord> records;
		{
			EmbedStore store(std::make_unique<FileStorageProxy>(initArg.storageDir));
			records = store.LoadAll();
		}
		if (records.empty()) {
			throw std::runtime_error("INFER: no embeddings stored in " + initArg.storageDir + " (run 'add' first)");
		}

		index = std::make_unique<BruteForceIndex>(records);
		logging::Infof("Loaded %zu embeddings across %zu labels (dim %zu) onto the GPU",
			index->Size(), index->LabelCount(), index->Dim());
	}

	void InferInstance::RunSelf(const RunArg& runArg) {
		if (runArg.imagePaths.empty()) {
			throw std::runtime_error("INFER: no images");
		}
		if (runArg.topK <= 0) {
			throw std::runtime_error("INFER: k must be positive");
		}

		std::vector<std::string> files;
		for (const auto& input : runArg.imagePaths) {
			auto found = FindImages(input);
			files.insert(files.end(), std::make_move_iterator(found.begin()), std::make_move_iterator(found.end()));
		}
		if (files.empty()) {
			throw std::runtime_error("INFER: no images found");
		}

		auto queries = EmbedImages(files, runArg);
		for (size_t i = 0; i < queries.size(); ++i) {
			if (!L2Normalise(queries[i])) {
				logging::Warnf("Embedding for %s is zero or non-finite; its matches are meaningless", files[i]);
			}
		}

		const size_t k = std::min(static_cast<size_t>(runArg.topK), index->Size());
		const auto results = index->Search(queries, k);

		std::string detail;
		std::string summary = std::format("\n{:<4} {:<32} {:<24} {:>7} {:>9}\n",
			"#", "image", "best match", "share", "nearest");
		summary += std::string(80, '-') + "\n";

		for (size_t i = 0; i < files.size(); ++i) {
			const auto& neighbours = results[i];
			const auto votes = VoteLabels(neighbours);
			const std::string name = std::filesystem::path(files[i]).filename().string();

			detail += std::format("\n[{}] {}\n", i, files[i]);
			if (votes.empty()) {
				detail += "    no neighbours\n";
				summary += std::format("{:<4} {:<32} {:<24}\n", i, name, "-");
				continue;
			}

			const LabelScore& best = votes.front();
			detail += std::format("    best match : {}  ({} of {} neighbours, vote share {:.1f}%, nearest distance {:.4f} -> id {})\n",
				best.label, best.votes, neighbours.size(), 100.0f * best.share, Distance(best.bestSimilarity), best.bestId);

			detail += "    labels     :";
			for (const auto& v : votes) {
				detail += std::format("  {} {:.1f}% ({}, {:.4f})", v.label, 100.0f * v.share, v.votes, Distance(v.bestSimilarity));
			}
			detail += "\n    neighbours :";
			for (const auto& nb : neighbours) {
				detail += std::format("  #{} {} {:.4f}", nb.id, nb.label, Distance(nb.similarity));
			}
			detail += "\n";

			summary += std::format("{:<4} {:<32} {:<24} {:>6.1f}% {:>9.4f}\n",
				i, name, best.label, 100.0f * best.share, Distance(best.bestSimilarity));
		}

		logging::Infof("Matches (k = %zu; distance = 1 - cosine similarity; labels: share (votes, nearest distance)):%s%s",
			k, detail, summary);
	}
}
