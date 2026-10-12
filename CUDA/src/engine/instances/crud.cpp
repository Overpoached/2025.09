#include "crud.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <set>
#include <stdexcept>
#include <utility>

#include "log.h"
#include "../image_files.h"
#include "../../search/file_storage_proxy.h"

namespace ohy {
	namespace {
		namespace fs = std::filesystem;

		struct LabelledImage {
			std::string path;
			std::string label;
		};

		// Label from the image's directory: dataset/honda_civic_2012/001.jpg -> "honda_civic_2012".
		std::string LabelFromParentDir(const fs::path& file) {
			const std::string label = fs::absolute(file).parent_path().filename().string();
			if (label.empty()) {
				throw std::runtime_error("Cannot deduce a label from the parent directory of " + file.string());
			}
			return label;
		}

		// Expands directories (recursively) into image files and pairs every file with its label:
		//   no labels         -> each file's parent directory name
		//   one label         -> that label for every file
		//   one per input arg -> that arg's label for every file found under it
		std::vector<LabelledImage> CollectImages(const std::vector<std::string>& inputs,
			const std::vector<std::string>& labels) {

			if (!labels.empty() && labels.size() != 1 && labels.size() != inputs.size()) {
				throw std::runtime_error("CRUD: give no label, one label for all inputs, or one label per input (got " +
					std::to_string(labels.size()) + " labels for " + std::to_string(inputs.size()) + " inputs)");
			}

			std::vector<LabelledImage> out;
			for (size_t i = 0; i < inputs.size(); ++i) {
				const std::string* fixedLabel =
					labels.empty() ? nullptr : &labels[labels.size() == 1 ? 0 : i];
				for (auto& file : FindImages(inputs[i])) {
					std::string label = fixedLabel ? *fixedLabel : LabelFromParentDir(file);
					out.push_back({ std::move(file), std::move(label) });
				}
			}
			return out;
		}
	}

	void CRUDInstance::InitSelf(const InitArg& initArg) {
		if (initArg.storageDir.empty()) {
			throw std::runtime_error("CRUD: storage directory not set");
		}
		store = std::make_unique<EmbedStore>(std::make_unique<FileStorageProxy>(initArg.storageDir));
	}

	void CRUDInstance::RunSelf(const RunArg& runArg) {
		switch (runArg.crudOp) {
		case CrudOp::Add:    RunAdd(runArg); break;
		case CrudOp::Update: RunUpdate(runArg); break;
		case CrudOp::Delete: RunDelete(runArg); break;
		case CrudOp::Read:   RunRead(runArg); break;
		default: throw std::runtime_error("CRUD: unknown operation");
		}
	}

	void CRUDInstance::RunAdd(const RunArg& runArg) {
		if (runArg.imagePaths.empty()) {
			throw std::runtime_error("CRUD add: no images");
		}
		if (runArg.runBy.empty()) {
			throw std::runtime_error("CRUD add: runBy (create_by) not set");
		}

		// Expand directories and resolve labels. Fails before any GPU work if an input is missing.
		const std::vector<LabelledImage> images = CollectImages(runArg.imagePaths, runArg.labels);

		// Drop files already stored, and repeats within this request, before spending GPU time on them.
		std::vector<std::string> todoPaths;
		std::vector<std::string> todoLabels;
		std::set<std::string> seen;
		size_t skipped = 0;
		for (const auto& image : images) {
			const std::string canonical = EmbedStore::CanonicalPath(image.path);
			if (!seen.insert(canonical).second || store->Contains(canonical)) {
				logging::Infof("Skipping %s: already stored", canonical);
				++skipped;
				continue;
			}
			todoPaths.push_back(canonical);
			todoLabels.push_back(image.label);
		}

		size_t added = 0;
		size_t failed = 0;
		if (!todoPaths.empty()) {
			const auto embeds = EmbedImages(todoPaths, runArg);

			for (size_t i = 0; i < todoPaths.size(); ++i) {
				const AddStatus status = store->Add(todoPaths[i], todoLabels[i], embeds[i], runArg.runBy);
				switch (status) {
				case AddStatus::Added:         ++added; break;
				case AddStatus::DuplicatePath: ++skipped; break;
				case AddStatus::Failed:        ++failed; break;
				}
				logging::Infof("%s: %s [%s]", ToString(status), todoPaths[i], todoLabels[i]);
			}
		}

		logging::Infof("CRUD add: %zu added, %zu skipped (duplicate), %zu failed, of %zu images",
			added, skipped, failed, images.size());
		if (failed > 0) {
			throw std::runtime_error("CRUD: " + std::to_string(failed) + " image(s) failed to store");
		}
	}

	void CRUDInstance::RunUpdate(const RunArg& runArg) {
		if (runArg.imagePaths.empty()) {
			throw std::runtime_error("CRUD update: no images");
		}
		if (runArg.runBy.empty()) {
			throw std::runtime_error("CRUD update: runBy (create_by) not set");
		}

		const std::vector<LabelledImage> images = CollectImages(runArg.imagePaths, runArg.labels);
		if (images.empty()) {
			throw std::runtime_error("CRUD update: no images found, so no labels to update");
		}

		// The labels this update owns. Only these are touched.
		std::set<std::string> labels;
		for (const auto& image : images) {
			labels.insert(image.label);
		}

		// A file stored under some other label stays there: an update never reaches outside its labels.
		// Repeats within the request are dropped. Files already stored under one of `labels` are
		// re-embedded, since their old rows are deleted as part of the replace.
		std::vector<std::string> todoPaths;
		std::vector<std::string> todoLabels;
		std::set<std::string> seen;
		size_t skipped = 0;
		for (const auto& image : images) {
			const std::string canonical = EmbedStore::CanonicalPath(image.path);
			if (!seen.insert(canonical).second) {
				++skipped;
				continue;
			}
			if (const auto stored = store->StoredLabel(canonical); stored && !labels.contains(*stored)) {
				logging::Warnf("Skipping %s: stored under label '%s', which this update does not cover "
					"(delete that label first to move the file)", canonical, *stored);
				++skipped;
				continue;
			}
			todoPaths.push_back(canonical);
			todoLabels.push_back(image.label);
		}

		const auto before = store->CountByLabel();

		std::vector<NewEmbed> entries;
		if (!todoPaths.empty()) {
			auto embeds = EmbedImages(todoPaths, runArg);
			entries.reserve(todoPaths.size());
			for (size_t i = 0; i < todoPaths.size(); ++i) {
				entries.push_back({ todoPaths[i], todoLabels[i], std::move(embeds[i]) });
			}
		}

		// All-or-nothing: on failure this throws and the store is unchanged.
		const ReplaceResult result = store->ReplaceLabels(labels, std::move(entries), runArg.runBy);

		const auto after = store->CountByLabel();
		std::string report;
		for (const auto& label : labels) {
			const auto b = before.find(label);
			const auto a = after.find(label);
			report += std::format("\n  {}: {} -> {}", label,
				b == before.end() ? 0 : b->second, a == after.end() ? 0 : a->second);
		}
		logging::Infof("CRUD update: %zu labels, %zu embeddings deleted, %zu added, %zu skipped%s",
			labels.size(), result.deleted, result.added, skipped, report);
	}

	void CRUDInstance::RunDelete(const RunArg& runArg) {
		if (runArg.all) {
			if (!runArg.labels.empty()) {
				throw std::runtime_error("CRUD delete: give --label or --all, not both");
			}
			const size_t n = store->DeleteAll();
			logging::Infof("CRUD delete: cleared the store, %zu embeddings deleted", n);
			return;
		}
		if (runArg.labels.empty()) {
			throw std::runtime_error("CRUD delete: give --label <label> or --all");
		}

		size_t total = 0;
		for (const auto& label : runArg.labels) {
			const size_t n = store->DeleteLabel(label);
			if (n == 0) {
				logging::Warnf("CRUD delete: no embeddings under label '%s'", label);
			}
			else {
				logging::Infof("CRUD delete: %zu embeddings deleted under label '%s'", n, label);
			}
			total += n;
		}
		logging::Infof("CRUD delete: %zu embeddings deleted in total", total);
	}

	void CRUDInstance::RunRead(const RunArg& runArg) {
		const auto counts = store->CountByLabel();

		// Every label, or just the requested ones (0 if absent).
		std::vector<std::pair<std::string, size_t>> rows;
		if (runArg.labels.empty()) {
			rows.assign(counts.begin(), counts.end());
		}
		else {
			for (const auto& label : runArg.labels) {
				const auto it = counts.find(label);
				rows.emplace_back(label, it == counts.end() ? 0 : it->second);
			}
		}

		size_t width = 5;   // "label"
		size_t total = 0;
		for (const auto& [label, n] : rows) {
			width = std::max(width, label.size());
			total += n;
		}

		std::string table = std::format("\n{:<{}}  {:>8}\n{}\n", "label", width, "count", std::string(width + 10, '-'));
		for (const auto& [label, n] : rows) {
			table += std::format("{:<{}}  {:>8}\n", label, width, n);
		}
		table += std::format("{}\n{:<{}}  {:>8}  ({} labels)\n", std::string(width + 10, '-'), "total", width, total, rows.size());
		if (runArg.labels.empty() && rows.empty()) {
			table = "\nThe store is empty.\n";
		}
		logging::Infof("%s", table);
	}
}
