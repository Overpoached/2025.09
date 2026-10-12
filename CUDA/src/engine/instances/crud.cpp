#include "crud.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <stdexcept>

#include "log.h"
#include "../../search/file_storage_proxy.h"

namespace ohy {
	namespace {
		namespace fs = std::filesystem;

		struct LabelledImage {
			std::string path;
			std::string label;
		};

		// What the OpenCV build can decode (vcpkg features "jpeg", "png"; BMP is built in).
		bool IsSupportedImage(const fs::path& p) {
			std::string ext = p.extension().string();
			std::transform(ext.begin(), ext.end(), ext.begin(),
				[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return ext == ".jpg" || ext == ".jpeg" || ext == ".jpe" || ext == ".png" || ext == ".bmp";
		}

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
		// A file named explicitly is used as is, whatever its extension; the decoder will reject it if needed.
		std::vector<LabelledImage> CollectImages(const std::vector<std::string>& inputs,
			const std::vector<std::string>& labels) {

			if (!labels.empty() && labels.size() != 1 && labels.size() != inputs.size()) {
				throw std::runtime_error("CRUD: give no label, one label for all inputs, or one label per input (got " +
					std::to_string(labels.size()) + " labels for " + std::to_string(inputs.size()) + " inputs)");
			}

			std::vector<LabelledImage> out;
			for (size_t i = 0; i < inputs.size(); ++i) {
				const fs::path input(inputs[i]);
				const std::string* fixedLabel =
					labels.empty() ? nullptr : &labels[labels.size() == 1 ? 0 : i];
				const auto labelFor = [&](const fs::path& file) {
					return fixedLabel ? *fixedLabel : LabelFromParentDir(file);
				};

				if (fs::is_directory(input)) {
					std::vector<fs::path> files;
					for (const auto& entry : fs::recursive_directory_iterator(
						input, fs::directory_options::skip_permission_denied)) {
						if (entry.is_regular_file() && IsSupportedImage(entry.path())) {
							files.push_back(entry.path());
						}
					}
					// Directory iteration order is unspecified; sort so ids are assigned reproducibly.
					std::sort(files.begin(), files.end());
					if (files.empty()) {
						logging::Warnf("No images found under %s", fs::absolute(input).string());
					}
					else {
						logging::Infof("Found %zu images under %s", files.size(), fs::absolute(input).string());
					}
					for (const auto& f : files) {
						out.push_back({ f.string(), labelFor(f) });
					}
				}
				else if (fs::is_regular_file(input)) {
					out.push_back({ input.string(), labelFor(input) });
				}
				else {
					throw std::runtime_error("Image file or directory not found: " + fs::absolute(input).string());
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
		if (runArg.imagePaths.empty()) {
			throw std::runtime_error("CRUD: no images");
		}
		if (runArg.runBy.empty()) {
			throw std::runtime_error("CRUD: runBy (create_by) not set");
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
}
