#include "image_files.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "log.h"

namespace ohy {
	namespace fs = std::filesystem;

	bool IsSupportedImage(const fs::path& p) {
		std::string ext = p.extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(),
			[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return ext == ".jpg" || ext == ".jpeg" || ext == ".jpe" || ext == ".png" || ext == ".bmp";
	}

	std::vector<std::string> FindImages(const std::string& input) {
		const fs::path path(input);
		if (fs::is_regular_file(path)) {
			return { path.string() };
		}
		if (!fs::is_directory(path)) {
			throw std::runtime_error("Image file or directory not found: " + fs::absolute(path).string());
		}

		std::vector<fs::path> files;
		for (const auto& entry : fs::recursive_directory_iterator(path, fs::directory_options::skip_permission_denied)) {
			if (entry.is_regular_file() && IsSupportedImage(entry.path())) {
				files.push_back(entry.path());
			}
		}
		// Directory iteration order is unspecified; sort so ids and output order are reproducible.
		std::sort(files.begin(), files.end());

		if (files.empty()) {
			logging::Warnf("No images found under %s", fs::absolute(path).string());
		}
		else {
			logging::Infof("Found %zu images under %s", files.size(), fs::absolute(path).string());
		}

		std::vector<std::string> out;
		out.reserve(files.size());
		for (const auto& f : files) {
			out.push_back(f.string());
		}
		return out;
	}
}
