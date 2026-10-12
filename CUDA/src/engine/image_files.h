#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace ohy {

	// Extensions the image decoder can read (OpenCV built with vcpkg features "jpeg", "png"; BMP is built in).
	bool IsSupportedImage(const std::filesystem::path& p);

	// A file -> that file, whatever its extension (the decoder rejects it if it must).
	// A directory -> every supported image under it, recursively, sorted so the order is reproducible.
	// Anything else throws.
	std::vector<std::string> FindImages(const std::string& input);
}
