#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

namespace ohy {

	namespace detail {
		// printf can't take std::string, so pass .c_str() for it. Reject other class types
		// at compile time instead of letting them hit varargs as undefined behaviour.
		template <typename T>
		decltype(auto) PrintfArg(const T& v) {
			if constexpr (std::is_same_v<T, std::string>) {
				return v.c_str();
			}
			else {
				static_assert(std::is_arithmetic_v<T> || std::is_pointer_v<T> || std::is_enum_v<T> || std::is_array_v<T>,
					"Format: pass numbers, const char* or std::string");
				return (v);
			}
		}
	}

	// va_list version, for functions that already take `...` and want to forward it.
	inline std::string VFormat(const char* fmt, va_list args) {
		va_list copy;
		va_copy(copy, args);                                  // the first vsnprintf consumes `args`
		const int n = std::vsnprintf(nullptr, 0, fmt, copy);  // measure
		va_end(copy);
		if (n < 0) { return std::string("<bad format> ") + fmt; }

		std::string out(static_cast<size_t>(n), '\0');
		std::vsnprintf(out.data(), out.size() + 1, fmt, args); // +1: writes the '\0' std::string already reserves
		return out;
	}

	// C-style format into a std::string. std::string arguments work with %s.
	//   std::string s = Format("%s: %d nodes, %.2f ms", name, count, ms);
	template <typename... Args>
	std::string Format(const char* fmt, const Args&... args) {
		const int n = std::snprintf(nullptr, 0, fmt, detail::PrintfArg(args)...);
		if (n < 0) { return std::string("<bad format> ") + fmt; }

		std::string out(static_cast<size_t>(n), '\0');
		std::snprintf(out.data(), out.size() + 1, fmt, detail::PrintfArg(args)...);
		return out;
	}

	template <typename T>
	std::string ToString(const std::vector<T>& vec, uint32_t pad = 0) {
		std::string ret = "[";
		for (auto& v : vec) {
			if (ret.size() > 1) { ret += ", "; }
			ret += std::to_string(v);
		}
		return ret + "]";
	}
}