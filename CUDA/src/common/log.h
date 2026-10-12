#pragma once

// Logging with std::format or printf-style formatting. File and line are captured
// automatically through std::source_location, so no macros are needed.
//
//   ohy::logging::Init({ .level = Level::Debug, .file = "cuda.log", .async = true });
//   logging::Info("loaded {} nodes from {}", n, path);      // std::format, checked at compile time
//   logging::Warnf("%.2f GB/s (%d%% of peak)", gbps, pct);  // printf style, NOT checked
//
// Do not include this from .cu files: nvcc compiles those as C++17, and this needs C++20.

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <format>
#include <memory>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ohy::logging {

	enum class Level : int { Trace, Debug, Info, Warn, Error, Fatal, Off };

	struct Record {
		Level level;
		std::chrono::system_clock::time_point time;
		uint32_t thread;          // small per-process number, 1 = first thread that logged
		const char* file;         // from std::source_location: static storage, always valid
		uint32_t line;
		std::string message;
	};

	// "2026-10-12 08:31:05.123Z INFO  [1] engine.cpp:212  message"
	std::string FormatRecord(const Record& r);

	// Where records go. The logger calls Write/Flush while holding its own mutex,
	// so a sink never sees two calls at once and does no locking of its own.
	class Sink {
	public:
		virtual ~Sink() = default;
		virtual void Write(const Record& r) = 0;
		virtual void Flush() {}
	};

	std::shared_ptr<Sink> MakeConsoleSink();
	std::shared_ptr<Sink> MakeFileSink(const std::string& path, bool append = true);
	// Callers only enqueue. One background thread writes to all `inner` sinks.
	std::shared_ptr<Sink> MakeAsyncSink(std::vector<std::shared_ptr<Sink>> inner);

	struct Config {
		Level level = Level::Info;
		bool console = true;
		std::string file;      // empty = no file
		bool append = true;
		bool async = false;    // true = console and file are written by a background thread
	};

	void Init(const Config& cfg);                               // replaces the current sinks
	void SetSinks(std::vector<std::shared_ptr<Sink>> sinks);    // custom setups
	void SetLevel(Level level);
	void Flush();
	void Shutdown();     // flush and stop the async thread. Call before main returns.

	namespace detail {
		extern std::atomic<int> gLevel;
		void Submit(Level level, const std::source_location& loc, std::string message);

		// printf can't take std::string, so pass .c_str() for it. Reject other class types
		// at compile time instead of letting them hit varargs as undefined behaviour.
		template <typename T>
		decltype(auto) PrintfArg(const T& v) {
			if constexpr (std::is_same_v<T, std::string>) {
				return v.c_str();
			}
			else {
				static_assert(std::is_arithmetic_v<T> || std::is_pointer_v<T> || std::is_enum_v<T> || std::is_array_v<T>,
					"printf-style log: pass numbers, const char* or std::string. Use the {} style for anything else.");
				return (v);
			}
		}

		template <typename... Args>
		std::string Sprintf(const char* fmt, const Args&... args) {
			const int n = std::snprintf(nullptr, 0, fmt, PrintfArg(args)...);
			if (n < 0) { return std::string("<bad format> ") + fmt; }
			std::string out(static_cast<size_t>(n), '\0');
			std::snprintf(out.data(), out.size() + 1, fmt, PrintfArg(args)...);
			return out;
		}
	}

	// std::format string plus the caller's location. The consteval constructor makes
	// the compiler check the format string against the argument types.
	template <typename... Args>
	struct FormatLoc {
		std::format_string<Args...> fmt;
		std::source_location loc;

		template <typename S> requires std::convertible_to<const S&, std::string_view>
		consteval FormatLoc(const S& s, std::source_location l = std::source_location::current())
			: fmt(s), loc(l) {
		}
	};

	// type_identity_t stops Args being deduced from the format string. They come from the arguments only.
	template <typename... Args>
	using Fmt = FormatLoc<std::type_identity_t<Args>...>;

	// printf format string plus the caller's location.
	struct PrintfLoc {
		const char* fmt;
		std::source_location loc;
		PrintfLoc(const char* s, std::source_location l = std::source_location::current()) : fmt(s), loc(l) {}
	};

	inline bool Enabled(Level level) {
		return level != Level::Off && static_cast<int>(level) >= detail::gLevel.load(std::memory_order_relaxed);
	}

	template <typename... Args>
	void Log(Level level, Fmt<Args...> f, Args&&... args) {
		if (Enabled(level)) { detail::Submit(level, f.loc, std::format(f.fmt, std::forward<Args>(args)...)); }
	}

	template <typename... Args>
	void Logf(Level level, PrintfLoc f, const Args&... args) {
		if (Enabled(level)) { detail::Submit(level, f.loc, detail::Sprintf(f.fmt, args...)); }
	}

	// std::format style
	template <typename... Args> void Trace(Fmt<Args...> f, Args&&... a) { Log<Args...>(Level::Trace, f, std::forward<Args>(a)...); }
	template <typename... Args> void Debug(Fmt<Args...> f, Args&&... a) { Log<Args...>(Level::Debug, f, std::forward<Args>(a)...); }
	template <typename... Args> void Info(Fmt<Args...> f, Args&&... a) { Log<Args...>(Level::Info, f, std::forward<Args>(a)...); }
	template <typename... Args> void Warn(Fmt<Args...> f, Args&&... a) { Log<Args...>(Level::Warn, f, std::forward<Args>(a)...); }
	template <typename... Args> void Error(Fmt<Args...> f, Args&&... a) { Log<Args...>(Level::Error, f, std::forward<Args>(a)...); }
	// Logs and flushes. Does not abort: throw or return as you normally would.
	template <typename... Args> void Fatal(Fmt<Args...> f, Args&&... a) { Log<Args...>(Level::Fatal, f, std::forward<Args>(a)...); Flush(); }

	// printf style
	template <typename... Args> void Tracef(PrintfLoc f, const Args&... a) { Logf(Level::Trace, f, a...); }
	template <typename... Args> void Debugf(PrintfLoc f, const Args&... a) { Logf(Level::Debug, f, a...); }
	template <typename... Args> void Infof(PrintfLoc f, const Args&... a) { Logf(Level::Info, f, a...); }
	template <typename... Args> void Warnf(PrintfLoc f, const Args&... a) { Logf(Level::Warn, f, a...); }
	template <typename... Args> void Errorf(PrintfLoc f, const Args&... a) { Logf(Level::Error, f, a...); }
	template <typename... Args> void Fatalf(PrintfLoc f, const Args&... a) { Logf(Level::Fatal, f, a...); Flush(); }
}