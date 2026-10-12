#include "log.h"

#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace ohy::logging {

	namespace detail {
		std::atomic<int> gLevel{ static_cast<int>(Level::Info) };
	}

	namespace {
		constexpr std::string_view kLevelNames[] = { "TRACE", "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL" };

		// __FILE__ on MSVC is the full path. Keep only the file name.
		std::string_view FileName(std::string_view path) {
			const size_t slash = path.find_last_of("/\\");
			return slash == std::string_view::npos ? path : path.substr(slash + 1);
		}

		// Readable thread ids (1, 2, 3...) instead of the OS's large numbers.
		uint32_t ThreadNumber() {
			static std::atomic<uint32_t> next{ 0 };
			thread_local const uint32_t mine = ++next;
			return mine;
		}

		class ConsoleSink final : public Sink {
		public:
			void Write(const Record& r) override {
				std::cout << FormatRecord(r) << '\n';
				if (r.level >= Level::Error) { std::cout.flush(); }
			}
			void Flush() override { std::cout.flush(); }
		};

		class FileSink final : public Sink {
		public:
			FileSink(const std::string& path, bool append)
				: out_(path, append ? std::ios::app : std::ios::trunc) {
				if (!out_) { throw std::runtime_error("log: cannot open " + path); }
			}
			void Write(const Record& r) override {
				out_ << FormatRecord(r) << '\n';
				if (r.level >= Level::Warn) { out_.flush(); }   // warnings and errors survive a crash
			}
			void Flush() override { out_.flush(); }

		private:
			std::ofstream out_;
		};

		class AsyncSink final : public Sink {
		public:
			explicit AsyncSink(std::vector<std::shared_ptr<Sink>> inner)
				: inner_(std::move(inner)), worker_([this] { Run(); }) {
			}

			~AsyncSink() override {
				{ std::lock_guard lock(mutex_); stopping_ = true; }
				wake_.notify_one();
				worker_.join();                       // Run() drains the queue before returning
			}

			void Write(const Record& r) override {
				{ std::lock_guard lock(mutex_); queue_.push_back(r); }
				wake_.notify_one();
			}

			// Blocks until everything queued so far is written. The logger's mutex is held
			// by our caller, so no new Write can arrive while we wait.
			void Flush() override {
				std::unique_lock lock(mutex_);
				idle_.wait(lock, [this] { return queue_.empty() && !writing_; });
				for (auto& s : inner_) { s->Flush(); }
			}

		private:
			void Run() {
				std::vector<Record> batch;
				std::unique_lock lock(mutex_);
				while (true) {
					wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
					if (queue_.empty()) { break; }    // stopping, and nothing left
					batch.swap(queue_);                // take everything in one go
					writing_ = true;
					lock.unlock();                     // callers keep enqueuing while we do I/O
					for (const Record& r : batch) {
						for (auto& s : inner_) { s->Write(r); }
					}
					batch.clear();
					lock.lock();
					writing_ = false;
					idle_.notify_all();
				}
				for (auto& s : inner_) { s->Flush(); }
			}

			std::vector<std::shared_ptr<Sink>> inner_;
			std::mutex mutex_;
			std::condition_variable wake_;
			std::condition_variable idle_;
			std::vector<Record> queue_;
			bool writing_ = false;
			bool stopping_ = false;
			std::thread worker_;                       // last: starts after everything above exists
		};

		struct State {
			std::mutex mutex;                          // serialises sink calls, so lines never interleave
			std::vector<std::shared_ptr<Sink>> sinks{ std::make_shared<ConsoleSink>() };   // works before Init
		};

		State& GetState() {
			static State state;
			return state;
		}
	}

	std::string FormatRecord(const Record& r) {
		const auto t = std::chrono::floor<std::chrono::milliseconds>(r.time);
		return std::format("{:%Y-%m-%d %H:%M:%S}Z {} [{}] {}:{}  {}",
			t, kLevelNames[static_cast<int>(r.level)], r.thread, FileName(r.file), r.line, r.message);
	}

	std::shared_ptr<Sink> MakeConsoleSink() { return std::make_shared<ConsoleSink>(); }
	std::shared_ptr<Sink> MakeFileSink(const std::string& path, bool append) { return std::make_shared<FileSink>(path, append); }
	std::shared_ptr<Sink> MakeAsyncSink(std::vector<std::shared_ptr<Sink>> inner) { return std::make_shared<AsyncSink>(std::move(inner)); }

	void Init(const Config& cfg) {
		std::vector<std::shared_ptr<Sink>> sinks;
		if (cfg.console) { sinks.push_back(MakeConsoleSink()); }
		if (!cfg.file.empty()) { sinks.push_back(MakeFileSink(cfg.file, cfg.append)); }
		if (cfg.async && !sinks.empty()) { sinks = { MakeAsyncSink(std::move(sinks)) }; }
		SetSinks(std::move(sinks));
		SetLevel(cfg.level);
	}

	void SetSinks(std::vector<std::shared_ptr<Sink>> sinks) {
		State& s = GetState();
		std::vector<std::shared_ptr<Sink>> old;
		{
			std::lock_guard lock(s.mutex);
			for (auto& sink : s.sinks) { sink->Flush(); }
			old.swap(s.sinks);
			s.sinks = std::move(sinks);
		}
		// `old` is destroyed here, outside the lock. An AsyncSink joins its thread in its destructor.
	}

	void SetLevel(Level level) { detail::gLevel.store(static_cast<int>(level), std::memory_order_relaxed); }

	void Flush() {
		State& s = GetState();
		std::lock_guard lock(s.mutex);
		for (auto& sink : s.sinks) { sink->Flush(); }
	}

	void Shutdown() { SetSinks({}); }

	namespace detail {
		void Submit(Level level, const std::source_location& loc, std::string message) {
			Record r{ level, std::chrono::system_clock::now(), ThreadNumber(), loc.file_name(), loc.line(), std::move(message) };
			State& s = GetState();
			std::lock_guard lock(s.mutex);
			for (auto& sink : s.sinks) { sink->Write(r); }
		}
	}
}