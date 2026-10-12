#include "embed_store.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "log.h"

namespace ohy {

	namespace {
		int64_t NowUnixMs() {
			using namespace std::chrono;
			return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
		}

		// Returns false for a zero (or non-finite) vector, which has no direction to compare.
		bool L2Normalise(std::vector<float>& v) {
			double sumSq = 0.0;
			for (float x : v) sumSq += static_cast<double>(x) * x;
			if (!(sumSq > 0.0) || !std::isfinite(sumSq)) {
				return false;
			}
			const double inv = 1.0 / std::sqrt(sumSq);
			for (float& x : v) x = static_cast<float>(x * inv);
			return true;
		}
	}

	const char* ToString(AddStatus status) {
		switch (status) {
		case AddStatus::Added: return "Added";
		case AddStatus::DuplicatePath: return "DuplicatePath";
		case AddStatus::Failed: return "Failed";
		}
		return "Unknown";
	}

	EmbedStore::EmbedStore(std::unique_ptr<StorageProxy> proxy)
		: proxy(std::move(proxy)) {
		if (!this->proxy) {
			throw std::runtime_error("EmbedStore: null proxy");
		}
		this->proxy->Open();
	}

	EmbedStore::~EmbedStore() {
		try {
			proxy->Close();
		}
		catch (...) {
		}
	}

	bool EmbedStore::Contains(const std::string& filePath) {
		return proxy->FindSaveEntryByPath(CanonicalPath(filePath)).has_value();
	}

	AddStatus EmbedStore::Add(const std::string& filePath, const std::string& label,
		std::vector<float> embed, const std::string& createBy) {

		const std::string path = CanonicalPath(filePath);

		// Authoritative duplicate check. Contains() is only an early-out for callers.
		if (proxy->FindSaveEntryByPath(path)) {
			return AddStatus::DuplicatePath;
		}
		if (!L2Normalise(embed)) {
			logging::Errorf("Embedding for %s is zero or non-finite; not stored", path);
			return AddStatus::Failed;
		}

		const int64_t now = NowUnixMs();
		try {
			proxy->Begin();

			EmbedRecord embedRec;
			embedRec.label = label;
			embedRec.embed = std::move(embed);
			embedRec.createOn = now;
			embedRec.createBy = createBy;
			const int64_t embedId = proxy->InsertEmbed(std::move(embedRec));

			SaveEntryRecord entry;
			entry.filePath = path;
			entry.embedId = embedId;
			entry.createOn = now;
			entry.createBy = createBy;
			proxy->InsertSaveEntry(std::move(entry));

			proxy->Commit();
		}
		catch (const std::exception& e) {
			logging::Errorf("Failed to store %s: %s", path, e.what());
			try {
				proxy->Rollback();
			}
			catch (...) {
				// Commit() already rolled back, or the transaction never opened.
			}
			return AddStatus::Failed;
		}
		return AddStatus::Added;
	}

	std::vector<EmbedRecord> EmbedStore::LoadAll() {
		return proxy->LoadAllEmbeds();
	}

	std::string EmbedStore::CanonicalPath(const std::string& p) {
		// weakly_canonical resolves symlinks and works for paths that do not exist yet.
		std::string out = std::filesystem::weakly_canonical(std::filesystem::absolute(p)).generic_string();
#ifdef _WIN32
		std::transform(out.begin(), out.end(), out.begin(),
			[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
		return out;
	}
}
