#include "embed_store.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <stdexcept>

#include "log.h"
#include "knn.h"

namespace ohy {

	namespace {
		int64_t NowUnixMs() {
			using namespace std::chrono;
			return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
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

	std::optional<std::string> EmbedStore::StoredLabel(const std::string& filePath) {
		const auto entry = proxy->FindSaveEntryByPath(CanonicalPath(filePath));
		if (!entry) {
			return std::nullopt;
		}
		const auto embed = proxy->FindEmbedById(entry->embedId);
		return embed ? embed->label : std::string();
	}

	std::map<std::string, size_t> EmbedStore::CountByLabel() {
		return proxy->CountEmbedsByLabel();
	}

	std::vector<EmbedRecord> EmbedStore::LoadAll() {
		return proxy->LoadAllEmbeds();
	}

	AddStatus EmbedStore::Add(const std::string& filePath, const std::string& label,
		std::vector<float> embed, const std::string& createBy) {

		const std::string path = CanonicalPath(filePath);

		// Authoritative duplicate check. Contains() is only an early-out for callers.
		if (proxy->FindSaveEntryByPath(path)) {
			return AddStatus::DuplicatePath;
		}

		try {
			proxy->Begin();
			InsertInTx(path, label, std::move(embed), createBy, NowUnixMs());
			proxy->Commit();
		}
		catch (const std::exception& e) {
			logging::Errorf("Failed to store %s: %s", path, e.what());
			RollbackQuietly();
			return AddStatus::Failed;
		}
		return AddStatus::Added;
	}

	ReplaceResult EmbedStore::ReplaceLabels(const std::set<std::string>& labels, std::vector<NewEmbed> entries,
		const std::string& createBy) {

		for (const auto& e : entries) {
			if (!labels.contains(e.label)) {
				throw std::runtime_error("ReplaceLabels: entry label '" + e.label + "' is not one of the labels being replaced");
			}
		}

		ReplaceResult result;
		const int64_t now = NowUnixMs();
		try {
			proxy->Begin();
			for (const auto& label : labels) {
				result.deleted += DeleteLabelInTx(label);
			}
			// A path still present here is stored under a label outside `labels`; the insert throws.
			for (auto& e : entries) {
				InsertInTx(CanonicalPath(e.filePath), e.label, std::move(e.embed), createBy, now);
				++result.added;
			}
			proxy->Commit();
		}
		catch (...) {
			RollbackQuietly();
			throw;
		}
		return result;
	}

	size_t EmbedStore::DeleteLabel(const std::string& label) {
		return ReplaceLabels({ label }, {}, "").deleted;
	}

	size_t EmbedStore::DeleteAll() {
		size_t count = 0;
		try {
			proxy->Begin();
			for (const auto& [label, n] : proxy->CountEmbedsByLabel()) {
				count += n;
			}
			proxy->Truncate();
			proxy->Commit();
		}
		catch (...) {
			RollbackQuietly();
			throw;
		}
		return count;
	}

	void EmbedStore::InsertInTx(const std::string& canonicalPath, const std::string& label,
		std::vector<float> embed, const std::string& createBy, int64_t now) {

		if (!L2Normalise(embed)) {
			throw std::runtime_error("Embedding for " + canonicalPath + " is zero or non-finite");
		}

		EmbedRecord embedRec;
		embedRec.label = label;
		embedRec.embed = std::move(embed);
		embedRec.createOn = now;
		embedRec.createBy = createBy;
		const int64_t embedId = proxy->InsertEmbed(std::move(embedRec));

		SaveEntryRecord entry;
		entry.filePath = canonicalPath;
		entry.embedId = embedId;
		entry.createOn = now;
		entry.createBy = createBy;
		proxy->InsertSaveEntry(std::move(entry));
	}

	size_t EmbedStore::DeleteLabelInTx(const std::string& label) {
		const std::vector<int64_t> ids = proxy->FindEmbedIdsByLabel(label);
		if (ids.empty()) {
			return 0;
		}
		proxy->DeleteSaveEntriesByEmbedIds(ids);   // save entries first: embeds are ON DELETE RESTRICT
		return proxy->DeleteEmbeds(ids);
	}

	void EmbedStore::RollbackQuietly() {
		try {
			proxy->Rollback();
		}
		catch (...) {
			// Commit() already rolled back, or the transaction never opened.
		}
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
