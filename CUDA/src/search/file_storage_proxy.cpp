#include "file_storage_proxy.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include "log.h"

namespace ohy {

	namespace {
		// main_data.bin layout:
		//   header:  char magic[4] = "OEMB", u32 version
		//   row:     i64 id, i64 createOn,
		//            u32 labelLen, char label[labelLen],
		//            u32 createByLen, char createBy[createByLen],
		//            u32 dim, f32 embed[dim]
		constexpr char kMagic[4] = { 'O', 'E', 'M', 'B' };
		constexpr uint32_t kVersion = 1;

		// Upper bounds used to catch a corrupt length field before allocating from it.
		constexpr uint32_t kMaxStringLen = 1u << 16;
		constexpr uint32_t kMaxDim = 1u << 16;

		const char* kEntryFile = "save_entry.tsv";
		const char* kEmbedFile = "main_data.bin";
		const char* kEntryHeader = "id\tfile_path\tembed_id\tcreate_on\tcreate_by";

		template <typename T>
		void WritePod(std::ostream& out, const T& v) {
			out.write(reinterpret_cast<const char*>(&v), sizeof(T));
		}

		void WriteString(std::ostream& out, const std::string& s) {
			WritePod(out, static_cast<uint32_t>(s.size()));
			out.write(s.data(), static_cast<std::streamsize>(s.size()));
		}

		template <typename T>
		void ReadPod(std::istream& in, T& v, const std::filesystem::path& path) {
			if (!in.read(reinterpret_cast<char*>(&v), sizeof(T))) {
				throw std::runtime_error("Truncated record in " + path.string());
			}
		}

		std::string ReadString(std::istream& in, const std::filesystem::path& path) {
			uint32_t len = 0;
			ReadPod(in, len, path);
			if (len > kMaxStringLen) {
				throw std::runtime_error("Corrupt string length in " + path.string());
			}
			std::string s(len, '\0');
			if (len > 0 && !in.read(s.data(), len)) {
				throw std::runtime_error("Truncated record in " + path.string());
			}
			return s;
		}

		void WriteEmbedHeader(std::ostream& out) {
			out.write(kMagic, sizeof(kMagic));
			WritePod(out, kVersion);
		}

		void WriteEmbedRow(std::ostream& out, const EmbedRecord& r) {
			WritePod(out, r.id);
			WritePod(out, r.createOn);
			WriteString(out, r.label);
			WriteString(out, r.createBy);
			WritePod(out, static_cast<uint32_t>(r.embed.size()));
			out.write(reinterpret_cast<const char*>(r.embed.data()),
				static_cast<std::streamsize>(r.embed.size() * sizeof(float)));
		}

		void WriteEntryRow(std::ostream& out, const SaveEntryRecord& r) {
			out << r.id << '\t' << r.filePath << '\t' << r.embedId << '\t'
				<< r.createOn << '\t' << r.createBy << '\n';
		}

		// TSV has no escaping, so fields must not contain the separators.
		void RequireTsvSafe(const std::string& s, const char* field) {
			if (s.find_first_of("\t\r\n") != std::string::npos) {
				throw std::runtime_error(std::string(field) + " contains a tab or newline: " + s);
			}
		}

		std::vector<std::string> SplitTabs(const std::string& line) {
			std::vector<std::string> fields;
			std::string field;
			std::istringstream ss(line);
			while (std::getline(ss, field, '\t')) {
				fields.push_back(field);
			}
			if (!line.empty() && line.back() == '\t') {
				fields.emplace_back();   // trailing empty field (e.g. empty create_by)
			}
			return fields;
		}

		uintmax_t FileSizeOrZero(const std::filesystem::path& p) {
			std::error_code ec;
			const uintmax_t size = std::filesystem::file_size(p, ec);
			return ec ? 0 : size;
		}

		std::filesystem::path TempPath(const std::filesystem::path& p) {
			std::filesystem::path tmp = p;
			tmp += ".tmp";
			return tmp;
		}
	}

	template <typename F>
	auto FileStorageProxy::AutoTx(F&& f) {
		RequireOpen();
		if (inTx) {
			return f();
		}
		Begin();
		try {
			auto result = f();
			Commit();
			return result;
		}
		catch (...) {
			if (inTx) Rollback();
			throw;
		}
	}

	FileStorageProxy::FileStorageProxy(std::filesystem::path dir)
		: dir(std::move(dir)) {
		entryPath = this->dir / kEntryFile;
		embedPath = this->dir / kEmbedFile;
	}

	FileStorageProxy::~FileStorageProxy() {
		try {
			Close();
		}
		catch (...) {
			// Destructors must not throw. Close() only discards in-memory state.
		}
	}

	void FileStorageProxy::Open() {
		if (isOpen) {
			return;
		}
		std::filesystem::create_directories(dir);

		try {
			LoadEmbeds();
			LoadSaveEntries();
		}
		catch (...) {
			entriesByPath.clear();
			embeds.clear();
			throw;
		}

		// Cross-table consistency. Not fatal: see the crash windows in the header.
		std::set<int64_t> embedIds;
		for (const auto& e : embeds) {
			embedIds.insert(e.id);
		}
		std::set<int64_t> referenced;
		for (const auto& [path, entry] : entriesByPath) {
			referenced.insert(entry.embedId);
			if (!embedIds.contains(entry.embedId)) {
				logging::Warnf("Save entry %lld (%s) points at missing embed id %lld",
					static_cast<long long>(entry.id), path, static_cast<long long>(entry.embedId));
			}
		}
		for (const auto& e : embeds) {
			if (!referenced.contains(e.id)) {
				logging::Warnf("Embed id %lld (%s) has no save entry", static_cast<long long>(e.id), e.label);
			}
		}

		isOpen = true;
		logging::Infof("Opened storage %s: %zu save entries, %zu embeddings (dim %zu)",
			dir.string(), entriesByPath.size(), embeds.size(), EmbedDim());
	}

	void FileStorageProxy::Close() {
		if (!isOpen) {
			return;
		}
		if (inTx) {
			logging::Warn("Closing storage with an open transaction; rolling back");
			Rollback();
		}
		entriesByPath.clear();
		embeds.clear();
		nextEntryId = 1;
		nextEmbedId = 1;
		isOpen = false;
	}

	void FileStorageProxy::Begin() {
		RequireOpen();
		if (inTx) {
			throw std::runtime_error("Begin: transaction already open");
		}
		inTx = true;
		txEntryId = nextEntryId;
		txEmbedId = nextEmbedId;
	}

	void FileStorageProxy::Commit() {
		RequireOpen();
		if (!inTx) {
			throw std::runtime_error("Commit: no open transaction");
		}
		// Both paths roll back and rethrow on failure.
		if (deletedPaths.empty() && deletedEmbedIds.empty()) {
			CommitAppend();
		}
		else {
			CommitRewrite();
		}

		pendingEmbeds.clear();
		pendingEntries.clear();
		deletedPaths.clear();
		deletedEmbedIds.clear();
		inTx = false;
	}

	void FileStorageProxy::CommitAppend() {
		// Remember the file sizes so a failed append can be cut back off.
		const uintmax_t embedSize = FileSizeOrZero(embedPath);
		const uintmax_t entrySize = FileSizeOrZero(entryPath);

		try {
			// Embeds first: a crash after this block leaves an orphan embed, never a
			// save entry pointing at nothing.
			if (!pendingEmbeds.empty()) {
				std::ofstream f(embedPath, std::ios::binary | std::ios::app);
				for (const auto& r : pendingEmbeds) {
					WriteEmbedRow(f, r);
				}
				f.flush();
				if (!f) {
					throw std::runtime_error("Failed to write " + embedPath.string());
				}
			}
			if (!pendingEntries.empty()) {
				std::ofstream f(entryPath, std::ios::binary | std::ios::app);
				for (const auto& r : pendingEntries) {
					WriteEntryRow(f, r);
				}
				f.flush();
				if (!f) {
					throw std::runtime_error("Failed to write " + entryPath.string());
				}
			}
		}
		catch (...) {
			std::error_code ec;
			std::filesystem::resize_file(embedPath, embedSize, ec);
			std::filesystem::resize_file(entryPath, entrySize, ec);
			Rollback();
			throw;
		}

		for (auto& r : pendingEmbeds) {
			embeds.push_back(std::move(r));
		}
		for (auto& r : pendingEntries) {
			std::string key = r.filePath;
			entriesByPath.emplace(std::move(key), std::move(r));
		}
	}

	void FileStorageProxy::CommitRewrite() {
		const std::filesystem::path embedTmp = TempPath(embedPath);
		const std::filesystem::path entryTmp = TempPath(entryPath);

		try {
			{
				std::ofstream f(embedTmp, std::ios::binary | std::ios::trunc);
				WriteEmbedHeader(f);
				for (const auto& r : embeds) {
					if (EmbedVisible(r.id)) WriteEmbedRow(f, r);
				}
				for (const auto& r : pendingEmbeds) {
					WriteEmbedRow(f, r);
				}
				f.flush();
				if (!f) {
					throw std::runtime_error("Failed to write " + embedTmp.string());
				}
			}
			{
				std::vector<const SaveEntryRecord*> rows;
				for (const auto& [path, r] : entriesByPath) {
					if (EntryVisible(path)) rows.push_back(&r);
				}
				for (const auto& r : pendingEntries) {
					rows.push_back(&r);
				}
				std::sort(rows.begin(), rows.end(),
					[](const SaveEntryRecord* a, const SaveEntryRecord* b) { return a->id < b->id; });

				std::ofstream f(entryTmp, std::ios::binary | std::ios::trunc);
				f << kEntryHeader << '\n';
				for (const auto* r : rows) {
					WriteEntryRow(f, *r);
				}
				f.flush();
				if (!f) {
					throw std::runtime_error("Failed to write " + entryTmp.string());
				}
			}

			// Save entries first: if the second rename never happens, the duplicate table already
			// reflects the delete and the leftover embeddings show up as orphans on the next Open().
			std::filesystem::rename(entryTmp, entryPath);
			std::filesystem::rename(embedTmp, embedPath);
		}
		catch (...) {
			std::error_code ec;
			std::filesystem::remove(embedTmp, ec);
			std::filesystem::remove(entryTmp, ec);
			Rollback();
			throw;
		}

		std::erase_if(embeds, [&](const EmbedRecord& r) { return !EmbedVisible(r.id); });
		for (auto& r : pendingEmbeds) {
			embeds.push_back(std::move(r));   // pending ids are all larger, so order is kept
		}
		std::erase_if(entriesByPath, [&](const auto& kv) { return !EntryVisible(kv.first); });
		for (auto& r : pendingEntries) {
			std::string key = r.filePath;
			entriesByPath.emplace(std::move(key), std::move(r));
		}
	}

	void FileStorageProxy::Rollback() {
		RequireOpen();
		if (!inTx) {
			throw std::runtime_error("Rollback: no open transaction");
		}
		pendingEmbeds.clear();
		pendingEntries.clear();
		deletedPaths.clear();
		deletedEmbedIds.clear();
		nextEntryId = txEntryId;
		nextEmbedId = txEmbedId;
		inTx = false;
	}

	std::optional<SaveEntryRecord> FileStorageProxy::FindSaveEntryByPath(const std::string& path) {
		RequireOpen();
		if (auto it = entriesByPath.find(path); it != entriesByPath.end() && EntryVisible(path)) {
			return it->second;
		}
		for (const auto& r : pendingEntries) {
			if (r.filePath == path) {
				return r;
			}
		}
		return std::nullopt;
	}

	std::optional<EmbedRecord> FileStorageProxy::FindEmbedById(int64_t id) {
		RequireOpen();
		const auto it = std::lower_bound(embeds.begin(), embeds.end(), id,
			[](const EmbedRecord& r, int64_t v) { return r.id < v; });
		if (it != embeds.end() && it->id == id && EmbedVisible(id)) {
			return *it;
		}
		for (const auto& r : pendingEmbeds) {
			if (r.id == id) {
				return r;
			}
		}
		return std::nullopt;
	}

	std::vector<int64_t> FileStorageProxy::FindEmbedIdsByLabel(const std::string& label) {
		RequireOpen();
		std::vector<int64_t> ids;
		for (const auto& r : embeds) {
			if (r.label == label && EmbedVisible(r.id)) ids.push_back(r.id);
		}
		for (const auto& r : pendingEmbeds) {
			if (r.label == label) ids.push_back(r.id);
		}
		return ids;
	}

	std::map<std::string, size_t> FileStorageProxy::CountEmbedsByLabel() {
		RequireOpen();
		std::map<std::string, size_t> counts;
		for (const auto& r : embeds) {
			if (EmbedVisible(r.id)) ++counts[r.label];
		}
		for (const auto& r : pendingEmbeds) {
			++counts[r.label];
		}
		return counts;
	}

	std::vector<EmbedRecord> FileStorageProxy::LoadAllEmbeds() {
		RequireOpen();
		std::vector<EmbedRecord> all;
		all.reserve(embeds.size() + pendingEmbeds.size());
		for (const auto& r : embeds) {
			if (EmbedVisible(r.id)) all.push_back(r);
		}
		all.insert(all.end(), pendingEmbeds.begin(), pendingEmbeds.end());
		return all;
	}

	int64_t FileStorageProxy::InsertEmbed(EmbedRecord rec) {
		return AutoTx([&] {
			if (rec.embed.empty() || rec.embed.size() > kMaxDim) {
				throw std::runtime_error("InsertEmbed: bad embedding size " + std::to_string(rec.embed.size()));
			}
			if (const size_t dim = EmbedDim(); dim != 0 && rec.embed.size() != dim) {
				throw std::runtime_error("InsertEmbed: dimension " + std::to_string(rec.embed.size()) +
					" does not match stored dimension " + std::to_string(dim));
			}
			if (rec.label.size() > kMaxStringLen || rec.createBy.size() > kMaxStringLen) {
				throw std::runtime_error("InsertEmbed: label or createBy too long");
			}

			rec.id = nextEmbedId++;
			const int64_t id = rec.id;
			pendingEmbeds.push_back(std::move(rec));
			return id;
		});
	}

	int64_t FileStorageProxy::InsertSaveEntry(SaveEntryRecord rec) {
		return AutoTx([&] {
			RequireTsvSafe(rec.filePath, "file_path");
			RequireTsvSafe(rec.createBy, "create_by");

			// The constraints a database would enforce: unique file_path, valid FK.
			if (FindSaveEntryByPath(rec.filePath)) {
				throw std::runtime_error("InsertSaveEntry: duplicate file_path " + rec.filePath);
			}
			if (!FindEmbedById(rec.embedId)) {
				throw std::runtime_error("InsertSaveEntry: embed id " + std::to_string(rec.embedId) + " does not exist");
			}

			rec.id = nextEntryId++;
			const int64_t id = rec.id;
			pendingEntries.push_back(std::move(rec));
			return id;
		});
	}

	size_t FileStorageProxy::DeleteSaveEntriesByEmbedIds(const std::vector<int64_t>& embedIds) {
		return AutoTx([&] {
			const std::set<int64_t> ids(embedIds.begin(), embedIds.end());
			size_t n = 0;
			for (const auto& [path, r] : entriesByPath) {
				if (EntryVisible(path) && ids.contains(r.embedId)) {
					deletedPaths.insert(path);
					++n;
				}
			}
			n += std::erase_if(pendingEntries, [&](const SaveEntryRecord& r) { return ids.contains(r.embedId); });
			return n;
		});
	}

	size_t FileStorageProxy::DeleteEmbeds(const std::vector<int64_t>& idList) {
		return AutoTx([&] {
			const std::set<int64_t> ids(idList.begin(), idList.end());

			// ON DELETE RESTRICT: the save entries must go first.
			for (const auto& [path, r] : entriesByPath) {
				if (EntryVisible(path) && ids.contains(r.embedId)) {
					throw std::runtime_error("DeleteEmbeds: embed id " + std::to_string(r.embedId) +
						" is still referenced by save entry " + path);
				}
			}
			for (const auto& r : pendingEntries) {
				if (ids.contains(r.embedId)) {
					throw std::runtime_error("DeleteEmbeds: embed id " + std::to_string(r.embedId) +
						" is still referenced by save entry " + r.filePath);
				}
			}

			size_t n = 0;
			for (const auto& r : embeds) {
				if (ids.contains(r.id) && EmbedVisible(r.id)) {
					deletedEmbedIds.insert(r.id);
					++n;
				}
			}
			n += std::erase_if(pendingEmbeds, [&](const EmbedRecord& r) { return ids.contains(r.id); });
			return n;
		});
	}

	void FileStorageProxy::Truncate() {
		AutoTx([&] {
			for (const auto& r : embeds) {
				deletedEmbedIds.insert(r.id);
			}
			for (const auto& [path, r] : entriesByPath) {
				deletedPaths.insert(path);
			}
			pendingEmbeds.clear();
			pendingEntries.clear();
			return 0;   // AutoTx needs a value
		});
	}

	void FileStorageProxy::LoadEmbeds() {
		embeds.clear();
		nextEmbedId = 1;

		if (!std::filesystem::exists(embedPath)) {
			std::ofstream f(embedPath, std::ios::binary);
			WriteEmbedHeader(f);
			if (!f) {
				throw std::runtime_error("Failed to create " + embedPath.string());
			}
			return;
		}

		std::ifstream f(embedPath, std::ios::binary);
		if (!f) {
			throw std::runtime_error("Failed to open " + embedPath.string());
		}
		char magic[4] = {};
		uint32_t version = 0;
		if (!f.read(magic, sizeof(magic)) || !std::equal(magic, magic + 4, kMagic)) {
			throw std::runtime_error("Not an embedding file (bad magic): " + embedPath.string());
		}
		ReadPod(f, version, embedPath);
		if (version != kVersion) {
			throw std::runtime_error("Unsupported embedding file version " + std::to_string(version) +
				" in " + embedPath.string());
		}

		while (f.peek() != std::char_traits<char>::eof()) {
			EmbedRecord r;
			ReadPod(f, r.id, embedPath);
			ReadPod(f, r.createOn, embedPath);
			r.label = ReadString(f, embedPath);
			r.createBy = ReadString(f, embedPath);
			uint32_t dim = 0;
			ReadPod(f, dim, embedPath);
			if (dim == 0 || dim > kMaxDim) {
				throw std::runtime_error("Corrupt embedding dimension in " + embedPath.string());
			}
			if (!embeds.empty() && dim != embeds.front().embed.size()) {
				throw std::runtime_error("Mixed embedding dimensions in " + embedPath.string());
			}
			if (!embeds.empty() && r.id <= embeds.back().id) {
				throw std::runtime_error("Embedding ids out of order in " + embedPath.string());
			}
			r.embed.resize(dim);
			if (!f.read(reinterpret_cast<char*>(r.embed.data()), dim * sizeof(float))) {
				throw std::runtime_error("Truncated record in " + embedPath.string());
			}
			nextEmbedId = r.id + 1;
			embeds.push_back(std::move(r));
		}
	}

	void FileStorageProxy::LoadSaveEntries() {
		entriesByPath.clear();
		nextEntryId = 1;

		if (!std::filesystem::exists(entryPath)) {
			std::ofstream f(entryPath, std::ios::binary);
			f << kEntryHeader << '\n';
			if (!f) {
				throw std::runtime_error("Failed to create " + entryPath.string());
			}
			return;
		}

		// Binary mode so the file has '\n' line endings on every platform; strip a stray '\r' anyway.
		std::ifstream f(entryPath, std::ios::binary);
		if (!f) {
			throw std::runtime_error("Failed to open " + entryPath.string());
		}
		std::string line;
		int lineNo = 0;
		while (std::getline(f, line)) {
			++lineNo;
			if (!line.empty() && line.back() == '\r') {
				line.pop_back();
			}
			if (lineNo == 1) {
				if (line != kEntryHeader) {
					throw std::runtime_error("Unexpected header in " + entryPath.string());
				}
				continue;
			}
			if (line.empty()) {
				continue;
			}

			const auto fields = SplitTabs(line);
			if (fields.size() != 5) {
				throw std::runtime_error("Malformed row at " + entryPath.string() + ":" + std::to_string(lineNo));
			}
			SaveEntryRecord r;
			try {
				r.id = std::stoll(fields[0]);
				r.filePath = fields[1];
				r.embedId = std::stoll(fields[2]);
				r.createOn = std::stoll(fields[3]);
				r.createBy = fields[4];
			}
			catch (const std::exception&) {
				throw std::runtime_error("Bad number at " + entryPath.string() + ":" + std::to_string(lineNo));
			}
			if (entriesByPath.contains(r.filePath)) {
				throw std::runtime_error("Duplicate file_path at " + entryPath.string() + ":" + std::to_string(lineNo));
			}
			nextEntryId = std::max(nextEntryId, r.id + 1);
			std::string key = r.filePath;
			entriesByPath.emplace(std::move(key), std::move(r));
		}
	}

	void FileStorageProxy::RequireOpen() const {
		if (!isOpen) {
			throw std::runtime_error("Storage not open: " + dir.string());
		}
	}

	size_t FileStorageProxy::EmbedDim() const {
		for (const auto& r : embeds) {
			if (EmbedVisible(r.id)) return r.embed.size();
		}
		if (!pendingEmbeds.empty()) return pendingEmbeds.front().embed.size();
		return 0;
	}
}
