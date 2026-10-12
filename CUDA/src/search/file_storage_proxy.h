#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "storage_proxy.h"

namespace ohy {
    /// <summary>
    /// StorageProxy backed by two files in one directory:
    ///   save_entry.tsv  text, one row per line:  id  file_path  embed_id  create_on  create_by
    ///   main_data.bin   binary, header then rows (see file_storage_proxy.cpp)
    ///
    /// Open() loads both tables into memory. Commit() of a transaction that only inserts appends to
    /// the files; one that deletes anything rewrites both files (temp file + rename).
    ///
    /// Crash windows (this is what a real database fixes):
    ///  - append: the embed file is written first, so a crash leaves at worst an orphan embedding.
    ///  - rewrite: the save entry file is replaced first, so a crash can leave deleted embeddings
    ///    still in main_data.bin with no save entry. Both are reported as warnings on the next Open().
    /// Ids come from max(id) + 1 on load, so deleting the newest rows lets their ids be reused.
    /// The binary format is native little-endian, which is fine for x64 Windows/Linux.
    /// </summary>
    class FileStorageProxy : public StorageProxy {
    public:
        explicit FileStorageProxy(std::filesystem::path dir);
        ~FileStorageProxy() override;

        void Open() override;
        void Close() override;

        void Begin() override;
        void Commit() override;
        void Rollback() override;

        std::optional<SaveEntryRecord> FindSaveEntryByPath(const std::string& path) override;
        std::optional<EmbedRecord> FindEmbedById(int64_t id) override;
        std::vector<int64_t> FindEmbedIdsByLabel(const std::string& label) override;
        std::map<std::string, size_t> CountEmbedsByLabel() override;
        std::vector<EmbedRecord> LoadAllEmbeds() override;

        int64_t InsertEmbed(EmbedRecord rec) override;
        int64_t InsertSaveEntry(SaveEntryRecord rec) override;
        size_t DeleteSaveEntriesByEmbedIds(const std::vector<int64_t>& embedIds) override;
        size_t DeleteEmbeds(const std::vector<int64_t>& ids) override;
        void Truncate() override;

    private:
        // Runs f inside the caller's transaction, or in its own one if none is open.
        template <typename F> auto AutoTx(F&& f);

        void LoadSaveEntries();
        void LoadEmbeds();
        void CommitAppend();
        void CommitRewrite();
        void RequireOpen() const;
        size_t EmbedDim() const;   // 0 while no visible embedding exists
        bool EmbedVisible(int64_t id) const { return !deletedEmbedIds.contains(id); }
        bool EntryVisible(const std::string& path) const { return !deletedPaths.contains(path); }

        std::filesystem::path dir;
        std::filesystem::path entryPath;
        std::filesystem::path embedPath;
        bool isOpen = false;

        // Committed rows. embeds is in ascending id order.
        std::unordered_map<std::string, SaveEntryRecord> entriesByPath;
        std::vector<EmbedRecord> embeds;
        int64_t nextEntryId = 1;
        int64_t nextEmbedId = 1;

        // Uncommitted changes, plus the id counters to restore on Rollback().
        // Deletes of committed rows are recorded as tombstones; deletes of pending rows just erase them.
        bool inTx = false;
        std::vector<SaveEntryRecord> pendingEntries;
        std::vector<EmbedRecord> pendingEmbeds;
        std::set<std::string> deletedPaths;
        std::set<int64_t> deletedEmbedIds;
        int64_t txEntryId = 0;
        int64_t txEmbedId = 0;
    };
}
