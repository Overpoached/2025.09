#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "storage_proxy.h"

namespace ohy {
    /// <summary>
    /// StorageProxy backed by two files in one directory:
    ///   save_entry.tsv  text, one row per line:  id  file_path  embed_id  create_on  create_by
    ///   main_data.bin   binary, header then append-only rows (see file_storage_proxy.cpp)
    ///
    /// Open() loads both tables into memory; Commit() appends to the files.
    /// The embed file is written before the save entry file, so a crash between the two
    /// leaves at worst an orphan embedding (reported as a warning on the next Open()).
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
        int64_t InsertEmbed(EmbedRecord rec) override;
        int64_t InsertSaveEntry(SaveEntryRecord rec) override;
        std::vector<EmbedRecord> LoadAllEmbeds() override;

    private:
        void LoadSaveEntries();
        void LoadEmbeds();
        void RequireOpen() const;
        size_t EmbedDim() const;   // 0 while the table is empty

        std::filesystem::path dir;
        std::filesystem::path entryPath;
        std::filesystem::path embedPath;
        bool isOpen = false;

        // Committed rows.
        std::unordered_map<std::string, SaveEntryRecord> entriesByPath;
        std::vector<EmbedRecord> embeds;
        int64_t nextEntryId = 1;
        int64_t nextEmbedId = 1;

        // Uncommitted rows, plus the id counters to restore on Rollback().
        bool inTx = false;
        std::vector<SaveEntryRecord> pendingEntries;
        std::vector<EmbedRecord> pendingEmbeds;
        int64_t txEntryId = 0;
        int64_t txEmbedId = 0;
    };
}
