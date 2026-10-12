#pragma once

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "storage_proxy.h"

namespace ohy {
    enum class AddStatus { Added, DuplicatePath, Failed };

    const char* ToString(AddStatus status);

    struct NewEmbed {
        std::string        filePath;
        std::string        label;
        std::vector<float> embed;   // raw model output; normalised on store
    };

    struct ReplaceResult {
        size_t deleted = 0;   // embeddings removed from the replaced labels
        size_t added = 0;
    };

    /// <summary>
    /// Business rules over the two tables, independent of where they are stored:
    ///  - a file path is embedded at most once (paths are canonicalised before comparing)
    ///  - an embedding and its save entry are written and deleted together, never one without the other
    ///  - embeddings are stored L2-normalised
    /// Owns the proxy; opens it on construction and closes it on destruction.
    /// </summary>
    class EmbedStore {
    public:
        explicit EmbedStore(std::unique_ptr<StorageProxy> proxy);
        ~EmbedStore();

        EmbedStore(const EmbedStore&) = delete;
        EmbedStore& operator=(const EmbedStore&) = delete;

        // Reads
        bool Contains(const std::string& filePath);   // cheap pre-check, before spending GPU time
        // Label of the embedding stored for this file; nullopt if the file is not stored.
        // An empty string means the save entry points at a missing embedding (see FileStorageProxy).
        std::optional<std::string> StoredLabel(const std::string& filePath);
        std::map<std::string, size_t> CountByLabel();
        std::vector<EmbedRecord> LoadAll();

        // Writes
        AddStatus Add(const std::string& filePath, const std::string& label,
            std::vector<float> embed, const std::string& createBy);
        // One transaction: delete every embedding (and save entry) under `labels`, then store `entries`.
        // Every entry's label must be in `labels`. All-or-nothing: throws and changes nothing on failure,
        // including when an entry's file is already stored under a label outside `labels`.
        ReplaceResult ReplaceLabels(const std::set<std::string>& labels, std::vector<NewEmbed> entries,
            const std::string& createBy);
        size_t DeleteLabel(const std::string& label);   // returns embeddings deleted
        size_t DeleteAll();                             // returns embeddings deleted

        // Absolute, lexically normalised, forward slashes; lower-cased on Windows (NTFS is case-insensitive).
        static std::string CanonicalPath(const std::string& p);

    private:
        // These run inside an open transaction.
        void InsertInTx(const std::string& canonicalPath, const std::string& label,
            std::vector<float> embed, const std::string& createBy, int64_t now);
        size_t DeleteLabelInTx(const std::string& label);
        void RollbackQuietly();

        std::unique_ptr<StorageProxy> proxy;
    };
}
