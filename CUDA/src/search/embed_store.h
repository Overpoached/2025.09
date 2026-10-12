#pragma once

#include <memory>
#include <string>
#include <vector>

#include "storage_proxy.h"

namespace ohy {
    enum class AddStatus { Added, DuplicatePath, Failed };

    const char* ToString(AddStatus status);

    /// <summary>
    /// Business rules over the two tables, independent of where they are stored:
    ///  - a file path is embedded at most once (paths are canonicalised before comparing)
    ///  - the embedding row and its save entry are written together or not at all
    ///  - embeddings are stored L2-normalised
    /// Owns the proxy; opens it on construction and closes it on destruction.
    /// </summary>
    class EmbedStore {
    public:
        explicit EmbedStore(std::unique_ptr<StorageProxy> proxy);
        ~EmbedStore();

        EmbedStore(const EmbedStore&) = delete;
        EmbedStore& operator=(const EmbedStore&) = delete;

        bool Contains(const std::string& filePath);   // cheap pre-check, before spending GPU time
        AddStatus Add(const std::string& filePath, const std::string& label,
            std::vector<float> embed, const std::string& createBy);
        std::vector<EmbedRecord> LoadAll();

        // Absolute, lexically normalised, forward slashes; lower-cased on Windows (NTFS is case-insensitive).
        static std::string CanonicalPath(const std::string& p);

    private:
        std::unique_ptr<StorageProxy> proxy;
    };
}
