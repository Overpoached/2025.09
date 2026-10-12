#pragma once

#include <optional>
#include <string>
#include <vector>

#include "records.h"

namespace ohy {
    /// <summary>
    /// Storage backend for the two tables. Stores and loads rows, enforces no business rules
    /// (that is EmbedStore's job). File-backed now; a Postgres backend implements the same interface.
    ///
    /// Inserts made between Begin() and Commit() become visible to this proxy immediately,
    /// but only reach durable storage on Commit(). Rollback() discards them.
    /// An insert outside a transaction commits on its own.
    /// Not thread-safe.
    /// </summary>
    class StorageProxy {
    public:
        virtual ~StorageProxy() = default;
        virtual void Open() = 0;
        virtual void Close() = 0;

        virtual void Begin() = 0;
        virtual void Commit() = 0;
        virtual void Rollback() = 0;

        virtual std::optional<SaveEntryRecord> FindSaveEntryByPath(const std::string& path) = 0;
        virtual int64_t InsertEmbed(EmbedRecord rec) = 0;          // ignores rec.id, returns assigned id
        virtual int64_t InsertSaveEntry(SaveEntryRecord rec) = 0;  // ignores rec.id, returns assigned id
        virtual std::vector<EmbedRecord> LoadAllEmbeds() = 0;
    };
}
