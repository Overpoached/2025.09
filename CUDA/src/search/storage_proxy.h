#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "records.h"

namespace ohy {
    /// <summary>
    /// Storage backend for the two tables. Stores and loads rows, enforces no business rules
    /// (that is EmbedStore's job). File-backed now; a Postgres backend implements the same interface.
    ///
    /// Changes made between Begin() and Commit() are visible to this proxy immediately,
    /// but only reach durable storage on Commit(). Rollback() discards them.
    /// A change made outside a transaction commits on its own.
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

        // Reads
        virtual std::optional<SaveEntryRecord> FindSaveEntryByPath(const std::string& path) = 0;
        virtual std::optional<EmbedRecord> FindEmbedById(int64_t id) = 0;
        virtual std::vector<int64_t> FindEmbedIdsByLabel(const std::string& label) = 0;
        virtual std::map<std::string, size_t> CountEmbedsByLabel() = 0;   // SELECT label, COUNT(*) GROUP BY label
        virtual std::vector<EmbedRecord> LoadAllEmbeds() = 0;

        // Writes
        virtual int64_t InsertEmbed(EmbedRecord rec) = 0;          // ignores rec.id, returns assigned id
        virtual int64_t InsertSaveEntry(SaveEntryRecord rec) = 0;  // ignores rec.id, returns assigned id
        virtual size_t DeleteSaveEntriesByEmbedIds(const std::vector<int64_t>& embedIds) = 0;   // returns rows deleted
        virtual size_t DeleteEmbeds(const std::vector<int64_t>& ids) = 0;   // throws if a save entry still points at one
        virtual void Truncate() = 0;                                         // delete every row of both tables
    };
}
