#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ohy {
    // "Save entry" table: one row per source file that has been embedded.
    // Used to reject the same file twice, and to trace an embedding back to its file.
    struct SaveEntryRecord {
        int64_t     id = 0;
        std::string filePath;      // canonical (see EmbedStore::CanonicalPath)
        int64_t     embedId = 0;   // FK -> EmbedRecord::id
        int64_t     createOn = 0;  // unix ms UTC; maps cleanly to timestamptz later
        std::string createBy;
    };

    // Main data table: the embeddings that nearest-neighbour search runs over.
    // Labels may repeat: several photos of the same model are several rows.
    struct EmbedRecord {
        int64_t            id = 0;
        std::string        label;
        std::vector<float> embed;  // L2-normalised, so dot product == cosine similarity
        int64_t            createOn = 0;
        std::string        createBy;
    };
}
