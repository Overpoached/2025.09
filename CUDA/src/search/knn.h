#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "records.h"
#include "../engine/tensor.h"

namespace ohy {

    // Scales v to unit length. Returns false for a zero or non-finite vector (no direction to compare).
    bool L2Normalise(std::vector<float>& v);

    struct Neighbour {
        int64_t     id = 0;            // EmbedRecord::id
        std::string label;
        float       similarity = 0.f;  // cosine similarity, [-1, 1]; distance = 1 - similarity
    };

    /// <summary>
    /// Exact nearest-neighbour search over every stored embedding (no index structure).
    /// The catalog is uploaded to the GPU once as an [N, D] matrix; a batch of Q queries is scored with
    /// one GEMM, [Q, D] x [N, D]^T -> [Q, N] cosine similarities, using our own matmul kernel.
    /// Stored and query vectors must both be L2-normalised, so a dot product is the cosine similarity.
    /// </summary>
    class BruteForceIndex {
    public:
        explicit BruteForceIndex(const std::vector<EmbedRecord>& records);

        size_t Size() const { return ids.size(); }
        size_t Dim() const { return dim; }
        size_t LabelCount() const;

        // Top-k neighbours per query, most similar first. k is clamped to Size().
        std::vector<std::vector<Neighbour>> Search(const std::vector<std::vector<float>>& queries, size_t k) const;

    private:
        size_t dim = 0;
        std::vector<int64_t> ids;        // row -> EmbedRecord::id
        std::vector<std::string> labels; // row -> label
        TensorPtr catalog;               // device [N, D]
    };

    struct LabelScore {
        std::string label;
        float   score = 0.f;           // sum of the similarities of this label's neighbours
        float   share = 0.f;           // score / sum of all scores, 0..1
        int     votes = 0;             // how many of the k neighbours have this label
        float   bestSimilarity = 0.f;  // this label's nearest neighbour
        int64_t bestId = 0;
    };

    // Similarity-weighted kNN vote over one query's neighbours, best label first.
    // Ranked by score, ties broken by the nearest neighbour. Negative similarities carry no weight.
    std::vector<LabelScore> VoteLabels(const std::vector<Neighbour>& neighbours);
}
