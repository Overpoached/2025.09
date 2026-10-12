// kNN search and label voting tests. BruteForceIndex runs on the GPU; VoteLabels is pure CPU.

#include "test.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../search/knn.h"

namespace {
	std::vector<float> RandomUnit(std::mt19937& rng, size_t dim) {
		std::normal_distribution<float> dist(0.f, 1.f);
		std::vector<float> v(dim);
		for (float& x : v) x = dist(rng);
		ohy::L2Normalise(v);
		return v;
	}
}

TEST_CASE(Knn_L2Normalise) {
	std::vector<float> v = { 3.f, 4.f };
	EXPECT_EQ(ohy::L2Normalise(v), true);
	EXPECT_NEAR(v[0], 0.6, 1e-6);
	EXPECT_NEAR(v[1], 0.8, 1e-6);

	std::vector<float> zero = { 0.f, 0.f };
	EXPECT_EQ(ohy::L2Normalise(zero), false);
}

// GPU GEMM similarities and top-k order vs a CPU brute force. Sizes are not tile multiples on purpose.
TEST_CASE(Knn_BruteForceIndex_MatchesCpu) {
	constexpr size_t kDim = 37, kRows = 53, kQueries = 19, kK = 7;
	std::mt19937 rng(1234);

	std::vector<ohy::EmbedRecord> records(kRows);
	for (size_t i = 0; i < kRows; ++i) {
		records[i].id = static_cast<int64_t>(i + 1);
		records[i].label = "label" + std::to_string(i % 5);
		records[i].embed = RandomUnit(rng, kDim);
	}
	std::vector<std::vector<float>> queries(kQueries);
	for (auto& q : queries) q = RandomUnit(rng, kDim);

	ohy::BruteForceIndex index(records);
	EXPECT_EQ(index.Size(), kRows);
	EXPECT_EQ(index.Dim(), kDim);
	EXPECT_EQ(index.LabelCount(), 5u);

	const auto results = index.Search(queries, kK);
	EXPECT_EQ(results.size(), kQueries);

	for (size_t q = 0; q < kQueries; ++q) {
		std::vector<double> sims(kRows);
		for (size_t r = 0; r < kRows; ++r) {
			sims[r] = std::inner_product(queries[q].begin(), queries[q].end(), records[r].embed.begin(), 0.0);
		}
		std::vector<size_t> order(kRows);
		std::iota(order.begin(), order.end(), size_t(0));
		std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return sims[a] > sims[b]; });

		EXPECT_EQ(results[q].size(), kK);
		for (size_t j = 0; j < kK; ++j) {
			EXPECT_EQ(results[q][j].id, records[order[j]].id);
			EXPECT_EQ(results[q][j].label, records[order[j]].label);
			EXPECT_NEAR(results[q][j].similarity, sims[order[j]], 1e-5);
		}
	}
}

TEST_CASE(Knn_BruteForceIndex_ExactMatchAndClampedK) {
	std::mt19937 rng(99);
	std::vector<ohy::EmbedRecord> records(3);
	for (size_t i = 0; i < records.size(); ++i) {
		records[i].id = static_cast<int64_t>(10 + i);
		records[i].label = "L" + std::to_string(i);
		records[i].embed = RandomUnit(rng, 16);
	}
	ohy::BruteForceIndex index(records);

	// Querying with a stored vector: it is its own nearest neighbour, distance ~0. k > N is clamped.
	const auto results = index.Search({ records[1].embed }, 10);
	EXPECT_EQ(results[0].size(), 3u);
	EXPECT_EQ(results[0][0].id, 11);
	EXPECT_NEAR(results[0][0].similarity, 1.0, 1e-5);
}

TEST_CASE(Knn_Vote_MajorityBeatsSingleNearest) {
	// One very close "a", three fairly close "b": the vote goes to "b", but "a" keeps the nearest neighbour.
	const std::vector<ohy::Neighbour> neighbours = {
		{ 1, "a", 0.95f }, { 2, "b", 0.90f }, { 3, "b", 0.89f }, { 4, "b", 0.88f },
	};
	const auto votes = ohy::VoteLabels(neighbours);
	EXPECT_EQ(votes.size(), 2u);
	EXPECT_EQ(votes[0].label, std::string("b"));
	EXPECT_EQ(votes[0].votes, 3);
	EXPECT_EQ(votes[0].bestId, 2);
	EXPECT_NEAR(votes[0].score, 0.90 + 0.89 + 0.88, 1e-5);
	EXPECT_NEAR(votes[0].share, (0.90 + 0.89 + 0.88) / (0.95 + 0.90 + 0.89 + 0.88), 1e-5);
	EXPECT_EQ(votes[1].label, std::string("a"));
	EXPECT_NEAR(votes[1].bestSimilarity, 0.95, 1e-6);
}

TEST_CASE(Knn_Vote_TieBrokenByNearest) {
	const std::vector<ohy::Neighbour> neighbours = {
		{ 1, "a", 0.75f }, { 2, "b", 0.875f }, { 3, "a", 0.5f }, { 4, "b", 0.375f },
	};
	// Equal scores (1.25 each, exact in float): "b" wins on its nearer neighbour.
	const auto votes = ohy::VoteLabels(neighbours);
	EXPECT_EQ(votes[0].label, std::string("b"));
	EXPECT_EQ(votes[1].label, std::string("a"));
}

TEST_CASE(Knn_Vote_NegativeSimilarityHasNoWeight) {
	const std::vector<ohy::Neighbour> neighbours = { { 1, "a", 0.5f }, { 2, "b", -0.2f } };
	const auto votes = ohy::VoteLabels(neighbours);
	EXPECT_EQ(votes[0].label, std::string("a"));
	EXPECT_NEAR(votes[0].share, 1.0, 1e-6);
	EXPECT_NEAR(votes[1].share, 0.0, 1e-6);
}
