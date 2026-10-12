#include "knn.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime.h>

#include "../kernels/kernel_utility.h"
#include "../kernels/nn/nn.h"

namespace ohy {

	bool L2Normalise(std::vector<float>& v) {
		double sumSq = 0.0;
		for (float x : v) sumSq += static_cast<double>(x) * x;
		if (!(sumSq > 0.0) || !std::isfinite(sumSq)) {
			return false;
		}
		const double inv = 1.0 / std::sqrt(sumSq);
		for (float& x : v) x = static_cast<float>(x * inv);
		return true;
	}

	BruteForceIndex::BruteForceIndex(const std::vector<EmbedRecord>& records) {
		if (records.empty()) {
			return;
		}
		dim = records.front().embed.size();
		if (dim == 0) {
			throw std::runtime_error("BruteForceIndex: empty embedding");
		}

		std::vector<float> host;
		host.reserve(records.size() * dim);
		ids.reserve(records.size());
		labels.reserve(records.size());
		for (const auto& r : records) {
			if (r.embed.size() != dim) {
				throw std::runtime_error("BruteForceIndex: embedding " + std::to_string(r.id) + " has dimension " +
					std::to_string(r.embed.size()) + ", expected " + std::to_string(dim));
			}
			host.insert(host.end(), r.embed.begin(), r.embed.end());
			ids.push_back(r.id);
			labels.push_back(r.label);
		}

		const std::vector<int64_t> shape = { static_cast<int64_t>(records.size()), static_cast<int64_t>(dim) };
		catalog = CreateTensor(Location::Device, DataType::FLOAT32, shape,
			reinterpret_cast<const char*>(host.data()), host.size() * sizeof(float));
	}

	size_t BruteForceIndex::LabelCount() const {
		return std::unordered_set<std::string>(labels.begin(), labels.end()).size();
	}

	std::vector<std::vector<Neighbour>> BruteForceIndex::Search(const std::vector<std::vector<float>>& queries,
		size_t k) const {

		std::vector<std::vector<Neighbour>> out(queries.size());
		const size_t n = Size();
		k = std::min(k, n);
		if (queries.empty() || k == 0) {
			return out;
		}

		// Score queries in chunks so the [Q, N] similarity matrix stays around 64 MB.
		const size_t chunk = std::max<size_t>(1, (size_t(16) << 20) / n);
		std::vector<float> hostQueries;
		std::vector<float> sims;
		std::vector<uint32_t> order(n);

		for (size_t start = 0; start < queries.size(); start += chunk) {
			const size_t q = std::min(chunk, queries.size() - start);

			hostQueries.clear();
			hostQueries.reserve(q * dim);
			for (size_t i = 0; i < q; ++i) {
				const auto& v = queries[start + i];
				if (v.size() != dim) {
					throw std::runtime_error("BruteForceIndex: query has dimension " + std::to_string(v.size()) +
						", stored embeddings have " + std::to_string(dim));
				}
				hostQueries.insert(hostQueries.end(), v.begin(), v.end());
			}

			auto dQueries = CreateTensor(Location::Device, DataType::FLOAT32,
				{ static_cast<int64_t>(q), static_cast<int64_t>(dim) },
				reinterpret_cast<const char*>(hostQueries.data()), hostQueries.size() * sizeof(float));
			auto dSims = CreateTensor(Location::Device, DataType::FLOAT32,
				{ static_cast<int64_t>(q), static_cast<int64_t>(n) }, nullptr, q * n * sizeof(float));

			// sims[Q, N] = queries[Q, D] x catalog[N, D]^T
			kernels::GEMMShape shape;
			shape.m = static_cast<int64_t>(q);
			shape.k = static_cast<int64_t>(dim);
			shape.n = static_cast<int64_t>(n);
			shape.transB = 1;
			shape.alpha = 1.0f;
			shape.beta = 0.0f;
			kernels::GEMM(dQueries->f32(), catalog->f32(), nullptr, dSims->f32(), shape, nullptr);

			sims.resize(q * n);
			CUDA_CHECK(cudaMemcpy(sims.data(), dSims->f32(), sims.size() * sizeof(float), cudaMemcpyDeviceToHost));

			for (size_t i = 0; i < q; ++i) {
				const float* row = sims.data() + i * n;
				std::iota(order.begin(), order.end(), 0u);
				// Most similar first; equal similarity -> older (smaller id) first, so results are deterministic.
				std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](uint32_t a, uint32_t b) {
					return row[a] != row[b] ? row[a] > row[b] : ids[a] < ids[b];
				});

				auto& neighbours = out[start + i];
				neighbours.reserve(k);
				for (size_t j = 0; j < k; ++j) {
					const uint32_t r = order[j];
					neighbours.push_back({ ids[r], labels[r], row[r] });
				}
			}
		}
		return out;
	}

	std::vector<LabelScore> VoteLabels(const std::vector<Neighbour>& neighbours) {
		std::vector<LabelScore> scores;
		std::unordered_map<std::string, size_t> slot;
		for (const auto& nb : neighbours) {
			auto [it, inserted] = slot.try_emplace(nb.label, scores.size());
			if (inserted) {
				LabelScore s;
				s.label = nb.label;
				s.bestSimilarity = nb.similarity;
				s.bestId = nb.id;
				scores.push_back(std::move(s));
			}
			LabelScore& s = scores[it->second];
			s.score += std::max(nb.similarity, 0.0f);
			s.votes += 1;
			if (nb.similarity > s.bestSimilarity) {
				s.bestSimilarity = nb.similarity;
				s.bestId = nb.id;
			}
		}

		float total = 0.f;
		for (const auto& s : scores) total += s.score;
		for (auto& s : scores) s.share = total > 0.f ? s.score / total : 0.f;

		std::sort(scores.begin(), scores.end(), [](const LabelScore& a, const LabelScore& b) {
			return a.score != b.score ? a.score > b.score : a.bestSimilarity > b.bestSimilarity;
		});
		return scores;
	}
}
