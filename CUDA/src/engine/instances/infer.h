#pragma once

#include <memory>

#include "instance.h"

#include "../../search/knn.h"

namespace ohy {

	/// <summary>
	/// Matches image against database of embed to find the closest match.
	/// Loads every stored embedding onto the GPU at init, then for each input image finds its
	/// k nearest stored embeddings and votes on the label.
	/// </summary>
	class InferInstance : public Instance {
	public:
		virtual ~InferInstance() = default;
	protected:
		virtual void InitSelf(const InitArg& initArg) override;
		virtual void RunSelf(const RunArg& runArg) override;

		std::unique_ptr<BruteForceIndex> index;
	};
}
