#pragma once

#include "instance.h"

namespace ohy {

	/// <summary>
	/// Matches image against database of embed to find the closest match
	/// </summary>
	class InferInstance : public Instance {
	public:
		virtual ~InferInstance() = default;
	protected:
		virtual void InitSelf(const InitArg& initArg) override;
		virtual void RunSelf(const RunArg& runArg) override;
	};
}