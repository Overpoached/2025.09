#pragma once

#include "instance.h"

namespace ohy {

	/// <summary>
	/// Compares list of images to see which one is the closest match
	/// </summary>
	class CompareInstance : public Instance {
	public:
		virtual ~CompareInstance() = default;
	protected:
		virtual void InitSelf(const InitArg& initArg) override;
		virtual void RunSelf(const RunArg& runArg) override;
	};
}