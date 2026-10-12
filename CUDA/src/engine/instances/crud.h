#pragma once

#include "instance.h"

namespace ohy {

	/// <summary>
	/// Create/Read/Update/Delete data
	/// </summary>
	class CRUDInstance : public Instance {
	public:
		virtual ~CRUDInstance() = default;
	protected:
		virtual void InitSelf(const InitArg& initArg) override;
		virtual void RunSelf(const RunArg& runArg) override;
	};
}