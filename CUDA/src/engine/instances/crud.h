#pragma once

#include <memory>

#include "instance.h"

#include "../../search/embed_store.h"

namespace ohy {

	/// <summary>
	/// Create/Read/Update/Delete data.
	/// Currently: embed a list of images and add them to the store, skipping files already stored.
	/// </summary>
	class CRUDInstance : public Instance {
	public:
		virtual ~CRUDInstance() = default;
	protected:
		virtual void InitSelf(const InitArg& initArg) override;
		virtual void RunSelf(const RunArg& runArg) override;

		std::unique_ptr<EmbedStore> store;
	};
}
