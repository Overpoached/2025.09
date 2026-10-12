#pragma once

#include <memory>

#include "instance.h"

#include "../../search/embed_store.h"

namespace ohy {

	/// <summary>
	/// Create/Read/Update/Delete data in the embedding store. RunArg::crudOp picks the operation:
	///   Add     embed images and store them, skipping files already stored
	///   Update  replace everything stored under the labels of the given images with those images
	///   Delete  delete everything under the given labels, or everything (RunArg::all)
	///   Read    count embeddings per label
	/// </summary>
	class CRUDInstance : public Instance {
	public:
		virtual ~CRUDInstance() = default;
	protected:
		virtual void InitSelf(const InitArg& initArg) override;
		virtual void RunSelf(const RunArg& runArg) override;

		std::unique_ptr<EmbedStore> store;

	private:
		void RunAdd(const RunArg& runArg);
		void RunUpdate(const RunArg& runArg);
		void RunDelete(const RunArg& runArg);
		void RunRead(const RunArg& runArg);
	};
}
