// Storage layer tests: FileStorageProxy and EmbedStore. Pure C++, no GPU needed.

#include "test.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../search/embed_store.h"
#include "../search/file_storage_proxy.h"

namespace {
	namespace fs = std::filesystem;

	// A fresh, empty directory per test.
	fs::path TempDir(const std::string& name) {
		const fs::path dir = fs::temp_directory_path() / ("ohy_storage_test_" + name);
		fs::remove_all(dir);
		fs::create_directories(dir);
		return dir;
	}

	ohy::EmbedRecord MakeEmbed(const std::string& label, std::vector<float> v) {
		ohy::EmbedRecord r;
		r.label = label;
		r.embed = std::move(v);
		r.createOn = 1234;
		r.createBy = "tester";
		return r;
	}

	ohy::SaveEntryRecord MakeEntry(const std::string& path, int64_t embedId) {
		ohy::SaveEntryRecord r;
		r.filePath = path;
		r.embedId = embedId;
		r.createOn = 1234;
		r.createBy = "tester";
		return r;
	}

	// Throws on the save entry insert, after the embed insert has gone through.
	class FailingEntryProxy : public ohy::FileStorageProxy {
	public:
		using FileStorageProxy::FileStorageProxy;
		int64_t InsertSaveEntry(ohy::SaveEntryRecord) override {
			throw std::runtime_error("injected failure");
		}
	};
}

TEST_CASE(Storage_FileProxy_PersistsAcrossReopen) {
	const fs::path dir = TempDir("persist");
	{
		ohy::FileStorageProxy proxy(dir);
		proxy.Open();
		proxy.Begin();
		const int64_t embedId = proxy.InsertEmbed(MakeEmbed("civic", { 0.6f, 0.8f }));
		const int64_t entryId = proxy.InsertSaveEntry(MakeEntry("c:/x/a.jpg", embedId));
		proxy.Commit();
		EXPECT_EQ(embedId, 1);
		EXPECT_EQ(entryId, 1);
	}

	ohy::FileStorageProxy proxy(dir);
	proxy.Open();
	const auto embeds = proxy.LoadAllEmbeds();
	EXPECT_EQ(embeds.size(), 1u);
	EXPECT_EQ(embeds[0].id, 1);
	EXPECT_EQ(embeds[0].label, std::string("civic"));
	EXPECT_EQ(embeds[0].createBy, std::string("tester"));
	EXPECT_EQ(embeds[0].createOn, 1234);
	EXPECT_EQ(embeds[0].embed.size(), 2u);
	EXPECT_NEAR(embeds[0].embed[0], 0.6f, 0.0f);   // binary round trip is exact
	EXPECT_NEAR(embeds[0].embed[1], 0.8f, 0.0f);

	const auto entry = proxy.FindSaveEntryByPath("c:/x/a.jpg");
	EXPECT_EQ(entry.has_value(), true);
	EXPECT_EQ(entry->embedId, 1);

	// Ids continue from what was loaded, not from 1.
	proxy.Begin();
	EXPECT_EQ(proxy.InsertEmbed(MakeEmbed("civic", { 1.0f, 0.0f })), 2);
	proxy.Commit();
}

TEST_CASE(Storage_FileProxy_RollbackDiscards) {
	const fs::path dir = TempDir("rollback");
	{
		ohy::FileStorageProxy proxy(dir);
		proxy.Open();
		proxy.Begin();
		const int64_t embedId = proxy.InsertEmbed(MakeEmbed("civic", { 1.0f, 0.0f }));
		proxy.InsertSaveEntry(MakeEntry("c:/x/a.jpg", embedId));
		proxy.Rollback();

		EXPECT_EQ(proxy.LoadAllEmbeds().size(), 0u);
		EXPECT_EQ(proxy.FindSaveEntryByPath("c:/x/a.jpg").has_value(), false);

		// The rolled-back ids are reused.
		proxy.Begin();
		EXPECT_EQ(proxy.InsertEmbed(MakeEmbed("civic", { 1.0f, 0.0f })), 1);
		proxy.Rollback();
	}

	ohy::FileStorageProxy proxy(dir);
	proxy.Open();
	EXPECT_EQ(proxy.LoadAllEmbeds().size(), 0u);
}

TEST_CASE(Storage_FileProxy_EnforcesConstraints) {
	const fs::path dir = TempDir("constraints");
	ohy::FileStorageProxy proxy(dir);
	proxy.Open();

	const int64_t embedId = proxy.InsertEmbed(MakeEmbed("civic", { 1.0f, 0.0f }));   // autocommit
	proxy.InsertSaveEntry(MakeEntry("c:/x/a.jpg", embedId));

	bool threw = false;
	try { proxy.InsertSaveEntry(MakeEntry("c:/x/a.jpg", embedId)); } catch (const std::exception&) { threw = true; }
	EXPECT_EQ(threw, true);   // duplicate file_path

	threw = false;
	try { proxy.InsertSaveEntry(MakeEntry("c:/x/b.jpg", 99)); } catch (const std::exception&) { threw = true; }
	EXPECT_EQ(threw, true);   // embed id does not exist

	threw = false;
	try { proxy.InsertEmbed(MakeEmbed("civic", { 1.0f, 0.0f, 0.0f })); } catch (const std::exception&) { threw = true; }
	EXPECT_EQ(threw, true);   // dimension mismatch

	threw = false;
	try { proxy.InsertSaveEntry(MakeEntry("c:/x/tab\there.jpg", embedId)); } catch (const std::exception&) { threw = true; }
	EXPECT_EQ(threw, true);   // would break the TSV
}

TEST_CASE(Storage_FileProxy_RejectsCorruptFile) {
	const fs::path dir = TempDir("corrupt");
	std::ofstream(dir / "main_data.bin", std::ios::binary) << "garbage!";

	ohy::FileStorageProxy proxy(dir);
	bool threw = false;
	try { proxy.Open(); } catch (const std::exception&) { threw = true; }
	EXPECT_EQ(threw, true);
}

TEST_CASE(Storage_FileProxy_RejectsTruncatedRecord) {
	const fs::path dir = TempDir("truncated");
	{
		ohy::FileStorageProxy proxy(dir);
		proxy.Open();
		proxy.InsertEmbed(MakeEmbed("civic", { 1.0f, 0.0f }));
	}
	const fs::path bin = dir / "main_data.bin";
	fs::resize_file(bin, fs::file_size(bin) - 2);

	ohy::FileStorageProxy proxy(dir);
	bool threw = false;
	try { proxy.Open(); } catch (const std::exception&) { threw = true; }
	EXPECT_EQ(threw, true);
}

TEST_CASE(Storage_EmbedStore_RejectsDuplicatePath) {
	const fs::path dir = TempDir("dup");
	ohy::EmbedStore store(std::make_unique<ohy::FileStorageProxy>(dir / "db"));

	const std::string path = (dir / "img" / "a.jpg").string();
	EXPECT_EQ(static_cast<int>(store.Add(path, "civic", { 1.0f, 0.0f }, "tester")),
		static_cast<int>(ohy::AddStatus::Added));
	EXPECT_EQ(store.Contains(path), true);

	// Same file, spelled differently.
	const std::string sameFile = (dir / "img" / "sub" / ".." / "." / "a.jpg").string();
	EXPECT_EQ(static_cast<int>(store.Add(sameFile, "civic", { 1.0f, 0.0f }, "tester")),
		static_cast<int>(ohy::AddStatus::DuplicatePath));
#ifdef _WIN32
	const std::string upper = (dir / "IMG" / "A.JPG").string();
	EXPECT_EQ(static_cast<int>(store.Add(upper, "civic", { 1.0f, 0.0f }, "tester")),
		static_cast<int>(ohy::AddStatus::DuplicatePath));
#endif

	// Same label, different file: allowed.
	EXPECT_EQ(static_cast<int>(store.Add((dir / "img" / "b.jpg").string(), "civic", { 0.0f, 1.0f }, "tester")),
		static_cast<int>(ohy::AddStatus::Added));
	EXPECT_EQ(store.LoadAll().size(), 2u);
}

TEST_CASE(Storage_EmbedStore_StoresNormalised) {
	const fs::path dir = TempDir("normalise");
	ohy::EmbedStore store(std::make_unique<ohy::FileStorageProxy>(dir));

	store.Add((dir / "a.jpg").string(), "civic", { 3.0f, 4.0f }, "tester");
	const auto all = store.LoadAll();
	EXPECT_EQ(all.size(), 1u);
	EXPECT_NEAR(all[0].embed[0], 0.6, 1e-6);
	EXPECT_NEAR(all[0].embed[1], 0.8, 1e-6);

	EXPECT_EQ(static_cast<int>(store.Add((dir / "zero.jpg").string(), "civic", { 0.0f, 0.0f }, "tester")),
		static_cast<int>(ohy::AddStatus::Failed));
}

TEST_CASE(Storage_EmbedStore_FailureLeavesNoOrphan) {
	const fs::path dir = TempDir("orphan");
	{
		ohy::EmbedStore store(std::make_unique<FailingEntryProxy>(dir));
		EXPECT_EQ(static_cast<int>(store.Add((dir / "a.jpg").string(), "civic", { 1.0f, 0.0f }, "tester")),
			static_cast<int>(ohy::AddStatus::Failed));
		EXPECT_EQ(store.LoadAll().size(), 0u);
	}

	ohy::FileStorageProxy proxy(dir);
	proxy.Open();
	EXPECT_EQ(proxy.LoadAllEmbeds().size(), 0u);
}

TEST_CASE(Storage_FileProxy_DeleteRewritesFiles) {
	const fs::path dir = TempDir("delete");
	{
		ohy::FileStorageProxy proxy(dir);
		proxy.Open();
		const int64_t a1 = proxy.InsertEmbed(MakeEmbed("a", { 1.0f, 0.0f }));
		const int64_t a2 = proxy.InsertEmbed(MakeEmbed("a", { 0.0f, 1.0f }));
		const int64_t b1 = proxy.InsertEmbed(MakeEmbed("b", { 0.6f, 0.8f }));
		proxy.InsertSaveEntry(MakeEntry("c:/x/a1.jpg", a1));
		proxy.InsertSaveEntry(MakeEntry("c:/x/a2.jpg", a2));
		proxy.InsertSaveEntry(MakeEntry("c:/x/b1.jpg", b1));

		proxy.Begin();
		const auto ids = proxy.FindEmbedIdsByLabel("a");
		EXPECT_EQ(ids.size(), 2u);
		EXPECT_EQ(proxy.DeleteSaveEntriesByEmbedIds(ids), 2u);
		EXPECT_EQ(proxy.DeleteEmbeds(ids), 2u);
		// Visible inside the transaction already.
		EXPECT_EQ(proxy.FindSaveEntryByPath("c:/x/a1.jpg").has_value(), false);
		EXPECT_EQ(proxy.FindEmbedById(a1).has_value(), false);
		proxy.Commit();
	}

	ohy::FileStorageProxy proxy(dir);
	proxy.Open();
	const auto counts = proxy.CountEmbedsByLabel();
	EXPECT_EQ(counts.size(), 1u);
	EXPECT_EQ(counts.at("b"), 1u);
	EXPECT_EQ(proxy.FindSaveEntryByPath("c:/x/a1.jpg").has_value(), false);
	EXPECT_EQ(proxy.FindSaveEntryByPath("c:/x/b1.jpg").has_value(), true);
	EXPECT_EQ(proxy.FindEmbedById(3).has_value(), true);
	EXPECT_EQ(fs::exists(dir / "main_data.bin.tmp"), false);
	EXPECT_EQ(fs::exists(dir / "save_entry.tsv.tmp"), false);

	// The deleted path can be stored again.
	const int64_t again = proxy.InsertEmbed(MakeEmbed("a", { 1.0f, 0.0f }));
	proxy.InsertSaveEntry(MakeEntry("c:/x/a1.jpg", again));
	EXPECT_EQ(proxy.CountEmbedsByLabel().at("a"), 1u);
}

TEST_CASE(Storage_FileProxy_DeleteEmbedStillReferencedThrows) {
	const fs::path dir = TempDir("restrict");
	ohy::FileStorageProxy proxy(dir);
	proxy.Open();
	const int64_t id = proxy.InsertEmbed(MakeEmbed("a", { 1.0f, 0.0f }));
	proxy.InsertSaveEntry(MakeEntry("c:/x/a.jpg", id));

	bool threw = false;
	try { proxy.DeleteEmbeds({ id }); } catch (const std::exception&) { threw = true; }
	EXPECT_EQ(threw, true);
	EXPECT_EQ(proxy.FindEmbedById(id).has_value(), true);
}

TEST_CASE(Storage_FileProxy_DeleteRollback) {
	const fs::path dir = TempDir("delete_rollback");
	ohy::FileStorageProxy proxy(dir);
	proxy.Open();
	const int64_t id = proxy.InsertEmbed(MakeEmbed("a", { 1.0f, 0.0f }));
	proxy.InsertSaveEntry(MakeEntry("c:/x/a.jpg", id));

	proxy.Begin();
	proxy.Truncate();
	EXPECT_EQ(proxy.CountEmbedsByLabel().size(), 0u);
	proxy.Rollback();

	EXPECT_EQ(proxy.CountEmbedsByLabel().at("a"), 1u);
	EXPECT_EQ(proxy.FindSaveEntryByPath("c:/x/a.jpg").has_value(), true);
}

TEST_CASE(Storage_EmbedStore_ReplaceLabels) {
	const fs::path dir = TempDir("replace");
	const auto img = [&](const char* name) { return (dir / "img" / name).string(); };
	{
		ohy::EmbedStore store(std::make_unique<ohy::FileStorageProxy>(dir / "db"));
		store.Add(img("a1.jpg"), "a", { 1.0f, 0.0f }, "tester");
		store.Add(img("a2.jpg"), "a", { 0.0f, 1.0f }, "tester");
		store.Add(img("b1.jpg"), "b", { 1.0f, 1.0f }, "tester");

		// a1 comes back (re-embedded), a2 goes, a3 is new. b is untouched.
		std::vector<ohy::NewEmbed> entries = {
			{ img("a1.jpg"), "a", { 2.0f, 0.0f } },
			{ img("a3.jpg"), "a", { 0.0f, 3.0f } },
		};
		const auto result = store.ReplaceLabels({ "a" }, std::move(entries), "tester");
		EXPECT_EQ(result.deleted, 2u);
		EXPECT_EQ(result.added, 2u);
	}

	ohy::EmbedStore store(std::make_unique<ohy::FileStorageProxy>(dir / "db"));
	const auto counts = store.CountByLabel();
	EXPECT_EQ(counts.at("a"), 2u);
	EXPECT_EQ(counts.at("b"), 1u);
	EXPECT_EQ(store.Contains(img("a1.jpg")), true);
	EXPECT_EQ(store.Contains(img("a2.jpg")), false);
	EXPECT_EQ(store.Contains(img("a3.jpg")), true);
	EXPECT_EQ(store.StoredLabel(img("b1.jpg")).value_or("?"), std::string("b"));
	EXPECT_EQ(store.StoredLabel(img("zz.jpg")).has_value(), false);
}

TEST_CASE(Storage_EmbedStore_ReplaceIsAllOrNothing) {
	const fs::path dir = TempDir("replace_conflict");
	const auto img = [&](const char* name) { return (dir / "img" / name).string(); };
	ohy::EmbedStore store(std::make_unique<ohy::FileStorageProxy>(dir / "db"));
	store.Add(img("a1.jpg"), "a", { 1.0f, 0.0f }, "tester");
	store.Add(img("b1.jpg"), "b", { 0.0f, 1.0f }, "tester");

	// b1.jpg is stored under "b", which this replace does not cover: the whole replace fails.
	std::vector<ohy::NewEmbed> entries = {
		{ img("a2.jpg"), "a", { 1.0f, 0.0f } },
		{ img("b1.jpg"), "a", { 0.0f, 1.0f } },
	};
	bool threw = false;
	try { store.ReplaceLabels({ "a" }, std::move(entries), "tester"); } catch (const std::exception&) { threw = true; }
	EXPECT_EQ(threw, true);

	EXPECT_EQ(store.Contains(img("a1.jpg")), true);
	EXPECT_EQ(store.Contains(img("a2.jpg")), false);
	EXPECT_EQ(store.StoredLabel(img("b1.jpg")).value_or("?"), std::string("b"));
	EXPECT_EQ(store.CountByLabel().at("a"), 1u);
}

TEST_CASE(Storage_EmbedStore_DeleteLabelAndAll) {
	const fs::path dir = TempDir("delete_store");
	const auto img = [&](const char* name) { return (dir / "img" / name).string(); };
	{
		ohy::EmbedStore store(std::make_unique<ohy::FileStorageProxy>(dir / "db"));
		store.Add(img("a1.jpg"), "a", { 1.0f, 0.0f }, "tester");
		store.Add(img("a2.jpg"), "a", { 0.0f, 1.0f }, "tester");
		store.Add(img("b1.jpg"), "b", { 1.0f, 1.0f }, "tester");
		store.Add(img("c1.jpg"), "c", { 1.0f, 2.0f }, "tester");

		EXPECT_EQ(store.DeleteLabel("a"), 2u);
		EXPECT_EQ(store.DeleteLabel("missing"), 0u);
		EXPECT_EQ(store.Contains(img("a1.jpg")), false);
		EXPECT_EQ(store.CountByLabel().size(), 2u);
	}
	{
		ohy::EmbedStore store(std::make_unique<ohy::FileStorageProxy>(dir / "db"));
		EXPECT_EQ(store.CountByLabel().size(), 2u);
		EXPECT_EQ(store.DeleteAll(), 2u);
		EXPECT_EQ(store.CountByLabel().size(), 0u);
		EXPECT_EQ(store.Contains(img("b1.jpg")), false);
	}

	ohy::EmbedStore store(std::make_unique<ohy::FileStorageProxy>(dir / "db"));
	EXPECT_EQ(store.LoadAll().size(), 0u);
	EXPECT_EQ(static_cast<int>(store.Add(img("b1.jpg"), "b", { 1.0f, 0.0f }, "tester")),
		static_cast<int>(ohy::AddStatus::Added));
}
