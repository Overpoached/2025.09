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
