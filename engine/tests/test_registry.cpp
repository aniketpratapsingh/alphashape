// Tests for the task registry and the single execution path.

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/tasks/registry.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {
namespace {

/// A minimal task used to exercise the registry without a real algorithm.
class StubTask : public AbstractTask {
public:
    explicit StubTask(std::string task_name, TargetType target = TargetType::Chunk)
        : name_(std::move(task_name)), target_(target) {}

    std::string name() const override { return name_; }

    void declare_params(Params& params) const override {
        params.declare(ParamSpec{"keypoint_limit", ParamType::Int, std::int64_t{40000}, "limit"});
        params.declare(ParamSpec{"enabled", ParamType::Bool, true, "flag"});
    }

    TargetType target_type() const override { return target_; }

    void apply(Document& /*document*/, Params& /*params*/, const Progress& /*progress*/) override {
        ++apply_calls;
    }

    int apply_calls = 0;

private:
    std::string name_;
    TargetType target_;
};

Document document_with_active_chunk() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("chunk");
    document.set_active_chunk_id(chunk.id());
    return document;
}

TEST(Registry, RegisterThenFindAndRequire) {
    TaskRegistry registry;
    registry.register_task(std::make_shared<StubTask>("Stub"));
    ASSERT_NE(registry.find("Stub"), nullptr);
    EXPECT_EQ(registry.require("Stub").task->name(), "Stub");
    EXPECT_EQ(registry.size(), 1U);
    EXPECT_EQ(registry.names(), std::vector<std::string>{"Stub"});
}

TEST(Registry, UnknownNameThrowsUnsupportedTaskError) {
    TaskRegistry registry;
    const std::string message = [&] {
        try {
            (void)registry.require("Ghost");
        } catch (const UnsupportedTaskError& e) {
            return std::string(e.what());
        }
        return std::string("<no exception>");
    }();
    EXPECT_NE(message.find("Unsupported task: Ghost"), std::string::npos) << message;
    EXPECT_EQ(registry.find("Ghost"), nullptr);
}

TEST(Registry, DuplicateRegistrationThrows) {
    TaskRegistry registry;
    registry.register_task(std::make_shared<StubTask>("Stub"));
    EXPECT_THROW(registry.register_task(std::make_shared<StubTask>("Stub")), Error);
}

TEST(Registry, SchemaPopulatedAtRegistration) {
    TaskRegistry registry;
    registry.register_task(std::make_shared<StubTask>("Stub"));
    const Params& schema = registry.require("Stub").schema;
    EXPECT_TRUE(schema.declares("keypoint_limit"));
    EXPECT_TRUE(schema.declares("enabled"));
    EXPECT_EQ(schema.get_int("keypoint_limit"), 40000);
}

TEST(Registry, DisabledEntryIsFindableListedAndRefusesToRun) {
    TaskRegistry registry;
    auto task = std::make_shared<StubTask>("Disabled");
    StubTask* raw = task.get();
    registry.register_task(task, /*enabled=*/false);

    ASSERT_NE(registry.find("Disabled"), nullptr);
    EXPECT_FALSE(registry.find("Disabled")->enabled);
    EXPECT_EQ(registry.names(), std::vector<std::string>{"Disabled"});

    Document document = document_with_active_chunk();
    Params params = make_params(registry.require("Disabled"));
    EXPECT_THROW(run_task(registry, document, "Disabled", params), Error);
    EXPECT_EQ(raw->apply_calls, 0);
}

TEST(Registry, ChunkTaskWithNoActiveChunkThrows) {
    TaskRegistry registry;
    registry.register_task(std::make_shared<StubTask>("Stub"));
    Document document;  // no chunks, active id is -1
    Params params = make_params(registry.require("Stub"));
    EXPECT_THROW(run_task(registry, document, "Stub", params), Error);
}

TEST(Registry, SuccessfulRunRecordsEveryDeclaredParamIncludingDefaults) {
    TaskRegistry registry;
    auto task = std::make_shared<StubTask>("Stub");
    StubTask* raw = task.get();
    registry.register_task(task);

    Document document = document_with_active_chunk();
    Params params = make_params(registry.require("Stub"));
    params.set("keypoint_limit", std::int64_t{50});  // one explicit, one default
    run_task(registry, document, "Stub", params);

    EXPECT_EQ(raw->apply_calls, 1);
    const Meta& meta = document.active_chunk()->meta();
    const std::string* limit = meta.find("Stub/keypoint_limit");
    ASSERT_NE(limit, nullptr);
    EXPECT_EQ(*limit, "50");
    const std::string* enabled = meta.find("Stub/enabled");  // default not touched by caller
    ASSERT_NE(enabled, nullptr);
    EXPECT_EQ(*enabled, "true");
}

TEST(Registry, MissingTaskNameThrowsUnsupportedTaskError) {
    TaskRegistry registry;
    Document document = document_with_active_chunk();
    Params params;  // empty
    EXPECT_THROW(run_task(registry, document, "Nope", params), UnsupportedTaskError);
}

}  // namespace
}  // namespace alphashape
