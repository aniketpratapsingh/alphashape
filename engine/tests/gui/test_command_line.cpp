#include <gtest/gtest.h>

#include "apps/gui/command_line.hpp"

using alphashape::LogLevel;
using alphashape::gui::Options;
using alphashape::gui::ParseResult;
using alphashape::gui::ThemeMode;
using alphashape::gui::parse_command_line;

TEST(CommandLine, EmptyIsPlainRun) {
    const ParseResult r = parse_command_line({});
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.options.action, Options::Action::Run);
    EXPECT_FALSE(r.options.new_project);
    EXPECT_FALSE(r.options.open_path.has_value());
    EXPECT_FALSE(r.options.theme.has_value());
    EXPECT_FALSE(r.options.offscreen_smoke);
}

TEST(CommandLine, ParsesEveryOption) {
    const ParseResult r = parse_command_line({"--theme", "light", "--reset-settings", "--log-level=debug",
                                              "--offscreen-smoke", "--open", "a.psx"});
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(r.options.theme, ThemeMode::Light);
    EXPECT_TRUE(r.options.reset_settings);
    EXPECT_EQ(r.options.log_level, LogLevel::Debug);
    EXPECT_TRUE(r.options.offscreen_smoke);
    EXPECT_EQ(r.options.open_path, "a.psx");
}

TEST(CommandLine, NewTakesAnOptionalPath) {
    ParseResult bare = parse_command_line({"--new"});
    ASSERT_TRUE(bare.ok());
    EXPECT_TRUE(bare.options.new_project);
    EXPECT_TRUE(bare.options.new_path.empty());

    ParseResult before_option = parse_command_line({"--new", "--theme", "dark"});
    ASSERT_TRUE(before_option.ok());
    EXPECT_TRUE(before_option.options.new_path.empty());
    EXPECT_EQ(before_option.options.theme, ThemeMode::Dark);

    ParseResult with_path = parse_command_line({"--new", "site.psx"});
    ASSERT_TRUE(with_path.ok());
    EXPECT_EQ(with_path.options.new_path, "site.psx");
}

TEST(CommandLine, PositionalArgumentOpens) {
    const ParseResult r = parse_command_line({"site.psx"});
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.options.open_path, "site.psx");
}

TEST(CommandLine, HelpAndVersion) {
    EXPECT_EQ(parse_command_line({"--help"}).options.action, Options::Action::Help);
    EXPECT_EQ(parse_command_line({"--version"}).options.action, Options::Action::Version);
}

TEST(CommandLine, RejectsBadUsage) {
    const std::vector<std::vector<std::string>> bad = {
        {"--bogus"},           {"--theme"},          {"--theme", "purple"},
        {"--log-level", "x"},  {"--open"},           {"--new", "a", "--open", "b"},
        {"a.psx", "b.psx"},    {"--new", "--new"},   {"--reset-settings=1"},
        {"--offscreen-smoke=1"},
    };
    for (const auto& args : bad) {
        EXPECT_FALSE(parse_command_line(args).ok()) << "accepted: " << args[0];
    }
}
