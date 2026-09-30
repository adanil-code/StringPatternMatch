/*
* String Pattern Match Engine
* Copyright 2026 Alexander Danileiko
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at:
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
* OF ANY KIND, either express or implied.
*/

//--------------------------------------------------------------------------------------------------
// Cross-Platform Unicode Case-Folding & Trie Engine Test & Benchmark Harness
//--------------------------------------------------------------------------------------------------
// 
// THE LOCALIZATION PROBLEM WITH STANDARD RUNTIMES:
// Standard C++ runtime functions like std::towupper() are inherently locale-aware. They depend
// entirely on the active C-runtime locale state (e.g., setlocale(LC_ALL, "")). On systems configured 
// with a US-EN locale, or inside minimal container environments lacking international locale packs, 
// std::towupper() will fail to correctly transform non-English text (such as Greek, Cyrillic, 
// or Hungarian), resulting in silent case-folding mismatches or untransformed strings.
// 
// HOW THIS SAMPLE BYPASSES THE LIMITATION:
// This test harness demonstrates how to bypass OS and runtime locale dependencies completely. 
// By pre-parsing the official Unicode specification (UnicodeData.txt) at build time, it embeds a 
// static, stateless, and locale-independent case-folding engine directly into the binary.
// 
// ARCHITECTURE OF THE LOOKUP TABLES (TWO-STAGE PAGE-TABLE TRIE & DELTA ENCODING):
// 1. The Full Unicode Space Challenge: A flat array mapping every 32-bit code point up to U+10FFFF 
//    would consume 16 GB of memory, making brute-force tables impossible.
// 2. Level 1 Directory (L1_Directory): The Unicode space is chunked into blocks of 256 characters. 
//    The upper bits of a character (character >> 8) act as an index into an 8.5 KB directory of 
//    16-bit block IDs. Uncased scripts (like Hebrew, Arabic, CJK, and symbols) map to Block ID 0, 
//    enabling a zero-overhead, branchless O(1) fast-path.
// 3. Level 2 Delta Data Blocks (L2_Data_Deltas): For cased scripts, instead of storing absolute 
//    32-bit code points, the table stores signed 32-bit integers (`int32_t`) representing the 
//    numerical distance (delta) between lowercase and uppercase characters. This compresses the 
//    active L2 memory footprint down to ~29 KB.
// 4. Total Footprint & Performance: The combined trie structure takes ~37.5 KB, fitting entirely
//    within the processor's high-speed L1/L2 cache and executing via a single branchless ALU 
//    addition.
//--------------------------------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <type_traits>
#include <iomanip>
#include <chrono>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#elif defined(__linux__)
#include <sys/resource.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/resource.h>
#endif

// Core StringPatternMatch Engine
#include "string_pattern_match.h"

// trie_data_generator.py script generated Unicode Trie Data (32-bit Delta Encoded)
#include "unicode_trie_data.h"

//--------------------------------------------------------------------------------------------------
// CustomUnicodeFunctor
//
// Semantics:
//   Functor consumed by CustomCasePolicy. Executes two-stage trie lookup.
//   Templated on IsTurkish to completely eliminate runtime branch overhead for standard text.
//--------------------------------------------------------------------------------------------------
template <bool IsTurkish = false>
struct CustomUnicodeFunctor
{
    inline wchar_t operator()(wchar_t character) const noexcept
    {
        // 1. Handle locale-specific anomalies (Turkish 'i' -> 'İ' and 'ı' -> 'I')
        if constexpr (IsTurkish)
        {
            if (character == L'i')
            {
                return L'\x0130'; // Latin Capital Letter I with Dot Above
            }

            if (character == L'\x0131')
            {
                return L'I';      // Latin Capital Letter I
            }
        }

        // 2. Cast to unsigned 32-bit to handle Linux/Mac UTF-32 wchar_t natively
        uint32_t uc = static_cast<std::make_unsigned_t<wchar_t>>(character);

        // 3. Reject out-of-bounds codepoints safely
        if (uc > 0x10FFFF) [[unlikely]]
        {
            return character;
        }

        // 4. Level 1: Fetch the Data Block ID (Shift right by 8 bits to divide by 256)
        uint16_t blockId = L1_Directory[uc >> 8];

        // 5. Branchless fast-path: Predicts true for symbols, spaces, numbers, and uncased scripts
        if (blockId == 0)
        {
            return character;
        }

        // 6. Level 2: Fetch the delta via direct block pointer to avoid index multiplication
        const int32_t* block = L2_Data_Deltas[blockId];
        return character + block[uc & 0xFF];
    }
};

//--------------------------------------------------------------------------------------------------
// Test Infrastructure Structures
//--------------------------------------------------------------------------------------------------
struct EngineRule
{
    uint32_t     RuleId;      // Unique numeric identifier for the rule
    std::wstring Language;    // Target language or script category
    std::wstring Description; // Human-readable description of the match rule
};

// Default Unicode Locale Matcher (IsTurkish = false)
using CrossPlatformUnicodeMatcher = StringPatternMatch<wchar_t, 
                                                       EngineRule, 
                                                       CustomCasePolicy<wchar_t, CustomUnicodeFunctor<false>>>;

// Specialized Turkish Locale Override Matcher (IsTurkish = true)
using CrossPlatformTurkishMatcher = StringPatternMatch<wchar_t, 
                                                       EngineRule, 
                                                       CustomCasePolicy<wchar_t, CustomUnicodeFunctor<true>>>;

// Data-driven test case structure to eliminate repetitive match logic
struct TestCase
{
    std::wstring  testName;                 // Descriptive name for the test scenario
    std::wstring  pattern;                  // Search pattern string containing literal text or wildcards
    WildcardScope scope;                    // Wildcard matching scope (e.g., Default or PathSegment)
    EngineRule    rule;                     // Rule metadata payload attached to the pattern
    std::wstring  target;                   // Target string evaluated against the pattern
    bool          expectMatch;              // Expected outcome of the match evaluation
    bool          useTurkishLocale = false; // Flag to route through Turkish engine
};

//--------------------------------------------------------------------------------------------------
// Functional Verification Suite
//--------------------------------------------------------------------------------------------------
void RunFunctionalTests(CrossPlatformUnicodeMatcher& matcher, 
                        CrossPlatformTurkishMatcher& matcherTR, 
                        const std::vector<TestCase>& testCases)
{
    std::wcout << L"================================================================================\n";
    std::wcout << L"                    FUNCTIONAL MULTI-LANGUAGE TEST SUITE                        \n";
    std::wcout << L"================================================================================\n\n";

    // Print Table Header
    std::wcout << std::left 
               << std::setw(8)  << L"Status" 
               << std::setw(6)  << L"ID" 
               << std::setw(34) << L"Test Scenario" 
               << std::setw(15) << L"Rule Match" << L"\n";
    std::wcout << L"--------------------------------------------------------------------------------\n";

    // Execute test suite loop with cleanly aligned columns
    size_t passedCount = 0;
    for (size_t i = 0; i < testCases.size(); ++i)
    {
        const auto& tc = testCases[i];
        EngineRule* matchedRule = nullptr;
        
        bool hasMatched = tc.useTurkishLocale ? 
            matcherTR.Search(tc.target, matchedRule) : 
            matcher.Search(tc.target, matchedRule);

        bool testPassed = (hasMatched == tc.expectMatch);
        if (hasMatched && tc.expectMatch)
        {
            // Verify correct rule mapping payload was retrieved
            testPassed = (matchedRule->RuleId == tc.rule.RuleId);
        }

        // Output formatting
        std::wstring statusStr = testPassed ? L"[PASS]" : L"[FAIL]";
        std::wstring idStr     = L"#" + std::to_wstring(i + 1);
        std::wstring matchStr  = hasMatched ? (L"ID: " + std::to_wstring(matchedRule->RuleId)) : L"None";

        std::wcout << std::left 
                   << std::setw(8)  << statusStr 
                   << std::setw(6)  << idStr 
                   << std::setw(34) << tc.testName 
                   << std::setw(15) << matchStr << L"\n";

        if (testPassed)
        {
            passedCount++;
        }
        else
        {
            // Indented error details if a test fails
            std::wcout << L"       -> Target:         " << tc.target << L"\n"
                       << L"       -> Expected Match: " << (tc.expectMatch ? L"YES" : L"NO") 
                       << L", Got: " << (hasMatched ? L"YES" : L"NO") << L"\n";
        }
    }

    std::wcout << L"--------------------------------------------------------------------------------\n";
    std::wcout << L"Test Execution Summary: " << passedCount << L" / " << testCases.size() << L" Tests Passed.\n\n";
}

//--------------------------------------------------------------------------------------------------
// High-Throughput Benchmarking Suite
//--------------------------------------------------------------------------------------------------
struct BenchmarkTarget
{
    std::wstring name;             // Descriptive benchmark scenario label
    std::wstring script;           // Unicode script or character family under test
    std::wstring target;           // Sample string processed during the benchmark
    bool         useTurkishLocale; // Flag indicating whether to route to the Turkish matcher
};

void RunBenchmarks(CrossPlatformUnicodeMatcher& matcher, 
                   CrossPlatformTurkishMatcher& matcherTR)
{
    constexpr size_t WARMUP_ITERATIONS    = 50'000;
    constexpr size_t BENCHMARK_ITERATIONS = 2'000'000;

    std::vector<BenchmarkTarget> targets = 
    {
        { L"Exact Match Lookup",      L"Hungarian",       L"SZÁMÍTÁSTECHNIKA ALAPJAI",                       false },
        { L"Prefix/Suffix Wildcards", L"Greek",           L"/VAR/LOG/ΕΛΛΗΝΙΚΆ_SYSTEM.LOG",                   false },
        { L"Segment Wildcard '?'",    L"Cyrillic",        L"/ДАННЫЕ/ПОЛЬЗОВАТЕЛЬ_5/ОТЧЕТ.DOC",               false },
        { L"Segment Wildcard '?'",    L"Armenian",        L"/ԴԱՆՆԻԵ/Պոլզովատել_3/ՕՏՉԵՏ.DOC",                 false },
        { L"Recursive Wildcard '**'", L"ASCII",           L"/backups/2026/server.tar.gz",                    false },
        { L"Diacritics & Accents",    L"Latin Extended",  L"/DOCS/ESPAÑOL_FRANÇAIS_ÖSTERREICH/INFORME.TXT",  false },
        { L"Branchless Identity (0)", L"Hebrew (Uncased)",L"/שלום/תיקייה/data.dat",                          false },
        { L"Branchless Identity (0)", L"Arabic (Uncased)",L"/ملفات/النظام/config.cfg",                       false },
        { L"Locale-Specific I/İ",     L"Turkish",         L"İSTANBUL_İLETİŞİM_SIRRI_2026.TXT",               true  },
        { L"Negative Early Reject",   L"Cyrillic",        L"/ДОКУМЕНТЫ/ПОЛЬЗОВАТЕЛЬ_5/ВЛОЖЕННАЯ_ПАПКА/ОТЧЕТ",false }
    };

    std::wcout << L"================================================================================\n";
    std::wcout << L"            THROUGHPUT BENCHMARK (2,000,000 SEARCH ITERATIONS EACH)           \n";
    std::wcout << L"================================================================================\n\n";

    std::wcout << std::left 
               << std::setw(26) << L"Benchmark Target"
               << std::setw(18) << L"Script"
               << std::right
               << std::setw(12) << L"Latency (ns)"
               << std::setw(12) << L"M ops/sec"
               << std::setw(12) << L"M chars/s" << L"\n";
    std::wcout << L"--------------------------------------------------------------------------------\n";

    // Non-volatile accumulator sink to prevent dead-code elimination by -O3/-flto
    uint64_t payloadSink = 0;

    for (const auto& bt : targets)
    {
        EngineRule* matchedRule = nullptr;

        // Lambda executor to benchmark either matcher without ternary type mismatch
        auto executeBenchmark = [&](auto& activeMatcher) -> double
        {
            // 1. Warm-up Phase: Pre-fault cache lines and branch predictors
            for (size_t i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                if (activeMatcher.Search(bt.target, matchedRule))
                {
                    payloadSink += matchedRule->RuleId;
                }
            }

            // 2. Timed Benchmark Loop
            auto startTime = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                if (activeMatcher.Search(bt.target, matchedRule))
                {
                    payloadSink += matchedRule->RuleId;
                }
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(endTime - startTime).count();
        };

        double durationMs = bt.useTurkishLocale ? executeBenchmark(matcherTR) : executeBenchmark(matcher);

        // 3. Compute Metrics
        double latencyNs    = (durationMs * 1'000'000.0) / BENCHMARK_ITERATIONS;
        double megaOpsSec   = (BENCHMARK_ITERATIONS / (durationMs / 1000.0)) / 1'000'000.0;
        double megaCharsSec = megaOpsSec * bt.target.length();

        std::wcout << std::left 
                   << std::setw(26) << bt.name
                   << std::setw(18) << bt.script
                   << std::right    << std::fixed << std::setprecision(2)
                   << std::setw(12) << latencyNs
                   << std::setw(12) << megaOpsSec
                   << std::setw(12) << megaCharsSec << L"\n";
    }

    std::wcout << L"--------------------------------------------------------------------------------\n";
    std::wcout << L"Optimization Checksum (Sink Verification): 0x" 
               << std::hex << payloadSink << std::dec << L"\n";
    std::wcout << L"================================================================================\n";
}

static void set_high_priority()
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#else
    setpriority(PRIO_PROCESS, 0, -10);
#endif
}

int main()
{
#ifdef _WIN32
    // Switch the Windows console to UTF-16 mode to render international characters correctly
    // Required to prevent std::wcout from aborting output upon hitting untranslatable OEM codepage sequences
    _setmode(_fileno(stdout), _O_U16TEXT);
#endif

    set_high_priority();

    // Engine 1: Default Unicode Locale Matcher
    CrossPlatformUnicodeMatcher matcher(OptimizationMode::Auto, 
                                        CustomCasePolicy<wchar_t, CustomUnicodeFunctor<false>>());

    // Engine 2: Turkish Locale Override Matcher
    CrossPlatformTurkishMatcher matcherTR(OptimizationMode::Auto, 
                                          CustomCasePolicy<wchar_t, CustomUnicodeFunctor<true>>());

    // Unified test cases including standard international tests and Turkish-specific overrides
    std::vector<TestCase> testCases = 
    {
        {
            L"Hungarian Exact Match",
            L"Számítástechnika alapjai",
            WildcardScope::Default,
            EngineRule{ 100, L"Hungarian", L"Exact match check" },
            L"SZÁMÍTÁSTECHNIKA Alapjai",
            true,
            false
        },
        {
            L"Greek Wildcard Log",
            L"/*/ελληνικά_*.log",
            WildcardScope::Default,
            EngineRule{ 101, L"Greek", L"Localized Greek logs" },
            L"/VAR/LOG/ΕΛΛΗΝΙΚΆ_SYSTEM.LOG",
            true,
            false
        },
        {
            L"Armenian Path Segment",
            L"/Դաննիե/Պոլզովատել_?/օտչետ.doc",
            WildcardScope::PathSegment,
            EngineRule{ 102, L"Armenian", L"Path segment boundary check" },
            L"/ԴԱՆՆԻԵ/Պոլզովատել_3/ՕՏՉԵՏ.DOC",
            true,
            false
        },
        {
            L"Cyrillic Path Segment",
            L"/Данные/Пользователь_?/отчет.doc",
            WildcardScope::PathSegment,
            EngineRule{ 103, L"Cyrillic", L"Path segment boundary check" },
            L"/ДАННЫЕ/ПОЛЬЗОВАТЕЛЬ_5/ОТЧЕТ.DOC",
            true,
            false
        },
        {
            L"English ASCII Archive",
            L"**/*.TAR.GZ",
            WildcardScope::Default,
            EngineRule{ 104, L"English/ASCII", L"Archive extension filter" },
            L"/backups/2026/server.tar.gz",
            true,
            false
        },
        {
            L"Latin Extended Accents",
            L"/docs/español_français_österreich/*.txt",
            WildcardScope::Default,
            EngineRule{ 105, L"Latin Extended", L"Accented Western European text" },
            L"/DOCS/ESPAÑOL_FRANÇAIS_ÖSTERREICH/INFORME.TXT",
            true,
            false
        },
        {
            L"Hebrew (Uncased)",
            L"/שלום/תיקייה/*.dat",
            WildcardScope::Default,
            EngineRule{ 106, L"Hebrew", L"Uncased identity mapping validation" },
            L"/שלום/תיקייה/data.dat",
            true,
            false
        },
        {
            L"Arabic (Uncased)",
            L"/ملفات/النظام/*.cfg",
            WildcardScope::Default,
            EngineRule{ 107, L"Arabic", L"Uncased identity mapping validation" },
            L"/ملفات/النظام/config.cfg",
            true,
            false
        },
        {
            L"Turkish Dotted/Dotless I",
            L"istanbul_iletişim_sırrı_*.txt",
            WildcardScope::Default,
            EngineRule{ 200, L"Turkish", L"Validates dotted i (İ) and dotless ı (I)" },
            L"İSTANBUL_İLETİŞİM_SIRRI_2026.TXT",
            true,
            true // Routes to matcherTR
        },
        {
            L"Negative: Boundary Violation",
            L"/Данные/Пользователь_?/отчет.doc",
            WildcardScope::PathSegment,
            EngineRule{ 108, L"Cyrillic", L"Should fail across subfolders" },
            L"/ДАННЫЕ/ПОЛЬЗОВАТЕЛЬ_5/ВЛОЖЕННАЯ_ПАПКА/ОТЧЕТ.DOC",
            false,
            false
        }
    };

    // Register patterns into respective engines (making a copy to satisfy rvalue reference movement)
    for (const auto& tc : testCases)
    {
        EngineRule ruleCopy = tc.rule;
        if (tc.useTurkishLocale)
        {
            matcherTR.AddPattern(tc.pattern, tc.scope, std::move(ruleCopy));
        }
        else
        {
            matcher.AddPattern(tc.pattern, tc.scope, std::move(ruleCopy));
        }
    }

    // Run Verification First
    RunFunctionalTests(matcher, matcherTR, testCases);    

    // Run Benchmarks
    RunBenchmarks(matcher, matcherTR);

    return 0;
}