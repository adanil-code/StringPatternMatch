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
// Cross-Platform Unicode Simple-Upcasing & Trie Engine Test & Benchmark Harness
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
// static, stateless, and locale-independent 1-to-1 simple-upcasing engine directly into the binary.
// Note: As a per-wchar_t 1-to-1 uppercasing table, multi-code-point foldings (e.g. 'ß' -> "SS"),
// Unicode normalization, and UTF-16 surrogate pairs on 16-bit wchar_t platforms are out of scope.
// 
// ARCHITECTURE OF THE LOOKUP TABLES (TWO-STAGE PAGE-TABLE TRIE & DELTA ENCODING):
// 1. The Full Unicode Space Challenge: A flat 32-bit array mapping every Unicode code point up to
//    U+10FFFF would consume ~4.25 MB of memory (and 16 GB if indexed across all 2^32 values),
//    blowing out CPU caches.
// 2. Level 1 Directory (L1_Directory): The Unicode space is chunked into blocks of 256 characters. 
//    The upper bits of a character (character >> 8) act as an index into a compact directory of 
//    8-bit block IDs (256 B on 16-bit wchar_t BMP builds, ~4.25 KB on 32-bit UTF-32 builds). 
//    Uncased scripts (like Hebrew, Arabic, CJK, and symbols) map to Block ID 0 (all zeroes), 
//    enabling a genuinely branchless O(1) lookup with no conditional skip.
// 3. Level 2 Delta Data Blocks (L2_Data_Deltas): For cased scripts, instead of storing absolute 
//    code points, the table stores deltas representing the numerical distance between lowercase 
//    and uppercase characters (16-bit unsigned wrap-around deltas on BMP wchar_t, ~11 KB total; 
//    32-bit signed deltas on UTF-32 wchar_t, ~33 KB total).
// 4. Total Footprint & Performance: Active script pages fit comfortably within the processor's 
//    high-speed L1/L2 cache and execute via a branchless two-stage lookup and modular ALU addition.
//--------------------------------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cwchar>
#include <locale>
#include <atomic>
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
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

// Core StringPatternMatch Engine
#include "string_pattern_match.h"

// trie_data_generator.py script generated Unicode Trie Data (Delta Encoded)
#include "unicode_trie_data.h"

// Guard against silent corruption of non-ASCII L"..." literals (e.g., MSVC without /utf-8)
static_assert(L'\x0130' == L'İ', "Source must be compiled as UTF-8 (pass /utf-8 on MSVC)");
static_assert((sizeof(wchar_t) == 2) == (WCHAR_MAX <= 0xFFFF),
              "Generated trie table width does not match sizeof(wchar_t)");

//--------------------------------------------------------------------------------------------------
// CustomUnicodeFunctor
//
// Semantics:
//   Functor consumed by CustomCasePolicy. Executes branchless two-stage trie lookup.
//   Templated on IsTurkish to completely eliminate runtime branch overhead for standard text.
//--------------------------------------------------------------------------------------------------
template <bool IsTurkish = false>
struct CustomUnicodeFunctor
{
    inline wchar_t operator()(wchar_t character) const noexcept
    {
        // 1. Cast to unsigned representation to handle Windows UTF-16 and Linux/Mac UTF-32 natively
        using UnsignedW = std::make_unsigned_t<wchar_t>;
        uint32_t uc = static_cast<UnsignedW>(character);

        // 2. Handle locale-specific anomalies (Turkish 'i' -> 'İ')
        //    Note: Turkish dotless 'ı' (U+0131) already maps to 'I' (U+0049) in UnicodeData.txt
        if constexpr (IsTurkish)
        {
            if (uc == static_cast<uint32_t>(L'i')) [[unlikely]]
            {
                return static_cast<wchar_t>(0x0130); // Latin Capital Letter I with Dot Above
            }
        }

        // 3. Reject out-of-bounds codepoints safely on UTF-32 platforms
        if constexpr (sizeof(wchar_t) > 2)
        {
            if (uc > 0x10FFFFu) [[unlikely]]
            {
                return character;
            }
        }

        // 4. Level 1: Fetch the Data Block ID (Shift right by 8 bits to divide by 256)
        uint8_t blockId = unicode_case_detail::L1_Directory[uc >> 8];

        // 5. Level 2: Genuinely branchless delta lookup (Block 0 is all zeroes for uncased scripts)
        //    Unsigned modular addition handles both 16-bit wrap-around and 32-bit two's-complement
        //    deltas cleanly with zero signed-overflow UB or narrowing warnings.
        auto delta = unicode_case_detail::L2_Data_Deltas[blockId][uc & 0xFFu];
        return static_cast<wchar_t>(static_cast<UnsignedW>(uc + static_cast<UnsignedW>(delta)));
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
    bool          registerPattern  = true;  // Flag to prevent duplicate/conflicting pattern registration
};

// Normalizes '/' path literals to the native OS separator ('\\' on Windows, '/' on Linux/macOS)
// so WildcardScope::PathSegment rules match StringPatternMatchDetail::IsPathSeparator
static std::wstring to_native_path(std::wstring path)
{
#ifdef _WIN32
    for (wchar_t& ch : path)
    {
        if (ch == L'/')
        {
            ch = L'\\';
        }
    }
#endif
    return path;
}

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
            testPassed = (matchedRule != nullptr && matchedRule->RuleId == tc.rule.RuleId);
        }

        // Output formatting
        std::wstring statusStr = testPassed ? L"[PASS]" : L"[FAIL]";
        std::wstring idStr     = L"#" + std::to_wstring(i + 1);
        std::wstring matchStr  = (hasMatched && matchedRule != nullptr)
            ? (L"ID: " + std::to_wstring(matchedRule->RuleId))
            : L"None";

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

// Zero-overhead compiler barrier to prevent loop-invariant hoisting under -O3/-flto
template <typename T>
inline void CompilerBarrier(const T*& ptr) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : "+r"(ptr) : : "memory");
#else
    (void)ptr;
    std::atomic_signal_fence(std::memory_order_acq_rel);
#endif
}

void RunBenchmarks(CrossPlatformUnicodeMatcher& matcher, 
                   CrossPlatformTurkishMatcher& matcherTR)
{
    constexpr size_t WARMUP_ITERATIONS    = 50'000;
    constexpr size_t BENCHMARK_ITERATIONS = 2'000'000;

    std::vector<BenchmarkTarget> targets = 
    {
        { L"Exact Match Lookup",      L"Hungarian",       L"SZÁMÍTÁSTECHNIKA ALAPJAI",                                       false },
        { L"Prefix/Suffix Wildcards", L"Greek",           to_native_path(L"/VAR/LOG/ΕΛΛΗΝΙΚΆ_SYSTEM.LOG"),                   false },
        { L"Segment Wildcard '?'",    L"Cyrillic",        to_native_path(L"/ДАННЫЕ/ПОЛЬЗОВАТЕЛЬ_5/ОТЧЕТ.DOC"),               false },
        { L"Segment Wildcard '?'",    L"Armenian",        to_native_path(L"/ԴԱՆՆԻԵ/Պոլզովատել_3/ՕՏՉԵՏ.DOC"),                 false },
        { L"Recursive Wildcard '**'", L"ASCII",           to_native_path(L"/backups/2026/server.tar.gz"),                    false },
        { L"Diacritics & Accents",    L"Latin Extended",  to_native_path(L"/docs/ESPAÑOL_FRANÇAIS_ÖSTERREICH/INFORME.TXT"),  false },
        { L"Branchless Identity (0)", L"Hebrew (Uncased)",to_native_path(L"/שלום/תיקייה/data.dat"),                          false },
        { L"Branchless Identity (0)", L"Arabic (Uncased)",to_native_path(L"/ملفات/النظام/config.cfg"),                       false },
        { L"Locale-Specific I/İ",     L"Turkish",         L"İSTANBUL_İLETİŞİM_SIRRI_2026.TXT",                               true  },
        { L"Negative Early Reject",   L"Cyrillic",        to_native_path(L"/ДОКУМЕНТЫ/ПОЛЬЗОВАТЕЛЬ_5/ВЛОЖЕННАЯ_ПАПКА/ОТЧЕТ"),false }
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
            const std::wstring* targetPtr = &bt.target;

            // 1. Warm-up Phase: Pre-fault cache lines and branch predictors
            for (size_t i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                CompilerBarrier(targetPtr);
                if (activeMatcher.Search(*targetPtr, matchedRule) && matchedRule != nullptr)
                {
                    payloadSink += matchedRule->RuleId;
                }
            }

            // 2. Timed Benchmark Loop (monotonic steady_clock avoids wall-clock jumps)
            auto startTime = std::chrono::steady_clock::now();

            for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                CompilerBarrier(targetPtr);
                if (activeMatcher.Search(*targetPtr, matchedRule) && matchedRule != nullptr)
                {
                    payloadSink += matchedRule->RuleId;
                }
            }

            auto endTime = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(endTime - startTime).count();
        };

        double durationMs = bt.useTurkishLocale ? executeBenchmark(matcherTR) : executeBenchmark(matcher);

        // 3. Compute Metrics (M chars/s reflects submitted input length throughput)
        double latencyNs    = (durationMs * 1'000'000.0) / BENCHMARK_ITERATIONS;
        double megaOpsSec   = (BENCHMARK_ITERATIONS / (durationMs / 1000.0)) / 1'000'000.0;
        double megaCharsSec = megaOpsSec * static_cast<double>(bt.target.length());

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
    // Negative niceness requires CAP_SYS_NICE/root; ignore EACCES on unprivileged runs
    (void)setpriority(PRIO_PROCESS, 0, -10);
#endif
}

static void configure_wide_console()
{
#ifdef _WIN32
    // Switch the Windows console to UTF-16 mode to render international characters correctly
    // Required to prevent std::wcout from aborting output upon hitting untranslatable OEM codepage sequences
    _setmode(_fileno(stdout), _O_U16TEXT);
#else
    // Imbue only the ctype/codecvt facet on std::wcout so non-ASCII failure logs render cleanly
    // without mutating the global C/C++ locale or altering classic "C" numeric table formatting
    try
    {
        std::wcout.imbue(std::locale(std::locale::classic(), std::locale(""), std::locale::ctype));
    }
    catch (const std::runtime_error&)
    {
        try
        {
            std::wcout.imbue(std::locale(std::locale::classic(), std::locale("C.UTF-8"), std::locale::ctype));
        }
        catch (const std::runtime_error&)
        {
            // Keep classic locale if no UTF-8 locale is installed
        }
    }
#endif
}

int main()
{
    configure_wide_console();

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
            to_native_path(L"/*/ελληνικά_*.log"),
            WildcardScope::Default,
            EngineRule{ 101, L"Greek", L"Localized Greek logs" },
            to_native_path(L"/VAR/LOG/ΕΛΛΗΝΙΚΆ_SYSTEM.LOG"),
            true,
            false
        },
        {
            L"Armenian Path Segment",
            to_native_path(L"/Դաննիե/Պոլզովատել_?/օտչետ.doc"),
            WildcardScope::PathSegment,
            EngineRule{ 102, L"Armenian", L"Path segment boundary check" },
            to_native_path(L"/ԴԱՆՆԻԵ/Պոլզովատել_3/ՕՏՉԵՏ.DOC"),
            true,
            false
        },
        {
            L"Cyrillic Path Segment",
            to_native_path(L"/Данные/Пользователь_?/отчет.doc"),
            WildcardScope::PathSegment,
            EngineRule{ 103, L"Cyrillic", L"Path segment boundary check" },
            to_native_path(L"/ДАННЫЕ/ПОЛЬЗОВАТЕЛЬ_5/отчет.DOC"),
            true,
            false
        },
        {
            L"English ASCII Archive",
            to_native_path(L"**/*.TAR.GZ"),
            WildcardScope::Default,
            EngineRule{ 104, L"English/ASCII", L"Archive extension filter" },
            to_native_path(L"/backups/2026/server.tar.gz"),
            true,
            false
        },
        {
            L"Latin Extended Accents",
            to_native_path(L"/docs/Español_Français_österreich/*.txt"),
            WildcardScope::Default,
            EngineRule{ 105, L"Latin Extended", L"Accented Western European text" },
            to_native_path(L"/DOCS/ESPAÑOL_FRANÇAIS_ÖSTERREICH/INFORME.TXT"),
            true,
            false
        },
        {
            L"Hebrew (Uncased)",
            to_native_path(L"/שלום/תיקייה/*.dat"),
            WildcardScope::Default,
            EngineRule{ 106, L"Hebrew", L"Uncased identity mapping validation" },
            to_native_path(L"/שלום/תיקייה/data.dat"),
            true,
            false
        },
        {
            L"Arabic (Uncased)",
            to_native_path(L"/ملفات/النظام/*.cfg"),
            WildcardScope::Default,
            EngineRule{ 107, L"Arabic", L"Uncased identity mapping validation" },
            to_native_path(L"/ملفات/النظام/config.cfg"),
            true,
            false
        },        
        {
            L"Turkish Dotted/Dotless I (Lower)",
            L"istanbul_iletişim_sırrı_*.txt",
            WildcardScope::Default,
            EngineRule{ 200, L"Turkish", L"Validates lowercase Turkish i in ASCII SWAR prefix" },
            L"İSTANBUL_İLETİŞİM_SIRRI_2026.TXT",
            true,
            true // Routes to matcherTR
        },
        {
            L"Turkish Dotted/Dotless I",
            L"istanbul_iletişim_sırrı_*.txt",
            WildcardScope::Default,
            EngineRule{ 200, L"Turkish", L"Validates dotted i (İ) and dotless ı (I)" },
            L"İstanbul_iletişim_sırrı_2026.TXT",
            true,
            true,  // Routes to matcherTR
            false  // Pattern already registered by previous test case
        },
        {
            L"Negative: Boundary Violation",
            to_native_path(L"/Данные/Пользователь_?/отчет.doc"),
            WildcardScope::PathSegment,
            EngineRule{ 103, L"Cyrillic", L"Should fail across subfolders" },
            to_native_path(L"/ДАННЫЕ/ПОЛЬЗОВАТЕЛЬ_5/ВЛОЖЕННАЯ_ПАПКА/ОТЧЕТ.DOC"),
            false,
            false,
            false  // Evaluates against Test #4's registered pattern without overwriting RuleId 103
        }
    };

    // Register patterns into respective engines (making a copy to satisfy rvalue reference movement)
    for (const auto& tc : testCases)
    {
        if (!tc.registerPattern)
        {
            continue;
        }

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