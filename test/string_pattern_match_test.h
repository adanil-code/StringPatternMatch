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

// -------------------------------------------------------------------------------------
// Test Logic for StringPatternMatch
// -------------------------------------------------------------------------------------
#pragma once

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <atomic>
#include <thread>
#include <vector>
#include <chrono>
#include <string>
#include <cctype>
#include <cstdio>
#include <memory>
#include <algorithm>
#include <cstring>
#include "string_pattern_match.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#ifndef KAFFINITY
using KAFFINITY = ULONG_PTR;
#endif
#elif defined(__linux__)
#include <sched.h>
#include <pthread.h>
#include <fstream>
#include <map>
#include <sys/resource.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/resource.h>
#endif

#ifndef COMPILER_BARRIER
#if defined(_MSC_VER)
#include <intrin.h>
#define COMPILER_BARRIER() _ReadWriteBarrier()
#else
#define COMPILER_BARRIER() std::atomic_signal_fence(std::memory_order_seq_cst)
#endif
#endif

extern std::atomic<int> g_abortTests;

#define LOG_INFO(...) std::printf(__VA_ARGS__)
#define LOG_ERR(...) std::printf(__VA_ARGS__)

constexpr uint32_t MaxTestThreads = 64;

// Configuration structure allowing callers to selectively run modules
enum class StringTestType
{
    Both = 0,
    Char,
    WChar
};

struct TestSuiteConfig
{
    bool             runCorrectness; // Flag to execute the functional correctness test suite
    bool             runPerformance; // Flag to execute the high-throughput performance test suite
    StringTestType   stringType;     // Specifies which character types to evaluate
    OptimizationMode optMode;        // Optimization mode to apply to the engine
};

extern TestSuiteConfig g_testConfig;

// Context tracking state for individual test worker threads
struct TestWorkerContext
{
    uint32_t           threadId;    // Unique identifier assigned to the worker thread
    std::atomic<bool>* startEvent;  // Event used to synchronize thread startup
    std::atomic<bool>* stopFlag;    // Shared flag signaling all threads to safely terminate
    void*              userContext; // Opaque pointer to user-defined test state data
};

using TestWorkerFunc = void (*)(TestWorkerContext* context);

// Manager handling thread pooling and synchronization
struct TestThreadManager
{
    std::vector<std::thread> threads;                  // Array of thread object handles
    TestWorkerContext        contexts[MaxTestThreads]; // Pre-allocated contexts for each worker
    uint32_t                 threadCount;              // Total number of actively running threads
    std::atomic<bool>        startEvent{false};        // Master event to unblock all workers simultaneously
    std::atomic<bool>        stopFlag{false};          // Master termination signal flag for the pool
};

inline void FlushLogToFile()
{
    std::fflush(stdout);
}

// -------------------------------------------------------------------------------------
// String Literal Mapping Utility for Cross-Type Templates
// -------------------------------------------------------------------------------------
template <SupportedChar CharT>
struct StrLiteral;

template <>
struct StrLiteral<char>
{
    static constexpr const char* get(const char*    c,
                                     const wchar_t* w)
    {
        (void)w;
        return c;
    }

    static constexpr char get_char(char    c,
                                   wchar_t w)
    {
        (void)w;
        return c;
    }
};

template <>
struct StrLiteral<wchar_t>
{
    static constexpr const wchar_t* get(const char*    c,
                                        const wchar_t* w)
    {
        (void)c;
        return w;
    }

    static constexpr wchar_t get_char(char    c,
                                      wchar_t w)
    {
        (void)c;
        return w;
    }
};

// -------------------------------------------------------------------------------------
// Returns CPU name string for logging and debugging purposes.
// -------------------------------------------------------------------------------------
inline std::string cpu_name()
{
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    int cpuInfo[4] = { 0 };
    char brand[0x40];
    std::memset(brand, 0, sizeof(brand));

#if defined(_MSC_VER)
    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = cpuInfo[0];

    if (nExIds >= 0x80000004)
    {
        __cpuid(cpuInfo, 0x80000002);
        std::memcpy(brand, cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000003);
        std::memcpy(brand + 16, cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000004);
        std::memcpy(brand + 32, cpuInfo, sizeof(cpuInfo));
    }
#else
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;

    __asm__ __volatile__("cpuid" : "=a"(eax) : "a"(0x80000000) : "ebx", "ecx", "edx");

    if (eax >= 0x80000004)
    {
        unsigned int data[12];

        for (unsigned int i = 0; i < 3; ++i)
        {
            __asm__ __volatile__("cpuid"
                : "=a"(data[i * 4 + 0]),
                "=b"(data[i * 4 + 1]),
                "=c"(data[i * 4 + 2]),
                "=d"(data[i * 4 + 3])
                : "a"(0x80000002 + i));
        }

        std::memcpy(brand, data, sizeof(data));
    }
#endif
    return std::string(brand);
#elif defined(__APPLE__)
    char buf[256];
    size_t size = sizeof(buf);

    if (sysctlbyname("machdep.cpu.brand_string", &buf, &size, NULL, 0) == 0)
    {
        return std::string(buf);
    }

    size = sizeof(buf);

    if (sysctlbyname("hw.model", &buf, &size, NULL, 0) == 0)
    {
        return std::string(buf);
    }

    return "Unknown CPU";
#elif defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string line;

    while (std::getline(f, line))
    {
        if (line.find("model name") != std::string::npos ||
            line.find("Hardware") != std::string::npos)
        {
            std::size_t pos = line.find(':');

            if (pos != std::string::npos)
            {
                return line.substr(pos + 2);
            }
        }
    }

    return "Unknown CPU";
#else
    return "Unknown CPU";
#endif
}

// -------------------------------------------------------------------------------------
// Processor Topology P-Core Detector (Excludes E-cores and LP-cores includes all 
// SMT/HT logical threads).
// -------------------------------------------------------------------------------------
struct PCoreInfo
{
#if defined(_WIN32)
    WORD      group;             // Processor group index
    KAFFINITY physicalCoreMask;  // Mask of all logical processors on this physical core
    KAFFINITY primaryThreadMask; // Isolated 1-bit mask representing a distinct logical processor on a P-core
#else
    int       logicalCpuId;      // Logical CPU identifier
#endif
};

#if defined(_WIN32)

// Returns all logical processor threads residing on physical P-cores.
inline std::vector<PCoreInfo> GetPhysicalPCores()
{
    std::vector<PCoreInfo> pCores;

    DWORD length = 0;

    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    {
        return pCores;
    }

    if (length == 0)
    {
        return pCores;
    }

    std::vector<uint8_t> buffer(length);

    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
                                          &length))
    {
        return pCores;
    }

    BYTE maxEfficiencyClass = 0;
    bool foundCore = false;

    for (DWORD offset = 0; offset < length;)
    {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);

        if (info->Relationship == RelationProcessorCore)
        {
            foundCore = true;

            maxEfficiencyClass = std::max<BYTE>(maxEfficiencyClass,
                                                info->Processor.EfficiencyClass);
        }

        if (info->Size == 0)
        {
            break;
        }

        offset += info->Size;
    }

    if (!foundCore)
    {
        return pCores;
    }

    for (DWORD offset = 0; offset < length;)
    {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);

        if (info->Relationship == RelationProcessorCore &&
            info->Processor.EfficiencyClass == maxEfficiencyClass)
        {
            for (WORD g = 0; g < info->Processor.GroupCount; ++g)
            {
                KAFFINITY coreMask = info->Processor.GroupMask[g].Mask;
                WORD group = info->Processor.GroupMask[g].Group;

                // Unpack each logical SMT/HT thread bit belonging to this physical P-core
                while (coreMask != 0)
                {
                    KAFFINITY singleThreadMask = coreMask & (~coreMask + 1);
                    coreMask &= ~singleThreadMask;

                    PCoreInfo core{};
                    core.group = group;
                    core.physicalCoreMask = info->Processor.GroupMask[g].Mask;
                    core.primaryThreadMask = singleThreadMask;
                    pCores.push_back(core);
                }
            }
        }

        if (info->Size == 0)
        {
            break;
        }

        offset += info->Size;
    }

    return pCores;
}

#elif defined(__linux__)

// Dynamically extracts asymmetric topology via Linux sysfs including all logical SMT/HT threads on P-cores
inline std::vector<PCoreInfo> GetPhysicalPCores()
{
    std::vector<PCoreInfo> pCores;
    std::map<int, int> coreCapacities;
    std::map<int, int> cpuToCoreId;
    int maxCapacity = 0;

    for (int i = 0; i < 1024; ++i)
    {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i);
        std::ifstream coreFile(path + "/topology/core_id");

        if (!coreFile.is_open())
        {
            continue;
        }

        int coreId = 0;
        
        if (coreFile >> coreId)
        {
            cpuToCoreId[i] = coreId;
        }

        std::ifstream capFile(path + "/cpu_capacity");
        int capacity = 0;
        
        if (capFile.is_open() && (capFile >> capacity))
        {
            coreCapacities[i] = capacity;
            
            if (capacity > maxCapacity)
            {
                maxCapacity = capacity;
            }
        }
    }

    for (const auto& [cpuId, coreId] : cpuToCoreId)
    {
        if (maxCapacity == 0 || coreCapacities[cpuId] == maxCapacity)
        {
            pCores.push_back({cpuId});
        }
    }

    return pCores;
}

#elif defined(__APPLE__)

// Queries Performance logical core topology on macOS (Apple Silicon)
inline std::vector<PCoreInfo> GetPhysicalPCores()
{
    std::vector<PCoreInfo> pCores;
    uint32_t performanceCoreCount = 0;
    size_t size = sizeof(performanceCoreCount);

    if ((sysctlbyname("hw.perflevel0.logicalcpu", &performanceCoreCount, &size, nullptr, 0) == 0 ||
         sysctlbyname("hw.perflevel0.physicalcpu", &performanceCoreCount, &size, nullptr, 0) == 0) && performanceCoreCount > 0)
    {
        pCores.reserve(performanceCoreCount);
        for (uint32_t i = 0; i < performanceCoreCount; ++i)
        {
            pCores.push_back({ static_cast<int>(i) });
        }
    }

    return pCores;
}

#else

inline std::vector<PCoreInfo> GetPhysicalPCores()
{
    return {};
}

#endif

// Alias for backwards compatibility
inline std::vector<PCoreInfo> GetPCoreLogicalThreads()
{
    return GetPhysicalPCores();
}

// -------------------------------------------------------------------------------------
// Initializes and starts a pool of worker threads synchronized on an event.
// Used to saturate the CPU and memory bus during high-throughput performance testing.
// Automatically pins threads to logical P-core threads to eliminate preemption and 
// throttling.
// -------------------------------------------------------------------------------------
inline void StartThreads(TestThreadManager* manager,
                         uint32_t           threadCount,
                         TestWorkerFunc     workerFunction,
                         void*              userContext)
{
    std::vector<PCoreInfo> pCores = GetPhysicalPCores();

    if (!pCores.empty())
    {
        if (threadCount > pCores.size())
        {
            threadCount = static_cast<uint32_t>(pCores.size());
        }
    }
    else
    {
        uint32_t hardwareConcurrency = std::thread::hardware_concurrency();
        
        if (hardwareConcurrency > 0 && threadCount > hardwareConcurrency)
        {
            threadCount = hardwareConcurrency;
        }
    }

    if (threadCount > MaxTestThreads)
    {
        threadCount = MaxTestThreads;
    }

    manager->threadCount = threadCount;
    manager->stopFlag.store(false);
    manager->startEvent.store(false);

    manager->threads.clear();
    manager->threads.reserve(threadCount);

    for (uint32_t index = 0; index < threadCount; ++index)
    {
        manager->contexts[index].threadId    = index;
        manager->contexts[index].startEvent  = &manager->startEvent;
        manager->contexts[index].stopFlag    = &manager->stopFlag;
        manager->contexts[index].userContext = userContext;

        manager->threads.emplace_back(
            workerFunction,
            &manager->contexts[index]);

#if defined(_WIN32)
        if (!pCores.empty() && index < pCores.size())
        {
            GROUP_AFFINITY groupAffinity{};
            groupAffinity.Group = pCores[index].group;
            groupAffinity.Mask = pCores[index].primaryThreadMask;

            SetThreadGroupAffinity(manager->threads.back().native_handle(), &groupAffinity, nullptr);
            SetThreadPriority(manager->threads.back().native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
        }
#elif defined(__linux__)
        if (!pCores.empty() && index < pCores.size())
        {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(pCores[index].logicalCpuId, &cpuset);
            pthread_setaffinity_np(manager->threads.back().native_handle(), sizeof(cpu_set_t), &cpuset);
        }
#endif
    }

    manager->startEvent.store(true);
    manager->startEvent.notify_all();
}

// -------------------------------------------------------------------------------------
// Signals threads to stop and waits for them to terminate safely.
// -------------------------------------------------------------------------------------
inline void StopAndWaitThreads(TestThreadManager* manager,
                               int                sleepSeconds)
{
    if (sleepSeconds > 0)
    {
        int iterations = sleepSeconds * 100;

        for (int index = 0; index < iterations; ++index)
        {
            if (g_abortTests.load() != 0)
            {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    manager->stopFlag.store(true);

    for (auto& thread : manager->threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

// -------------------------------------------------------------------------------------
// Target Context
// -------------------------------------------------------------------------------------
struct PatternContext
{
    uint32_t patternId; // Unique identifier linked to the matched pattern rule
};

// -------------------------------------------------------------------------------------
// Validates initialization and simultaneously checks MatchAll and MatchFirst results.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
inline bool RunSingleCorrectnessTest(const CharT* text,
                                     const CharT* pattern,
                                     uint8_t      scope,
                                     bool         shouldMatch,
                                     const char*  testName,
                                     TPolicy      policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));
    PatternContext ctx = {100};

    try
    {
        if (!engine.AddPattern(pattern,
                               static_cast<WildcardScope>(scope),
                               std::move(ctx)))
        {
            LOG_ERR("[!] %s - AddPattern Failed\n", testName);
            return false;
        }
    }
    catch (const std::invalid_argument&)
    {
        LOG_ERR("[!] %s - AddPattern Threw Unexpected std::invalid_argument\n", testName);
        return false;
    }

    std::vector<PatternContext*> matchedCtxAll;
    bool matchedAll = engine.Search(text, matchedCtxAll);

    PatternContext* matchedCtxFirst = nullptr;
    bool matchedFirst = engine.Search(text, matchedCtxFirst);

    if (matchedAll == shouldMatch && matchedFirst == shouldMatch)
    {
        LOG_INFO("[+] PASS: %s\n", testName);
        return true;
    }

    LOG_ERR("[!] FAIL: %s (Expected %d, Got MatchAll: %d, MatchFirst: %d)\n", testName, shouldMatch ? 1 : 0, matchedAll ? 1 : 0, matchedFirst ? 1 : 0);
    return false;
}

// -------------------------------------------------------------------------------------
// Validates overlapping rules to ensure MatchFirst properly extracts a single context.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
inline bool RunMultiPatternCorrectnessTest(const CharT* pattern1,
                                           const CharT* pattern2,
                                           const CharT* text,
                                           TPolicy      policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));
    PatternContext ctx1 = {1};
    PatternContext ctx2 = {2};

    if (!engine.AddPattern(pattern1, WildcardScope::Default, std::move(ctx1)))
    {
        return false;
    }

    if (!engine.AddPattern(pattern2, WildcardScope::Default, std::move(ctx2)))
    {
        return false;
    }

    std::vector<PatternContext*> results;
    bool matchedAll = engine.Search(text, results);

    PatternContext* firstResult = nullptr;
    bool matchedFirst = engine.Search(text, firstResult);

    if (matchedAll && results.size() == 2 && matchedFirst && firstResult != nullptr && (firstResult->patternId == 1 || firstResult->patternId == 2))
    {
        LOG_INFO("[+] PASS: Multi-Pattern MatchAll & MatchFirst Overlap\n");
        return true;
    }

    LOG_ERR("[!] FAIL: Multi-Pattern MatchAll & MatchFirst Overlap\n");
    return false;
}

// -------------------------------------------------------------------------------------
// Actually matches several overlapping patterns concurrently and validates results.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
inline bool RunMatchAllSeveralPatternsTest(const CharT* pattern1,
                                           const CharT* pattern2,
                                           const CharT* pattern3,
                                           const CharT* pattern4,
                                           const CharT* text,
                                           TPolicy      policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));
    PatternContext ctx1 = {10};
    PatternContext ctx2 = {20};
    PatternContext ctx3 = {30};
    PatternContext ctx4 = {40};

    if (!engine.AddPattern(pattern1, WildcardScope::Default, std::move(ctx1)))
    {
        return false;
    }

    if (!engine.AddPattern(pattern2, WildcardScope::Default, std::move(ctx2)))
    {
        return false;
    }

    if (!engine.AddPattern(pattern3, WildcardScope::Default, std::move(ctx3)))
    {
        return false;
    }

    if (!engine.AddPattern(pattern4, WildcardScope::Default, std::move(ctx4)))
    {
        return false;
    }

    std::vector<PatternContext*> results;
    bool matchedAll = engine.Search(text, results);

    if (matchedAll && (results.size() == 3))
    {
        bool hasPattern1 = false;
        bool hasPattern2 = false;
        bool hasPattern4 = false;

        for (size_t i = 0; i < results.size(); i++)
        {
            if (results[i]->patternId == 10)
            {
                hasPattern1 = true;
            }

            if (results[i]->patternId == 20)
            {
                hasPattern2 = true;
            }

            if (results[i]->patternId == 40)
            {
                hasPattern4 = true;
            }
        }

        if (hasPattern1 && hasPattern2 && hasPattern4)
        {
            LOG_INFO("[+] PASS: MatchAll Successfully Resolved Several Overlapping Patterns\n");
            return true;
        }
    }

    LOG_ERR("[!] FAIL: MatchAll Failed to Resolve Multiple Expected Patterns\n");
    return false;
}

// -------------------------------------------------------------------------------------
// Validates the engine correctly handles queries when zero patterns are registered,
// executing the early O(1) fast-fail branch.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
inline bool RunEmptyEngineTest(const CharT* text,
                               TPolicy      policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));

    PatternContext* firstResult = nullptr;
    bool matchedFirst = engine.Search(text, firstResult);

    if (!matchedFirst)
    {
        LOG_INFO("[+] PASS: Empty Engine Fast-Fail\n");
        return true;
    }

    LOG_ERR("[!] FAIL: Empty Engine Fast-Fail (Unexpected Match)\n");
    return false;
}

// -------------------------------------------------------------------------------------
// Validates explicit engine state reset via Clear(), ensuring memory arenas
// and vectors safely flush state outside of normal destruction.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
inline bool RunClearLifecycleTest(const CharT* pattern,
                                  const CharT* text,
                                  TPolicy      policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));
    PatternContext ctx = {100};

    if (!engine.AddPattern(pattern, WildcardScope::Default, std::move(ctx)))
    {
        return false;
    }

    PatternContext* matchedContext = nullptr;
    
    if (!engine.Search(text, matchedContext))
    {
        return false;
    }

    engine.Clear();

    if (!engine.Search(text, matchedContext))
    {
        LOG_INFO("[+] PASS: Engine Clear Lifecycle\n");
        return true;
    }

    LOG_ERR("[!] FAIL: Engine Clear Lifecycle (Ghost Match)\n");
    return false;
}

// -------------------------------------------------------------------------------------
// Validates that AsciiCaseFoldPolicy rejects patterns containing non-ASCII bytes
// -------------------------------------------------------------------------------------
inline bool RunAsciiPolicyRejectionTest()
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<char, PatternContext, AsciiCaseFoldPolicy<char>> engine(g_testConfig.optMode);
    PatternContext ctx = {100};

    const char* invalidPattern = "caf\xC3\xA9*";
    bool threwExpected = false;

    try
    {
        engine.AddPattern(invalidPattern, WildcardScope::Default, std::move(ctx));
    }
    catch (const std::invalid_argument&)
    {
        threwExpected = true;
    }

    if (threwExpected)
    {
        LOG_INFO("[+] PASS: AsciiCaseFoldPolicy Non-ASCII Pattern Registration Rejection\n");
        return true;
    }

    LOG_ERR("[!] FAIL: AsciiCaseFoldPolicy Permitted Non-ASCII Pattern Registration\n");
    return false;
}

// -------------------------------------------------------------------------------------
// Validates that custom normalization functors are not bypassed by SIMD CaseFold logic.
// -------------------------------------------------------------------------------------
struct CustomLowercaseFunctor
{
    inline wchar_t operator()(wchar_t character) const noexcept
    {
        if (character >= L'A' && character <= L'Z')
        {
            return character + 32;
        }
        
        return static_cast<wchar_t>(std::towlower(static_cast<wint_t>(character)));
    }
};

inline bool RunCustomFunctorCorrectnessTest()
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    using CustomPolicy = CustomCasePolicy<wchar_t, CustomLowercaseFunctor>;
    StringPatternMatch<wchar_t, PatternContext, CustomPolicy> engine(g_testConfig.optMode,
                                                                     CustomPolicy(CustomLowercaseFunctor{}));
    PatternContext ctx = {999};

    std::wstring patternString = L"HeLlo*";

    if (!engine.AddPattern(patternString, WildcardScope::Default, std::move(ctx)))
    {
        LOG_ERR("[!] FAIL: SIMD Custom Functor Hijack - AddPattern Failed\n");
        return false;
    }

    std::wstring textString = L"hElLo_world";

    PatternContext* firstResult = nullptr;
    bool matchedFirst = engine.Search(textString, firstResult);

    if (matchedFirst && firstResult != nullptr && firstResult->patternId == 999)
    {
        LOG_INFO("[+] PASS: SIMD Custom Functor Hijack (False Negatives)\n");
        return true;
    }

    LOG_ERR("[!] FAIL: SIMD Custom Functor Hijack (False Negatives)\n");
    return false;
}

// -------------------------------------------------------------------------------------
// Validates that patterns sharing identical prefixes are resolved in strict FIFO
// registration order during MatchFirst execution.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
inline bool RunFifoCollisionOrderTest(TPolicy policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));
    PatternContext ctx1 = {101};
    PatternContext ctx2 = {102};
    PatternContext ctx3 = {103};

    const CharT* p1 = StrLiteral<CharT>::get("service_*_start", L"service_*_start");
    const CharT* p2 = StrLiteral<CharT>::get("service_*",       L"service_*");
    const CharT* p3 = StrLiteral<CharT>::get("service_api_*",   L"service_api_*");
    
    const CharT* text = StrLiteral<CharT>::get("service_api_start", L"service_api_start");

    if (!engine.AddPattern(p1, WildcardScope::Default, std::move(ctx1)))
    {
        return false;
    }
    
    if (!engine.AddPattern(p2, WildcardScope::Default, std::move(ctx2)))
    {
        return false;
    }
    
    if (!engine.AddPattern(p3, WildcardScope::Default, std::move(ctx3)))
    {
        return false;
    }

    PatternContext* firstResult = nullptr;
    
    if (engine.Search(text, firstResult) && firstResult && firstResult->patternId == 101)
    {
        LOG_INFO("[+] PASS: FIFO Pattern Collision Chain Order Preserved\n");
        return true;
    }

    LOG_ERR("[!] FAIL: FIFO Pattern Collision Chain Order Violated\n");
    return false;
}

// -------------------------------------------------------------------------------------
// Registers patterns at lengths 2, 4, 7, and 16 to evaluate transitions between
// small-delta scalar unrolling (delta <= 4) and vectorized hashing (delta > 4).
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
inline bool RunRollingHashDeltaProgressionTest(TPolicy policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));
    PatternContext c1 = {1};
    PatternContext c2 = {2};
    PatternContext c3 = {3};
    PatternContext c4 = {4};

    if (!engine.AddPattern(StrLiteral<CharT>::get("ab*", L"ab*"), WildcardScope::Default, std::move(c1)))
    {
        return false;
    }
    
    if (!engine.AddPattern(StrLiteral<CharT>::get("abcd*", L"abcd*"), WildcardScope::Default, std::move(c2)))
    {
        return false;
    }
    
    if (!engine.AddPattern(StrLiteral<CharT>::get("abcdefg*", L"abcdefg*"), WildcardScope::Default, std::move(c3)))
    {
        return false;
    }
    
    if (!engine.AddPattern(StrLiteral<CharT>::get("abcdefghijklmnop*", L"abcdefghijklmnop*"), WildcardScope::Default, std::move(c4)))
    {
        return false;
    }

    std::vector<PatternContext*> results;
    const CharT* text = StrLiteral<CharT>::get("abcdefghijklmnop_extra", L"abcdefghijklmnop_extra");
    bool matched = engine.Search(text, results);

    if (matched && results.size() == 4)
    {
        LOG_INFO("[+] PASS: Rolling Hash Delta Progression (Micro-steps & Vector Transitions)\n");
        return true;
    }

    LOG_ERR("[!] FAIL: Rolling Hash Delta Progression (Expected 4 matches, got %zu)\n", results.size());
    return false;
}

// -------------------------------------------------------------------------------------
// OS-Specific Path Separators for Tests
// -------------------------------------------------------------------------------------
#if defined(_WIN32)
#define PATH_FILE "C:\\Temp\\Sub\\file.txt"
#define PATH_FILE_PAT "C:\\*\\file.txt"
#define PATH_TEST "C:\\Temp\\Sub\\test.txt"
#define PATH_TEST_PAT "C:\\*\\*.txt"
#define PATH_AB "a\\b"
#define PATH_FMT_0 "\\Device\\HarddiskVolume2\\Program Files\\App%u\\*\\crypt.exe"
#define PATH_FMT_1 "\\Device\\HarddiskVolume2\\Windows\\System32\\module_%u.dll"
#define PATH_FMT_2 "\\Device\\HarddiskVolume2\\Users\\*\\AppData\\Roaming\\payload_%u.exe"
#define PATH_FMT_3 "\\Device\\HarddiskVolume2\\Temp\\malware_%u.tmp"
#define PATH_TGT_MATCH "\\Device\\HarddiskVolume2\\Program Files\\App0\\bin\\crypt.exe"
#define PATH_TGT_MISMATCH "\\Device\\HarddiskVolume2\\Program Files\\SafeApp\\bin\\clean.exe"
#define PATH_DESC "File Paths (\\Device\\...)"
#else
#define PATH_FILE "/tmp/Sub/file.txt"
#define PATH_FILE_PAT "/*/file.txt"
#define PATH_TEST "/tmp/Sub/test.txt"
#define PATH_TEST_PAT "/*/*.txt"
#define PATH_AB "a/b"
#define PATH_FMT_0 "/var/lib/docker/overlay2/applications/App%u/*/crypt.bin"
#define PATH_FMT_1 "/usr/lib/x86_64-linux-gnu/systemd/system/module_%u.so"
#define PATH_FMT_2 "/home/*/.local/share/applications/bin/payload_%u.bin"
#define PATH_FMT_3 "/var/tmp/systemd-private-execution/malware_%u.tmp"
#define PATH_TGT_MATCH "/var/lib/docker/overlay2/applications/App0/bin/crypt.bin"
#define PATH_TGT_MISMATCH "/var/lib/docker/overlay2/applications/SafeApp/bin/clean.bin"
#define PATH_DESC "File Paths (/var/lib/docker/...)"
#endif

// -------------------------------------------------------------------------------------
// Macros dynamically wrapping string literals based on the character type (char vs wchar_t)
// -------------------------------------------------------------------------------------
#define WIDEN_STRING_IMPL(x) L##x
#define WIDEN_STRING(x) WIDEN_STRING_IMPL(x)
#define NARROW_STRING(x) x

#define EXECUTE_CORRECTNESS_SUITE(CharType, PolicyType, STR)                                                                                                                                           \
    do                                                                                                                                                                                                 \
    {                                                                                                                                                                                                  \
        constexpr bool isCI = PolicyType::isCaseInsensitive;                                                                                                                                           \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("file.txt"), STR("file.txt"), 0, true, "Exact Literal Match")));                                                                \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("file.txt"), STR("*.txt"), 0, true, "Wildcard Prefix Match")));                                                                 \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("prefix_file.txt"), STR("prefix*"), 0, true, "Wildcard Suffix Match")));                                                        \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("my_cool_file.txt"), STR("*cool*"), 0, true, "Wildcard Middle Match")));                                                        \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a"), STR("?"), 0, true, "Single Any (?) Match")));                                                                             \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("ab"), STR("?"), 0, false, "Single Any (?) Mismatch on Length")));                                                              \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abc"), STR("a?c"), 0, true, "Single Any (?) Middle")));                                                                        \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("Upper.TXT"), STR("upper.txt"), 0, isCI, isCI ? "Case Insensitivity Check (Match)" : "Case Sensitivity Check (Mismatch)")));   \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR(PATH_FILE), STR(PATH_FILE_PAT), 0, true, "Wildcard Default Scope (Allows Path Segment)")));                 \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR(PATH_FILE), STR(PATH_FILE_PAT), 1, false, "Wildcard Path Segment Scope (Blocks Path Segment)")));           \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("ExactCase.TXT"), STR("ExactCase.TXT"), 0, true, "Case Sensitivity Check (Exact Match)")));                                     \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("ExactStringTestMatch"), STR("ExactStringTestMatch"), 0, true, "Exact String Check (Match)")));                                 \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("ExactStringTestMatch"), STR("ExactStringMismatch"), 0, false, "Exact String Check (Mismatch)")));                              \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a_z_test"), STR("A_Z_TEST"), 0, isCI, "Boundary Transformation ('a' and 'z')")));                                             \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("`a{z`a{z"), STR("`A{Z`A{Z"), 0, isCI, "Adjacent Exclusion ('`' and '{')")));                                                  \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abcdefgh"), STR("ABCDEFGH"), 0, isCI, "Full Block Transformation")));                                                          \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("!@#$[]{}"), STR("!@#$[]{}"), 0, true, "Full Block Exclusion")));                                                               \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR(""), STR(""), 0, true, "Zero-Length Exact Match")));                                                                            \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR(""), STR("*"), 0, true, "Zero-Length Wildcard Match")));                                                                        \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a"), STR(""), 0, false, "Zero-Length Pattern Mismatch")));                                                                     \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a"), STR("a*"), 0, true, "Suffix Wildcard on Single Char")));                                                                  \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abcdef"), STR("a*f"), 0, true, "Boundary Wildcard Match")));                                                                   \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abcdef"), STR("a*z"), 0, false, "Boundary Wildcard Mismatch")));                                                               \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abc"), STR("????"), 0, false, "Too Many Wildcards (?)")));                                                                     \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abc"), STR("??"), 0, false, "Too Few Wildcards (?)")));                                                                        \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a1b2c3d4"), STR("a?b?c?d?"), 0, true, "Alternating Wildcards (?)")));                                                          \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("hello"), STR("**hello**"), 0, true, "Redundant Asterisks (*)")));                                                              \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("hello"), STR("*h*e*l*l*o*"), 0, true, "Scattered Asterisks (*)")));                                                            \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR(PATH_TEST), STR(PATH_TEST_PAT), 1, false, "Strict Path Wildcard Multiple Slashes (Path Segment Scope)")));     \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR(PATH_TEST), STR(PATH_TEST_PAT), 0, true, "Path Wildcard Multiple Slashes (Default Scope)")));                 \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a*b"), STR("a|*b"), 0, true, "Escape Character (Escaped *)")));                                                                \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a?b"), STR("a|?b"), 0, true, "Escape Character (Escaped ?)")));                                                                \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a|b"), STR("a||b"), 0, true, "Escape Character (Escaped |)")));                                                                \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a*b"), STR("a|?b"), 0, false, "Escape Character (Mismatch on Escaped)")));                                                     \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abc"), STR("a|*c"), 0, false, "Escape Character (Literal mismatch against wildcard)")));                                       \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abc|"), STR("abc|"), 0, true, "Escape Character (Dangling Escape)")));                                                         \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a|x"), STR("a|x"), 0, true, "Escape Character (Unrecognized Escape acts as literal)")));                                       \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("a|*b"), STR("a|||*b"), 0, true, "Escape Character (Consecutive Interleaved Escapes)")));                                       \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("A*B"), STR("a|*b"), 0, isCI, "Escape Character (Escaped Wildcard Case Policy)")));                                            \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("axb"), STR("a*?b"), 0, true, "Mixed Wildcards (*?) Match")));                                                                  \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("ab"), STR("a*?b"), 0, false, "Mixed Wildcards (*?) Length Mismatch")));                                                        \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("axyzb"), STR("a?*b"), 0, true, "Mixed Wildcards (?*) Match")));                                                                \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR(""), STR("?"), 0, false, "Single Any (?) against Empty String")));                                                              \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR(PATH_AB), STR("a?b"), 1, false, "Scope Path Segment blocks Single Any (?)")));                                                  \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("short"), STR("muchlongerpattern"), 0, false, "Length Bounds Fast-Fail")));                                                     \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("123456789"), STR("123456789"), 0, true, "SWAR Unaligned Tail Check (Match)")));                                               \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("123456789"), STR("123456780"), 0, false, "SWAR Unaligned Tail Check (Mismatch)")));                                           \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_12_suffix"), STR("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_12*"), 0, true, "65-Char Prefix Boundary Match (AVX-512 Tail + 1)"))); \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_1X_suffix"), STR("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_12*"), 0, false, "65-Char Prefix Boundary Mismatch (Tail Loop Rejection)"))); \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abc"), STR("abc*abc"), 0, false, "Prefix/Tail Overlap: Under-Length Fast-Fail")));                                             \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abcabc"), STR("abc*abc"), 0, true, "Prefix/Tail Overlap: Boundary Minimal Match")));                                           \
        CheckTest((RunSingleCorrectnessTest<CharType, PolicyType>(STR("abc_middle_abc"), STR("abc*abc"), 0, true, "Prefix/Tail Overlap: Full Match")));                                               \
        CheckTest((RunEmptyEngineTest<CharType, PolicyType>(STR("FastExitCheck"))));                                                                                                                   \
        CheckTest((RunClearLifecycleTest<CharType, PolicyType>(STR("flush.txt"), STR("flush.txt"))));                                                                                                  \
        CheckTest((RunMultiPatternCorrectnessTest<CharType, PolicyType>(STR("*.txt"), STR("file.txt"), STR("file.txt"))));                                                                            \
        CheckTest((RunMatchAllSeveralPatternsTest<CharType, PolicyType>(STR("*.txt"), STR("test.*"), STR("*.log"), STR("te*xt"), STR("test.txt"))));                                                  \
        CheckTest((RunFifoCollisionOrderTest<CharType, PolicyType>()));                                                                                                                                \
        CheckTest((RunRollingHashDeltaProgressionTest<CharType, PolicyType>()));                                                                                                                       \
    } while (0)

// -------------------------------------------------------------------------------------
// Validates truth table against multiple wildcard constraints for a specific char type.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT>
inline void RunCorrectnessForType(uint32_t& passedCount,
                                  uint32_t& failedCount)
{
    const char* typeName = std::is_same_v<CharT, char> ? "char" : "wchar_t";
    LOG_INFO("\n--- Correctness Tests (%s) ---\n", typeName);

    auto CheckTest = [&](bool result)
    {
        if (result)
        {
            passedCount++;
        }
        else
        {
            failedCount++;
        }
    };

    if constexpr (std::is_same_v<CharT, wchar_t>)
    {
        LOG_INFO("[+] Testing UnicodeCaseFoldPolicy<wchar_t>:\n");
        EXECUTE_CORRECTNESS_SUITE(wchar_t, UnicodeCaseFoldPolicy<wchar_t>, WIDEN_STRING);

        LOG_INFO("[+] Testing CaseSensitivePolicy<wchar_t>:\n");
        EXECUTE_CORRECTNESS_SUITE(wchar_t, CaseSensitivePolicy<wchar_t>, WIDEN_STRING);

        CheckTest(RunCustomFunctorCorrectnessTest());
    }
    else
    {
        LOG_INFO("[+] Testing CaseSensitivePolicy<char>:\n");
        EXECUTE_CORRECTNESS_SUITE(char, CaseSensitivePolicy<char>, NARROW_STRING);

        LOG_INFO("[+] Testing AsciiCaseFoldPolicy<char>:\n");
        EXECUTE_CORRECTNESS_SUITE(char, AsciiCaseFoldPolicy<char>, NARROW_STRING);

        CheckTest(RunAsciiPolicyRejectionTest());
    }
}

inline void RunAllCorrectnessTests(TestSuiteConfig* config)
{
    uint32_t passedCount = 0;
    uint32_t failedCount = 0;

    if (config->stringType == StringTestType::Both || config->stringType == StringTestType::WChar)
    {
        RunCorrectnessForType<wchar_t>(passedCount, failedCount);
    }

    if (config->stringType == StringTestType::Both || config->stringType == StringTestType::Char)
    {
        RunCorrectnessForType<char>(passedCount, failedCount);
    }

    if (failedCount == 0)
    {
        LOG_INFO("\n[+] ALL TESTS SUCCEEDED (%u passed)\n", passedCount);
    }
    else
    {
        LOG_ERR("\n[!] SOME TESTS FAILED (%u passed, %u failed)\n", passedCount, failedCount);
    }
}

// -------------------------------------------------------------------------------------
// Data packet fed into threaded performance workers.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
struct PerfWorkerContext
{
    StringPatternMatch<CharT, PatternContext, TPolicy>* engine;         // Pointer to the active string pattern matcher instance
    std::basic_string_view<CharT>                       targetText;     // View of the target text buffer to search
    std::atomic<int64_t>                                totalOpsPerSec; // Pre-aggregated rate of operations across all threads
    std::atomic<int>                                    startFlag;      // Flag indicating the worker has formally started
    bool                                                useMatchFirst;  // Toggles between MatchFirst and MatchAll modes
    bool                                                expectMatch;    // Indicates whether a match is logically expected
    std::atomic<int>                                    testFailed;     // Flag set if a correctness logic violation occurs
    std::atomic<int>                                    actualResult;   // Captures the erroneous boolean result on failure
};

// -------------------------------------------------------------------------------------
// Iteratively runs search operations in a tight loop to calculate throughput.
// Branches loop internally to prevent branch prediction overhead during measurement.
// Terminates loop and safely notifies orchestrator if an unexpected match logic 
// failure occurs. Applies thread affinity directly to the kernel thread scope if 
// executing natively.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
inline void PerfWorkerT(TestWorkerContext* context)
{
#if defined(__APPLE__)
    // Direct worker threads to physical Performance cores on macOS
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    PerfWorkerContext<CharT, TPolicy>* workerContext = static_cast<PerfWorkerContext<CharT, TPolicy>*>(context->userContext);
    context->startEvent->wait(false);

    uint64_t localOps = 0;
    bool isAborted = false;

    while (workerContext->startFlag.load() == 0)
    {
        if (g_abortTests.load() != 0)
        {
            isAborted = true;
            break;
        }

        std::this_thread::yield(); // mitigates thread start skew during metrics capture
    }

    if (!isAborted)
    {
        uint32_t searchLength = static_cast<uint32_t>(std::min<size_t>(workerContext->targetText.length(), 0xFFFFFFFFULL));
        std::basic_string_view<CharT> searchView(workerContext->targetText.data(), searchLength);

        // 1. Capture exact start time immediately before the work loop
        auto start = std::chrono::high_resolution_clock::now();

        if (workerContext->useMatchFirst)
        {
            PatternContext* resultData = nullptr;

            while (!context->stopFlag->load() && !g_abortTests.load())
            {
                bool result = workerContext->engine->Search(searchView, resultData);

                // Assert throughput correctness - instantly abort thread and flag for shutdown on logic failure
                if (result != workerContext->expectMatch)
                {
                    workerContext->actualResult.store(result ? 1 : 0);
                    workerContext->testFailed.store(1);
                    break;
                }

                localOps++;
                COMPILER_BARRIER();
            }
        }
        else
        {
            std::vector<PatternContext*> results;
            results.reserve(10);

            while (!context->stopFlag->load() && !g_abortTests.load())
            {
                bool result = workerContext->engine->Search(searchView, results);

                // Assert throughput correctness - instantly abort thread and flag for shutdown on logic failure
                if (result != workerContext->expectMatch)
                {
                    workerContext->actualResult.store(result ? 1 : 0);
                    workerContext->testFailed.store(1);
                    break;
                }

                localOps++;
                COMPILER_BARRIER();
            }
        }

        // 2. Capture exact end time immediately after the stop flag is detected
        auto end = std::chrono::high_resolution_clock::now();

        // 3. Calculate this thread's exact ops/sec, completely excluding OS teardown overhead
        uint64_t localNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        if (localNs > 0 && localOps > 0)
        {
            uint64_t localOpsPerSec = (localOps * 1000000000ULL) / localNs;
            workerContext->totalOpsPerSec.fetch_add(static_cast<int64_t>(localOpsPerSec));
        }
    }
}

// -------------------------------------------------------------------------------------
// Unified harness configuring metrics and orchestrating threaded execution.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy>
inline bool ExecutePerformanceBenchmark(StringPatternMatch<CharT, PatternContext, TPolicy>& engine,
                                        std::basic_string_view<CharT>                       targetText,
                                        uint32_t                                            patternCount,
                                        bool                                                expectMatch,
                                        const char*                                         testName,
                                        const char*                                         description,
                                        const char*                                         patternType)
{
    bool tierSuccess = true;
    const char* typeName = std::is_same_v<CharT, char> ? "char" : "wchar_t";
    uint32_t textLengthChars = static_cast<uint32_t>(targetText.length());

    LOG_INFO("\n--- Perf Test: %s (%s) [%u Patterns | Text: %u Chars] ---\n", testName, typeName, patternCount, textLengthChars);
    LOG_INFO("     %s\n", description);
    LOG_INFO("     Pattern Type: %s\n", patternType);
    LOG_INFO("     ------------------------------------------------------------------------------------\n");
    LOG_INFO("     %-18s | %-15s | %-15s | %-15s\n", "Engine Mode", "Throughput", "Ops/sec", "Latency/Op");
    LOG_INFO("     ------------------------------------------------------------------------------------\n");

    auto RunMode = [&](bool        useMatchFirst,
                       const char* label) -> void
    {
        if (g_abortTests.load() != 0)
        {
            return;
        }

        PerfWorkerContext<CharT, TPolicy> workerContext;
        workerContext.engine        = &engine;
        workerContext.targetText    = targetText;
        workerContext.totalOpsPerSec.store(0);
        workerContext.startFlag.store(0);
        workerContext.useMatchFirst = useMatchFirst;
        workerContext.expectMatch   = expectMatch;
        workerContext.testFailed.store(0);
        workerContext.actualResult.store(0);

        TestThreadManager manager;

        StartThreads(&manager, 1, PerfWorkerT<CharT, TPolicy>, &workerContext);

        std::this_thread::sleep_for(std::chrono::seconds(2));

        workerContext.startFlag.store(1);
        
        // Wait for threads to clean up (time taken here no longer corrupts the math)
        StopAndWaitThreads(&manager, 2);

        if (workerContext.testFailed.load())
        {
            LOG_ERR("     %-18s | [!] FAILED: Unexpected Result (Expected %d, Actual %d)\n", label, expectMatch ? 1 : 0, workerContext.actualResult.load());
            // Flag global suite termination
            g_abortTests.store(1);
            tierSuccess = false;
        }
        else
        {
            // Retrieve the purely isolated ops/sec from the threads
            const double opsPerSec = static_cast<double>(workerContext.totalOpsPerSec.load());

            // Calculate Physical Stream Ingestion Bandwidth
            const double bytesPerOp = static_cast<double>(textLengthChars) * sizeof(CharT);
            const double bytesPerSec = opsPerSec * bytesPerOp;
            const double mbPerSec = bytesPerSec / (1024.0 * 1024.0);

            // High precision latency calculation factoring in true thread parallelism
            const double totalThreads = manager.threadCount > 0 ? static_cast<double>(manager.threadCount) : 1.0;
            const double latencyUs = opsPerSec > 0.0 ? (1000000.0 * totalThreads) / opsPerSec : 0.0;

            // Format with two decimal fraction places if throughput is below 1 MB/s
            if (mbPerSec < 1.0)
            {
                LOG_INFO("     %-18s | %10.4f MB/s | %15llu | %9.4f us\n", label, mbPerSec, static_cast<unsigned long long>(opsPerSec), latencyUs);
            }
            else
            {
                LOG_INFO("     %-18s | %10.2f MB/s | %15llu | %9.4f us\n", label, mbPerSec, static_cast<unsigned long long>(opsPerSec), latencyUs);
            }
        }
    };

    RunMode(false, "MatchAll");
    RunMode(true, "MatchFirst");

    return tierSuccess;
}

// -------------------------------------------------------------------------------------
// Orchestrates performance generation routines for standard pathogical tests.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy = CaseSensitivePolicy<CharT>>
inline bool RunPerformanceTier(uint32_t    patternCount,
                               uint32_t    textLengthChars,
                               bool        pathological,
                               bool        expectMatch,
                               const char* testName,
                               const char* description,
                               const char* patternType,
                               TPolicy     policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));

    auto textData = std::make_unique<CharT[]>(static_cast<size_t>(textLengthChars) + 1);

    for (uint32_t i = 0; i < textLengthChars; i++)
    {
        if (expectMatch)
        {
            textData[i] = static_cast<CharT>('A' + (i % 26));
        }
        else
        {
            textData[i] = static_cast<CharT>('X'); // Forces negative matches
        }
    }

    // Inject a guaranteed match for Pattern 0 near the end of the text
    if (expectMatch && textLengthChars >= 16)
    {
        size_t searchLength = std::min<size_t>(textLengthChars, 0xFFFFFFFFULL);

        if (pathological)
        {
            textData[searchLength - 4] = static_cast<CharT>('A');
            textData[searchLength - 3] = static_cast<CharT>('0');
            textData[searchLength - 2] = static_cast<CharT>('B');
        }
        else
        {
            textData[searchLength - 9] = static_cast<CharT>('S');
            textData[searchLength - 8] = static_cast<CharT>('Y');
            textData[searchLength - 7] = static_cast<CharT>('S');
            textData[searchLength - 6] = static_cast<CharT>('0');
            textData[searchLength - 5] = static_cast<CharT>('0');
            textData[searchLength - 4] = static_cast<CharT>('0');
            textData[searchLength - 3] = static_cast<CharT>('0');
            textData[searchLength - 2] = static_cast<CharT>('B');
        }
    }

    textData[textLengthChars] = static_cast<CharT>('\0');

    constexpr size_t maxPathLength = 64;
    auto stringArray = std::make_unique<CharT[]>(static_cast<size_t>(patternCount) * maxPathLength);

    for (uint32_t i = 0; i < patternCount; i++)
    {
        CharT* currentPattern = &stringArray[i * maxPathLength];

        if (pathological)
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                std::snprintf(currentPattern, maxPathLength, "*A%uB*", i);
            }
            else
            {
                std::swprintf(currentPattern, maxPathLength, L"*A%uB*", i);
            }
        }
        else
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                std::snprintf(currentPattern, maxPathLength, "SYS%04u*B*", i);
            }
            else
            {
                std::swprintf(currentPattern, maxPathLength, L"SYS%04u*B*", i);
            }
        }

        PatternContext ctx = {i};

        if (!engine.AddPattern(currentPattern, WildcardScope::Default, std::move(ctx)))
        {
            return false;
        }
    }

    return ExecutePerformanceBenchmark(engine,
                                       std::basic_string_view<CharT>(textData.get(), textLengthChars),
                                       patternCount,
                                       expectMatch,
                                       testName,
                                       description,
                                       patternType);
}

// -------------------------------------------------------------------------------------
// Benchmarks realistic file path matching scenarios.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy = CaseSensitivePolicy<CharT>>
inline bool RunRealisticPathPerformanceTier(uint32_t    patternCount,
                                            bool        expectMatch,
                                            const char* testName,
                                            const char* description,
                                            const char* patternType,
                                            TPolicy     policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));

    constexpr size_t maxPathLength = 128;
    auto stringArray = std::make_unique<CharT[]>(static_cast<size_t>(patternCount) * maxPathLength);

    for (uint32_t i = 0; i < patternCount; i++)
    {
        CharT* currentPattern = &stringArray[i * maxPathLength];

        if constexpr (std::is_same_v<CharT, char>)
        {
            if (i % 4 == 0)      std::snprintf(currentPattern, maxPathLength, PATH_FMT_0, i);
            else if (i % 4 == 1) std::snprintf(currentPattern, maxPathLength, PATH_FMT_1, i);
            else if (i % 4 == 2) std::snprintf(currentPattern, maxPathLength, PATH_FMT_2, i);
            else                 std::snprintf(currentPattern, maxPathLength, PATH_FMT_3, i);
        }
        else
        {
            if (i % 4 == 0)      std::swprintf(currentPattern, maxPathLength, WIDEN_STRING(PATH_FMT_0), i);
            else if (i % 4 == 1) std::swprintf(currentPattern, maxPathLength, WIDEN_STRING(PATH_FMT_1), i);
            else if (i % 4 == 2) std::swprintf(currentPattern, maxPathLength, WIDEN_STRING(PATH_FMT_2), i);
            else                 std::swprintf(currentPattern, maxPathLength, WIDEN_STRING(PATH_FMT_3), i);
        }

        PatternContext ctx = {i};

        if (!engine.AddPattern(currentPattern, WildcardScope::Default, std::move(ctx)))
        {
            return false;
        }
    }

    // Change App42 to App0 to ensure (0 % 4 == 0) guarantees generation of a matching pattern
    const CharT* targetPath = expectMatch
                                  ? StrLiteral<CharT>::get(PATH_TGT_MATCH, WIDEN_STRING(PATH_TGT_MATCH))
                                  : StrLiteral<CharT>::get(PATH_TGT_MISMATCH, WIDEN_STRING(PATH_TGT_MISMATCH));

    return ExecutePerformanceBenchmark(engine,
                                       std::basic_string_view<CharT>(targetPath),
                                       patternCount,
                                       expectMatch,
                                       testName,
                                       description,
                                       patternType);
}

// -------------------------------------------------------------------------------------
// Validates complex meaningful strings surrounded by wildcard operators.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy = CaseSensitivePolicy<CharT>>
inline bool RunMeaningfulTextPerformanceTier(uint32_t    patternCount,
                                             uint32_t    textLengthChars,
                                             bool        expectMatch,
                                             const char* testName,
                                             const char* description,
                                             const char* patternType,
                                             TPolicy     policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));

    // Meaningful prose context to exercise realistic string density dynamically selected by char size
    const CharT* meaningfulText = StrLiteral<CharT>::get(
        "The Windows Kernel memory manager employs aggressive demand paging and lookaside lists to satisfy pool allocations efficiently. ",
        L"The Windows Kernel memory manager employs aggressive demand paging and lookaside lists to satisfy pool allocations efficiently. ");

    size_t snippetLength = std::char_traits<CharT>::length(meaningfulText);

    auto textData = std::make_unique<CharT[]>(static_cast<size_t>(textLengthChars) + 1);

    for (uint32_t i = 0; i < textLengthChars; i++)
    {
        textData[i] = meaningfulText[i % snippetLength];
    }

    textData[textLengthChars] = static_cast<CharT>('\0');

    constexpr size_t maxPathLength = 128;
    auto stringArray = std::make_unique<CharT[]>(static_cast<size_t>(patternCount) * maxPathLength);

    for (uint32_t i = 0; i < patternCount; i++)
    {
        CharT* currentPattern = &stringArray[i * maxPathLength];

        if (expectMatch)
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                std::snprintf(currentPattern, maxPathLength, "*allocation*%u*", i);
            }
            else
            {
                std::swprintf(currentPattern, maxPathLength, L"*allocation*%u*", i);
            }
        }
        else
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                std::snprintf(currentPattern, maxPathLength, "*random*notfound*%u*", i);
            }
            else
            {
                std::swprintf(currentPattern, maxPathLength, L"*random*notfound*%u*", i);
            }
        }

        // Ensure at least one known good pattern matches if we expect a match
        if (expectMatch && i == patternCount - 1)
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                std::snprintf(currentPattern, maxPathLength, "*allocation*");
            }
            else
            {
                std::swprintf(currentPattern, maxPathLength, L"*allocation*");
            }
        }

        PatternContext ctx = {i};

        if (!engine.AddPattern(currentPattern, WildcardScope::Default, std::move(ctx)))
        {
            return false;
        }
    }

    return ExecutePerformanceBenchmark(engine,
                                       std::basic_string_view<CharT>(textData.get(), textLengthChars),
                                       patternCount,
                                       expectMatch,
                                       testName,
                                       description,
                                       patternType);
}

// -------------------------------------------------------------------------------------
// Benchmarks purely exact string matching with no wildcards.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy = CaseSensitivePolicy<CharT>>
inline bool RunExactStringPerformanceTier(uint32_t    patternCount,
                                          uint32_t    textLengthChars,
                                          bool        expectMatch,
                                          const char* testName,
                                          const char* description,
                                          const char* patternType,
                                          TPolicy     policy = TPolicy{})
{
    if (g_abortTests.load() != 0)
    {
        return false;
    }

    StringPatternMatch<CharT, PatternContext, TPolicy> engine(g_testConfig.optMode, std::move(policy));

    auto textData = std::make_unique<CharT[]>(static_cast<size_t>(textLengthChars) + 1);

    for (uint32_t i = 0; i < textLengthChars; i++)
    {
        textData[i] = static_cast<CharT>('A' + (i % 26));
    }

    textData[textLengthChars] = static_cast<CharT>('\0');

    auto stringArray = std::make_unique<CharT[]>(static_cast<size_t>(patternCount) * (static_cast<size_t>(textLengthChars) + 1));

    for (uint32_t i = 0; i < patternCount; i++)
    {
        CharT* currentPattern = &stringArray[i * (textLengthChars + 1)];

        std::memcpy(currentPattern, textData.get(), textLengthChars * sizeof(CharT));
        currentPattern[textLengthChars] = static_cast<CharT>('\0');

        if (!expectMatch || i != (patternCount - 1))
        {
            if (textLengthChars > 0)
            {
                currentPattern[textLengthChars / 2] = static_cast<CharT>('X');
            }
        }

        PatternContext ctx = {i};

        if (!engine.AddPattern(currentPattern, WildcardScope::Default, std::move(ctx)))
        {
            return false;
        }
    }

    return ExecutePerformanceBenchmark(engine,
                                       std::basic_string_view<CharT>(textData.get(), textLengthChars),
                                       patternCount,
                                       expectMatch,
                                       testName,
                                       description,
                                       patternType);
}

// -------------------------------------------------------------------------------------
// Unified helper wrapping boilerplate execution of performance tiers for any text base.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT, CasePolicy<CharT> TPolicy, CasePolicy<CharT> TCSPolicy>
inline void ExecutePerformanceTiers(uint32_t& passedCount,
                                    uint32_t& failedCount)
{
    auto CheckTest = [&](bool result)
    {
        // If the global abort flag is set due to a catastrophic perf test logic failure, exit the block.
        if (g_abortTests.load() != 0)
        {
            return;
        }

        if (result)
        {
            passedCount++;
        }
        else
        {
            failedCount++;
        }
    };

    CheckTest((RunPerformanceTier<CharT, TPolicy>(10, 512, true, true, "Small Pathological", "Evaluates throughput against a saturated set of short pathological wildcard rules.", "Pathological (*A%dB*)")));
    CheckTest((RunPerformanceTier<CharT, TPolicy>(100, 10240, true, true, "Medium Pathological", "Evaluates throughput against a saturated set of medium pathological wildcard rules.", "Pathological (*A%dB*)")));
    CheckTest((RunPerformanceTier<CharT, TPolicy>(1000, 102400, true, true, "Large Pathological", "Evaluates throughput against a saturated set of long pathological wildcard rules.", "Pathological (*A%dB*)")));
    CheckTest((RunPerformanceTier<CharT, TPolicy>(100, 10240, true, false, "Negative Mismatch", "Tests engine behavior when no patterns match the target string.", "Pathological (*A%dB*)")));

    CheckTest((RunRealisticPathPerformanceTier<CharT, TPolicy>(10, true, "Realistic Paths", "Benchmarks against a small set of typical kernel file paths.", PATH_DESC)));
    CheckTest((RunRealisticPathPerformanceTier<CharT, TPolicy>(100, true, "Realistic Paths", "Benchmarks against a medium set of typical kernel file paths.", PATH_DESC)));
    CheckTest((RunRealisticPathPerformanceTier<CharT, TPolicy>(1000, true, "Realistic Paths", "Benchmarks against a massive set of typical kernel file paths.", PATH_DESC)));
    CheckTest((RunRealisticPathPerformanceTier<CharT, TPolicy>(10000, true, "Realistic Paths", "Benchmarks against a massive set of typical kernel file paths.", PATH_DESC)));
    CheckTest((RunRealisticPathPerformanceTier<CharT, TPolicy>(1000, false, "Negative Paths", "Tests engine behavior when no paths match the target.", PATH_DESC)));

    CheckTest((RunMeaningfulTextPerformanceTier<CharT, TPolicy>(100, 1024, true, "Meaningful Text", "Simulates 1KB real-world prose with multi-wildcard boundaries.", "Prose Boundaries (*xxx*)")));
    CheckTest((RunMeaningfulTextPerformanceTier<CharT, TPolicy>(100, 10240, true, "Meaningful Text", "Simulates 10KB real-world prose with multi-wildcard boundaries.", "Prose Boundaries (*xxx*)")));
    CheckTest((RunMeaningfulTextPerformanceTier<CharT, TPolicy>(100, 102400, true, "Meaningful Text", "Simulates 100KB real-world prose with multi-wildcard boundaries.", "Prose Boundaries (*xxx*)")));
    CheckTest((RunMeaningfulTextPerformanceTier<CharT, TPolicy>(100, 102400, false, "Meaningful Negative", "Tests mismatch behavior on 100KB real-world prose context.", "Prose Boundaries (*xxx*)")));
    
    CheckTest((RunMeaningfulTextPerformanceTier<CharT, TCSPolicy>(100, 102400, true, "Case-Sensitive Prose", "Simulates 100KB prose utilizing raw character evaluations.", "Prose Boundaries (*xxx*)")));

    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(10, 128, true, "Exact String", "Simulates 128-char text with exact string matching.", "Exact Literal (No Wildcards)")));
    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(100, 128, true, "Exact String", "Simulates 128-char text with exact string matching.", "Exact Literal (No Wildcards)")));
    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(1000, 128, true, "Exact String", "Simulates 128-char text with exact string matching.", "Exact Literal (No Wildcards)")));
    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(10000, 128, true, "Exact String", "Simulates 128-char text with exact string matching.", "Exact Literal (No Wildcards)")));
    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(100, 128, false, "Exact String Negative", "Tests mismatch behavior on 128-char exact string context.", "Exact Literal (No Wildcards)")));

    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(10, 256, true, "Exact String", "Simulates 256-char text with exact string matching.", "Exact Literal (No Wildcards)")));
    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(100, 256, true, "Exact String", "Simulates 256-char text with exact string matching.", "Exact Literal (No Wildcards)")));
    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(1000, 256, true, "Exact String", "Simulates 256-char text with exact string matching.", "Exact Literal (No Wildcards)")));
    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(10000, 256, true, "Exact String", "Simulates 256-char text with exact string matching.", "Exact Literal (No Wildcards)")));
    CheckTest((RunExactStringPerformanceTier<CharT, TPolicy>(100, 256, false, "Exact String Negative", "Tests mismatch behavior on 256-char exact string context.", "Exact Literal (No Wildcards)")));
}

// -------------------------------------------------------------------------------------
// Executes all parameterized performance benchmark variations.
// -------------------------------------------------------------------------------------
template <SupportedChar CharT>
inline void RunPerformanceForType(uint32_t& passedCount,
                                  uint32_t& failedCount)
{
    if constexpr (std::is_same_v<CharT, wchar_t>)
    {
        ExecutePerformanceTiers<wchar_t, UnicodeCaseFoldPolicy<wchar_t>, CaseSensitivePolicy<wchar_t>>(passedCount, failedCount);
    }
    else
    {
        ExecutePerformanceTiers<char, AsciiCaseFoldPolicy<char>, CaseSensitivePolicy<char>>(passedCount, failedCount);
    }
}

inline void RunAllPerformanceTests(TestSuiteConfig* config)
{
    LOG_INFO("\n=========================================================================================\n");
    LOG_INFO("--- Performance Tests (Throughput) ---\n");
    LOG_INFO("=========================================================================================\n");

    uint32_t passedCount = 0;
    uint32_t failedCount = 0;

    if (config->stringType == StringTestType::Both || config->stringType == StringTestType::WChar)
    {
        RunPerformanceForType<wchar_t>(passedCount, failedCount);
    }

    if (config->stringType == StringTestType::Both || config->stringType == StringTestType::Char)
    {
        RunPerformanceForType<char>(passedCount, failedCount);
    }

    if (failedCount == 0)
    {
        LOG_INFO("\n[+] ALL PERFORMANCE TESTS SUCCEEDED (%u passed)\n", passedCount);
    }
    else
    {
        LOG_ERR("\n[!] SOME PERFORMANCE TESTS FAILED (%u passed, %u failed)\n", passedCount, failedCount);
    }
}

static void set_high_priority()
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#else
    setpriority(PRIO_PROCESS, 0, -10);
#endif
}

// -------------------------------------------------------------------------------------
// Main routine that bootstraps all correctness checks and triggers performance evaluation.
// -------------------------------------------------------------------------------------
inline void RunTests(void* context)
{
    TestSuiteConfig* config = static_cast<TestSuiteConfig*>(context);

    LOG_INFO("\n");
    LOG_INFO("=========================================================================================\n");
    LOG_INFO("                 CPU: %s\n", cpu_name().c_str());
    LOG_INFO("=========================================================================================\n");

    LOG_INFO("\n=========================================================================================\n");
    LOG_INFO("                      STRINGPATTERNMATCH TEST SUITE\n");
    LOG_INFO("=========================================================================================\n");

    set_high_priority();

    // Correctness checks are always prioritized before executing expensive bandwidth operations
    if (config && config->runCorrectness)
    {
        RunAllCorrectnessTests(config);
    }

    if (config && config->runPerformance)
    {
        RunAllPerformanceTests(config);
    }

    LOG_INFO("\n--- All Tests Finished ---\n\n");

    FlushLogToFile();
}