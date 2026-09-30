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
// Test Logic bootstrap for StringPatternMatch
// -------------------------------------------------------------------------------------
#include <atomic>
#include <csignal>
#include <cstring>
#include <thread>
#include "string_pattern_match_test.h"

// -------------------------------------------------------------------------------------
// Globals
// -------------------------------------------------------------------------------------
std::atomic<int> g_abortTests{ 0 };
TestSuiteConfig  g_testConfig{ .runCorrectness = true, 
                               .runPerformance = true, 
                               .stringType     = StringTestType::Both, 
                               .optMode        = OptimizationMode::Vectorized };

// -------------------------------------------------------------------------------------
// Signal Handling for Graceful Abort
// -------------------------------------------------------------------------------------
void SignalHandler(int signalNumber)
{
    if (signalNumber == SIGINT || signalNumber == SIGTERM)
    {
        LOG_INFO("\n[!] Abort requested. Signaling worker threads...\n");
        g_abortTests.store(1);
    }
}

// -------------------------------------------------------------------------------------
// User-Mode Entry Point
// -------------------------------------------------------------------------------------
int main(int   argc,
         char* argv[])
{
    // Register console abort signals
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // Parse command line arguments for string type and optimization mode selection
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--type=char") == 0)
        {
            g_testConfig.stringType = StringTestType::Char;
        }
        else if (std::strcmp(argv[i], "--type=wchar") == 0)
        {
            g_testConfig.stringType = StringTestType::WChar;
        }
        else if (std::strcmp(argv[i], "--type=both") == 0)
        {
            g_testConfig.stringType = StringTestType::Both;
        }
        else if (std::strcmp(argv[i], "--opt=auto") == 0)
        {
            g_testConfig.optMode = OptimizationMode::Auto;
        }
        else if (std::strcmp(argv[i], "--opt=scalar") == 0)
        {
            g_testConfig.optMode = OptimizationMode::Scalar;
        }
        else if (std::strcmp(argv[i], "--opt=vectorized") == 0)
        {
            g_testConfig.optMode = OptimizationMode::Vectorized;
        }
        else
        {
            LOG_ERR("[!] Unknown argument: %s\n", argv[i]);
            LOG_INFO("Usage: %s [--type=char|wchar|both] [--opt=auto|scalar|vectorized]\n", argv[0]);
            return 1;
        }
    }
    
    LOG_INFO("Usage: %s [--type=char|wchar|both] [--opt=auto|scalar|vectorized]\n\n", argv[0]);
    LOG_INFO("Running StringPatternMatch Test Suite.\n");
        
    // Check hardware support if Vectorized execution is mandated by the command line
    if (g_testConfig.optMode == OptimizationMode::Vectorized)
    {
        if (StringPatternMatchDetail::DetectCpuSimd() == StringPatternMatchDetail::SimdInstructionSet::Scalar)
        {
            LOG_INFO("[!] WARNING: Vectorized optimization requested but hardware SIMD is not supported. Engine will fall back to Scalar mode.\n\n");
        }
    }

    // Launch master benchmark execution thread
    std::thread masterThread(RunTests,
                             &g_testConfig);

    if (masterThread.joinable())
    {
        masterThread.join();
    }
    
    return 0;
}