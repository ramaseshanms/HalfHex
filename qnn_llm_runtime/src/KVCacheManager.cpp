// KVCacheManager.cpp — implementation is header-only in KVCacheManager.h
// This file exists for build system completeness and future extensions.
#include "KVCacheManager.h"

KVCacheManager::~KVCacheManager() {
    release();
}
