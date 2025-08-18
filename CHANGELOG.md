## v0.3.0 (2025-08-19)
- chore: add release.sh script (7bb84aa)


# Changelog

## [0.2.0] - 2025-08-17
### Added
- **Warmup API** (`db.warmup`) to preload hot data into memory
- **Autotune API** (`hotpaths.autotune`) to select the best backend automatically (Python / NumPy / Cython / C++)
- **Health Check API** (`db.health_check`) to monitor cache hit ratio and memory usage
- **Dispatch Configuration** (`hotpaths.get_dispatch_config`) for backend transparency
- **Logging support** to debug which kernel/backend was used

### Changed
- Internal optimization of hotpath dispatching
- Improved memory layout handling for custom arrays

### Fixed
- Minor stability issues in profile utils

---

## [0.1.0] - 2025-07-xx
### Added
- Initial release of FastLayer
- Core modules: `memDB`, `hotpaths`, `profile_utils`
- Basic NumPy / Numba accelerated hotpaths
- AUR package: `python-fastlayer-git`

