# Changelog

## [0.4.0](https://github.com/scotteveritt/tqdb/compare/v0.3.0...v0.4.0) (2026-03-30)


### ⚠ BREAKING CHANGES

* redesign CLI with cobra, text search, workspace support

### Features

* add goreleaser for cross-platform binary releases ([fdaa116](https://github.com/scotteveritt/tqdb/commit/fdaa11659dfc11b3cb55ab5eb79106858021113a))
* bit-packed .tq format + configurable bit-width + improvement experiments ([a933865](https://github.com/scotteveritt/tqdb/commit/a933865fb4b8d047626838dae6987523bd920e04))
* float32 scoring in all search and attention loops ([ad175fd](https://github.com/scotteveritt/tqdb/commit/ad175fdd3cced19f8c4547623d62b065da6b4055))
* redesign CLI with cobra, text search, workspace support ([3a5a188](https://github.com/scotteveritt/tqdb/commit/3a5a188660be32fcdf0c22471d481b4fb10a0a5c))


### Bug Fixes

* check error returns on Close/Body.Close (errcheck lint) ([def8264](https://github.com/scotteveritt/tqdb/commit/def8264c824d6824e11124daf38e9857a52d15c1))
* profile all hot paths for PGO, not just brute-force ([3fd969a](https://github.com/scotteveritt/tqdb/commit/3fd969a973e51c2306b367c9444de6ea57732086))
* remove unused vecNormF function (lint) ([df0aaec](https://github.com/scotteveritt/tqdb/commit/df0aaec30a3eaf1a61cf3b66f945053eec3f4efd))
* skip long-running hypothesis tests in CI, add Centroids32 to Codebook ([4ee895a](https://github.com/scotteveritt/tqdb/commit/4ee895af2a02b7e0b4f57a8d376265badbaec1cd))
* suppress unparam lint in improvement test helper ([6785ae2](https://github.com/scotteveritt/tqdb/commit/6785ae2c60436ed06968a68a7f5f4d5be3a45bf2))


### Performance Improvements

* add PGO profiles for 10% faster brute-force search ([444ecb0](https://github.com/scotteveritt/tqdb/commit/444ecb0a336e14f253d786e2e1103e78c50a1b90))
* regenerate PGO profiles from current code ([cbae071](https://github.com/scotteveritt/tqdb/commit/cbae07159801458129ecbdcab6fd6e0ae4e72aec))

## [0.3.0](https://github.com/scotteveritt/tqdb/compare/v0.2.19...v0.3.0) (2026-03-30)


### ⚠ BREAKING CHANGES

* redesign Store.Add to accept Document, add SkipIVF to IndexConfig

### Features

* redesign Store.Add to accept Document, add SkipIVF to IndexConfig ([fc872bc](https://github.com/scotteveritt/tqdb/commit/fc872bccdaf8e1d3d0e628e2886314331fe0745f))

## [0.2.19](https://github.com/scotteveritt/tqdb/compare/v0.2.18...v0.2.19) (2026-03-30)


### Features

* filter indexes + e2e integration test ([9385b45](https://github.com/scotteveritt/tqdb/commit/9385b459bf3545d97a095db03e10255a4ec57d0b))
* IVF partitioning — ScaNN-style approximate nearest neighbor search ([e804a33](https://github.com/scotteveritt/tqdb/commit/e804a3379fcba445c08725e0f09f08095f2c25e4))
* performance optimizations + ForEach/AddRaw API for migration support ([0ad683a](https://github.com/scotteveritt/tqdb/commit/0ad683a7a38d8b3f875f6f605a4136809ea8760f))
* reject duplicate IDs in Collection.Add (VS2 alignment) ([6b030fe](https://github.com/scotteveritt/tqdb/commit/6b030fe5c9d622cbe30f5b9a8c04b91ee8726904))
* reproducible ANN benchmark suite ([bff7020](https://github.com/scotteveritt/tqdb/commit/bff70202661c74c5ff0658ff55d9f792bc25332e))
* Store IVF (auto-built) + rescore option for higher recall ([6be6a04](https://github.com/scotteveritt/tqdb/commit/6be6a04767bc8ba0a6bc2a578aea1973882fc23a))
* tqdb — SQLite for quantized vectors ([248b217](https://github.com/scotteveritt/tqdb/commit/248b21731d86015e2a6838133dc91f47eca4efb2))
* VS2-aligned filters, Document type, CRUD, and search options ([454be77](https://github.com/scotteveritt/tqdb/commit/454be77afb9ceb36b7f604c9f7b9e554859fb6f6))


### Bug Fixes

* add cmd/ directory, fix .gitignore pattern ([0c1641c](https://github.com/scotteveritt/tqdb/commit/0c1641c7023fdcfb58533c7ee2d8a2605b891914))
* Add silently skips duplicates instead of returning error ([d77897e](https://github.com/scotteveritt/tqdb/commit/d77897e5e1528399964ad91e4d86fe7f1ce5b5b0))
* format v2 with content section, filter perf, data destruction warning ([9eeac53](https://github.com/scotteveritt/tqdb/commit/9eeac53845cbbb2968c562625ab2bdf0a917cdd7))
* golangci-lint v2 config and type assertion checks ([2fdea28](https://github.com/scotteveritt/tqdb/commit/2fdea28953f2d6a025a52ef431a2cf51abb32fc9))
* remove dead outlier code, add Store.AddDocument, cache JSON, update README ([d41f894](https://github.com/scotteveritt/tqdb/commit/d41f894a2bc30125333d169efd858c1dd7ede683))
* resolve golangci-lint errcheck and gocritic issues ([ce64d52](https://github.com/scotteveritt/tqdb/commit/ce64d52b83a33c3a7fec32f49689cb365005a92e))
* single v1 format, remove version compat code ([b306c15](https://github.com/scotteveritt/tqdb/commit/b306c153b55588303f377d0393006251f7b554aa))
* use fmt.Fprintf instead of WriteString(fmt.Sprintf) (staticcheck QF1012) ([de1f22d](https://github.com/scotteveritt/tqdb/commit/de1f22dd69360da739cf6c29eb56e8c160699a55))


### Performance Improvements

* optimize QJL projection + benchmark MSE vs Prod recall ([4ed8576](https://github.com/scotteveritt/tqdb/commit/4ed85761ba74a162ce92f8904614192652b4dcf1))
