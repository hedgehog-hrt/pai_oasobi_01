# Third-party notices and model terms

Reviewed on 1 November 2026. Original project software is Apache-2.0; the scope is specified in [docs/LICENSE_SCOPE.md](docs/LICENSE_SCOPE.md). The following upstream rights are retained. License texts are included without edits, and [LICENSES/SOURCES.json](LICENSES/SOURCES.json) records their sources and SHA-256 values. These notices cover the material described below, not every transitive dependency in an independently installed GPU environment.

## Included upstream excerpts and font software

### NVIDIA Isaac-GR00T

- Source: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/944fe70c4fd54e3fcb80a7d7258a328f9c061176), commit `944fe70c4fd54e3fcb80a7d7258a328f9c061176`.
- Included material: the upstream `pyproject.toml` text in `src/rosa_repro/shared_environment_sources.json`, also retained in historical source snapshots. Source URLs and hashes identify the upstream text. The enclosing JSON, generated dependency locks and project-specific environment handling are project packaging; they are not an upstream release.
- License: [Apache-2.0, upstream copy](LICENSES/Isaac-GR00T-LICENSE.txt).
- Attribution: NVIDIA/Isaac-GR00T project. No upstream source tree or model weights are included in the Git repository.
- A root `NOTICE` file was not present at this pinned commit (HTTP 404 when checked); no upstream NOTICE text was invented.

### vLLM

- Source: [vllm-project/vllm v0.11.2](https://github.com/vllm-project/vllm/tree/v0.11.2).
- Included material: upstream `requirements/cuda.txt` and `requirements/common.txt` in `src/rosa_repro/shared_environment_sources.json`, also retained in historical snapshots. Their source URLs and hashes are preserved; generated locks are project-produced derivatives of dependency specifications.
- License: [Apache-2.0, upstream copy](LICENSES/vLLM-LICENSE.txt).
- Attribution: vLLM project. The full runtime is installed separately.
- A root `NOTICE` file was not present at this tag (HTTP 404 when checked).

### Noto Sans JP

- Source: [Google Fonts Noto Sans JP](https://github.com/google/fonts/tree/main/ofl/notosansjp).
- Copyright 2014-2021 Adobe (http://www.adobe.com/), with Reserved Font Name 'Source'.
- License: [SIL Open Font License 1.1](LICENSES/NotoSansJP-OFL.txt). This copy matches the OFL file retained with the fonts used during report creation.
- Included material: embedded Noto Sans JP Regular/Bold font software in the fixed English report. Static instances were generated from the variable font; fonts were embedded/obfuscated for DOCX and embedded/subset as required by PDF production. The existing report files have not been changed by this licensing update.
- The font remains under OFL 1.1. The report's text and figures do not acquire the font's license merely by using the font.

### LIBERO and simulation inputs

- Source: [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01), commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Copyright (c) 2023 Lifelong Robot Learning.
- Software license: [MIT](LICENSES/LIBERO-LICENSE.txt).
- The simulator source is obtained separately. The fixed reproduction archive contains state banks and saved simulated observations; the videos show rendered simulation observations. The repository Apache license does not relicense these data or third-party scene, mesh, texture and dataset assets. Their applicable upstream terms remain in effect. The MIT software notice is not an assertion of identical licensing for all such materials.

## External models: weights are not bundled

| Model and use | Pinned revision | Upstream terms / evidence |
| --- | --- | --- |
| NVIDIA GR00T-N1.7-LIBERO, S1 | `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21` | [Pinned model card](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO/blob/2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21/README.md) identifies the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| NVIDIA Cosmos-Reason2-2B, GR00T dependency | Obtain the version required by the selected GR00T environment | [Model/access page](https://huggingface.co/nvidia/Cosmos-Reason2-2B): NVIDIA Open Model License Agreement and provider access process; this entry does not assert a newly verified checkpoint pin |
| Qwen2.5-VL-3B-Instruct, Safety | `66285546d2b821cf421d4f5eb2576359d3770cd3` | [Qwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE); [local copy](LICENSES/Qwen2.5-VL-3B-LICENSE.txt) |
| Qwen2.5-VL-7B-Instruct, S2 and Monitor | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | Apache-2.0 in the [pinned model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/cc594898137f460bfe9f0759e9844b3ce807cfb5/README.md) |
| Qwen3-VL-4B-Instruct, comparison | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Apache-2.0 in the [pinned model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md) |
| Qwen3-VL-8B-Instruct, comparison | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 in the [pinned model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md) |

**Qwen2.5-VL-3B-Instruct is not covered by an unrestricted Apache model grant.** Sections 1(i) and 2 of its license define the permitted standard use as non-commercial research/evaluation and require a separate license for commercial use. If redistributing Qwen Materials or derivatives, follow the full agreement, including its license/notice requirements. The reference attribution is: "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved."

The NVIDIA model license has its own conditions. The code license of Isaac-GR00T does not replace the license of model weights/configurations supplied by NVIDIA. Obtain models directly from the providers and accept their applicable terms; do not add credentials or model caches to this repository.

**Built on NVIDIA Cosmos.** The recorded GR00T-based prototype uses the Cosmos-Reason2 dependency. Section 3 of the NVIDIA agreement includes attribution requirements for model redistribution and systems using NVIDIA Cosmos. Preserve applicable model notices when distributing or deploying such a system; this repository's software license does not waive them.

## Other dependencies and research references

Ray, OR-Tools, PyTorch, Transformers, MuJoCo, CUDA/cuDNN and other installed packages keep their respective upstream license files and notices. They are external runtime/build dependencies, not relicensed by this project. Full binary/container redistribution requires an inventory of the actual packages and assets in that distribution; this source publication is not such a binary bundle.

The ROSA paper is a cited research reference. The independent project license grants no rights to the paper, NVIDIA trademarks, or third-party model/data materials beyond their own permissions. No endorsement by any upstream author or provider is implied.
