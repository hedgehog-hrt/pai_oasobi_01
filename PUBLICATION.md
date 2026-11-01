# Public snapshot: 1 November 2026

This directory contains the publication copy of the ROSA reproduction prototype. The original working directory, historical logs and fixed deliverables were preserved separately. No new GPU experiments were performed for this publication package.

## Included material

- Current v0.20.17 source, entry-point scripts, tests, schemas, synthetic examples and configuration templates.
- Implementation and experiment notes. Most historical notes are in Japanese; the repository README, result summary and submission report are in English.
- The fixed English submission report, as PDF and DOCX, in `reports/`.
- The fixed 3:24 P1–P4 demonstration and the earlier fixed 2:54 P2–P4 demonstration in `media/`, with their explanatory/source-mapping sidecars.
- A fresh `SHA256SUMS` inventory and a portable verifier, `scripts/verify_publication.py`.

The [versioned reproduction ZIP](docs/REPRODUCTION_ASSETS.md) and its license-and-notice companion are supplied separately for a GitHub Release. They are not part of this Git source tree. The reference ZIP's existing internal checksums and historical source snapshots are preserved.

The report, videos and reproduction ZIP retain their original bytes. P1's four environments and the three P2–P4 environments are separate experiments. The combined video does not show seven environments running simultaneously. Frame-mapping paths identify the original saved observations; the complete raw captures are not included here.

## Excluded material

The public copy excludes raw experimental result directories, host-specific deployment/collection logs, connection records, old report packages, temporary authoring files, virtual environments, caches, model weights, credentials and Git history. The selected reproduction ZIP contains the limited simulation observations and state banks described in its README, not the full raw experiment log archive.

Historical notes retain result paths as provenance identifiers. Links to omitted files were converted to explicit non-bundled references. Published aggregate results are in [RESULTS.md](RESULTS.md) and the [English report](reports/ROSA_single_GPU_submission_EN_20261101.pdf); omitted evidence is not claimed to be downloadable from this repository.

## Publication-specific changes

The root README links and directory guide were updated for this layout. `RESULTS.md` is an English publication summary, replacing the working copy's history index. `.gitignore` excludes generated results, credentials and local connection configuration. `.gitattributes` preserves the exact bytes covered by checksums.

The current connection template, template generator and associated CPU test fixtures use generic SSH port 22 with a placeholder hostname. Configure the actual connection locally; do not commit it. Historical source snapshots inside the immutable reproduction ZIP retain their original generic examples and version identities. They do not contain the actual Pod endpoint or authentication material detected by the publication checks.

These packaging changes do not claim a new model or GPU validation. Current v0.20.17 still has CPU-only validation of the Monitor termination fix; v0.20.16 remains the last GPU-tested version.

## Verification

From this repository root:

```bash
python3 scripts/verify_publication.py
```

The verifier checks every listed hash, missing files and unexpected non-ignored files. With Git installed it respects `.gitignore`; without Git it checks file hashes and reports that the extra-file check is unavailable. The standard alternatives are `sha256sum -c SHA256SUMS` on Linux or `shasum -a 256 -c SHA256SUMS` on macOS; these alternatives do not reject extra files. The checksum manifest itself is excluded from its own inventory. Checksums detect changes relative to this snapshot; they are not a digital signature or proof of authorship.

Privacy checks cover filenames, textual contents, recursively inspected ZIP/NPZ/DOCX members, PDF text/metadata and video container metadata. The review checks known personal identifiers and local paths, actual historical endpoints, recognizable access tokens, private keys, credential-bearing URLs, and other email/IP candidates. Simulator imagery and the fixed report/video content were previously visually reviewed; those artifacts are unchanged. Pattern checks are not a guarantee that every conceivable sensitive fact is detectable. Review newly added logs or files independently before publishing them.

See [the publication verification record](PUBLICATION_VALIDATION.json) for the checks performed on this copy. The README's CPU examples remain separate from real GPU reproduction.

## License status

Original project software is now licensed under [Apache License 2.0](LICENSE), as selected by the project owner. See the [README licensing section](README.md#licensing), [detailed scope](docs/LICENSE_SCOPE.md), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md). The fixed reports/media and recorded experiment data are excluded from the repository-wide software grant; no CC BY 4.0 content license was assigned. No personal copyright attribution was invented.

The original Qwen2.5-VL-3B Safety model remains subject to its research license, including its non-commercial research/evaluation limit and requirement for a separate license for commercial use. GR00T/Cosmos model terms, upstream software terms and the embedded fonts' OFL remain distinct. No model weights or upstream GR00T/LIBERO source trees are included in this Git repository. Distribute the separate reproduction ZIP with its matching license-and-notice companion.
