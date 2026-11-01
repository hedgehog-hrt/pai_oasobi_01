# Versioned reproduction materials

The historical experiment materials are distributed separately from the Git source tree. In the local publication package, the reference ZIP and its license-and-notice companion are in the sibling `release-assets/` directory. On GitHub, obtain both ZIPs and `SHA256SUMS` from the repository's Releases area after the maintainer attaches them. A repository-specific URL is not invented in this pre-publication copy.

| Property | Value |
| --- | --- |
| Filename | `ROSA_reproduction_reference_20261101_rev1.zip` |
| Size | 57,243,338 bytes |
| SHA-256 | `ee83463004ccd24c7aea2abbaa5baf974f280b580e67ca4545e8013085046f5d` |
| Contents | 15 historical source snapshots, 11 closed-loop cases, individual profile settings, state banks and 227 saved simulation observations |
| Excluded | Model weights, actual connection endpoints, authentication material and the complete raw time-series logs |

Verify the downloaded ZIP before extracting it:

```bash
# Linux
sha256sum ROSA_reproduction_reference_20261101_rev1.zip
# macOS
shasum -a 256 ROSA_reproduction_reference_20261101_rev1.zip
```

Compare the output to the value above. After extraction, change into `ROSA_reproduction_reference_20261101_rev1/` and verify its existing internal `SHA256SUMS`. Follow its `README.txt`, `reproduction_index.json` and `sources/index.json`, together with Appendices B–F of the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf). The archive's preparation helper does not load models or start a GPU trial.

The ZIP is byte-for-byte identical to the fixed Revision 1 artifact. Its historical descriptions of local distribution reflect when it was created; the publication copy and Release attachment instructions are here. Running the latest Git source is not a substitute for selecting the source snapshot bound to a historical case. Do not mix v0.20.17 CPU-only fixes into a purported repetition of v0.20.16.

## License-and-notice companion

Also obtain `ROSA_licenses_and_notices_20261101.zip` and verify it using the Release's `SHA256SUMS`. Extract it beside the reference archive and retain it when sharing or extracting historical source snapshots. It supplies Apache-2.0 for the original project software, upstream license texts and attribution, and the content/model exceptions described in [LICENSE_SCOPE.md](LICENSE_SCOPE.md). The companion identifies the exact reference ZIP by its SHA-256 and includes its own internal checksums.

The existing reference ZIP and its nested source snapshots are not rewritten. Its saved observations/state banks receive no blanket Apache or CC BY license from the software grant. Model weights remain external. Distribute the reference ZIP and license companion together, rather than presenting the reference ZIP as a fully Apache-licensed collection of code, models and data.
