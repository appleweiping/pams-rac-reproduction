# UCFRep-pose-110 asset audit (public summary)

Status: **`protocol-unverifiable`**

This package is a label-free, public-safe summary of the server-side asset
audit. It does not contain raw videos, video locators, video IDs, per-video
repetition counts, action labels, train/evaluation membership, pose
annotations, cycle-boundary annotations, or personal information.

## Conclusion

The exact UCFRep-pose-110 protocol cannot be reconstructed from the assets that
were available during this audit:

- the official 89/21 membership lists were not available;
- an independently audited mapping for all 110 videos was not available;
- the training-only salient-pose annotations for the 89 training videos were
  not available;
- the available five-class raw inventory is a lookup inventory, not a valid
  UCFRep-pose split;
- intersecting the standard UCFRep-526 ID manifests with those five classes
  yields 119 records, and its aggregate class cardinalities are incompatible
  with the published UCFRep-pose-110 cardinalities.

Consequently, PoseRAC-v1, GMFL, SPKDB, and BIGC cannot receive a strict
UCFRep-pose-110 reproduction result from these assets. Their fair-table cells
remain `protocol-unverifiable`; no substitute 89/21 split is fabricated.

## Public-safety decision

The two record-level inventories remain server-local:

- `raw-five-class-inventory.json` contains 646 records with video IDs, action
  labels, video locators, file sizes, and content hashes.
- `standard-five-class-intersection.json` contains 119 records with video IDs,
  action labels, source partitions, and pose-cache status.

Neither inventory contains per-video repetition counts or pose/cycle
annotations, but both expose record-level identifiers or labels and are
therefore intentionally excluded from this public package. The original
`asset-audit.json` is also replaced by this summary because it contains
class-by-partition aggregates.

`receipt.json` is copied byte-for-byte from the server artifact and contains
only artifact filenames and SHA-256 digests. `public-inventory.json` records the
review decision without reproducing any withheld record.

The SHA-256 of the reviewed server-side `asset-audit.json` is
`9c04fc8c8693a33ff6a78e1907c48a1801f3b2d90e55c6b79873378fb8632dca`.
