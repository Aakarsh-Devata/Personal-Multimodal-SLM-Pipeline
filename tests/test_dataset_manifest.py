"""Offline dataset-contract tests. Synthetic bytes/labels are not model outputs."""
import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slm_pipeline.datasets.__main__ import main, check_reference_destination
from slm_pipeline.datasets.manifest import load_manifest, prepare_local, validate_manifest, verify_source
from slm_pipeline.datasets.references import read_epic_references, timestamp_seconds

EXAMPLE = Path(__file__).resolve().parents[1] / "datasets" / "epic-kitchens-smoke.example.json"


class DatasetManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(EXAMPLE)
        self.record = self.manifest["records"][0]
        self.record.update(source_duration_seconds=10.0, clip_interval_seconds=[0, 10.0],
                           verification_status="pending", sha256=None, size_bytes=None)
        for ref in self.record["reference_annotations"]:
            ref["sha256"] = None
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def source(self):
        path = self.root / "fixture.mp4"
        path.write_bytes(b"synthetic local fixture, not real video")
        return path

    def action_csv(self, rows=None):
        path = self.root / "actions.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["narration_id", "video_id", "start_timestamp", "stop_timestamp", "narration"])
            writer.writeheader()
            writer.writerows(rows or [dict(narration_id="P02_05_0", video_id="P02_05", start_timestamp="00:00:01.250", stop_timestamp="00:00:03.500", narration="synthetic test action")])
        return path

    def test_example_metadata_is_valid_but_does_not_claim_local_verification(self):
        self.assertEqual(load_manifest(EXAMPLE)["records"][0]["size_bytes"], 70130936)
        with patch("slm_pipeline.datasets.manifest.probe_video", side_effect=AssertionError("no I/O")):
            self.assertEqual(main(["validate", "--manifest", str(EXAMPLE)]), 0)

    def test_bad_timestamps_and_intervals_are_rejected(self):
        for interval in ([True, 2], [0, float("nan")], [-1, 2], [4, 4], [5, 4], [0, 11], [0, "4"]):
            with self.subTest(interval=interval):
                bad = copy.deepcopy(self.manifest)
                bad["records"][0]["clip_interval_seconds"] = interval
                with self.assertRaises(ValueError):
                    validate_manifest(bad)
        self.record["source_duration_seconds"] = 200
        self.record["clip_interval_seconds"] = [0, 121]
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)

    def test_status_checksum_license_and_timestamp_contract(self):
        for key, value in [("verification_status", "downloaded"), ("sha256", "invented"), ("license_id", ""),
                           ("source_url", "http://example.org/a.mp4"), ("video_id", "../escape"),
                           ("timestamp_origin", "wall_clock"), ("timestamp_unit", "frames"), ("size_bytes", True)]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.manifest)
                bad["records"][0][key] = value
                with self.assertRaises(ValueError):
                    validate_manifest(bad)
        self.record["verification_status"] = "verified"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)

    def test_duplicate_video_and_nonreference_annotation_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["records"].append(copy.deepcopy(self.record))
        with self.assertRaises(ValueError):
            validate_manifest(bad)
        self.record["reference_annotations"][0]["role"] = "transcript"
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest)

    @patch("slm_pipeline.datasets.manifest.probe_video", return_value=10.0)
    def test_verify_checks_real_bytes_and_requires_known_digest(self, probe):
        path = self.source()
        with self.assertRaises(ValueError):
            verify_source(self.record, path)
        self.record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(verify_source(self.record, path)["size_bytes"], path.stat().st_size)
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            verify_source(self.record, path)

    @patch("slm_pipeline.datasets.manifest.probe_video", return_value=12.0)
    def test_video_duration_mismatch_is_rejected(self, probe):
        with self.assertRaisesRegex(ValueError, "duration"):
            verify_source(self.record, self.source(), require_checksum=False)

    @patch("slm_pipeline.datasets.manifest.probe_video", return_value=10.0)
    def test_prepare_records_observed_digest_and_retains_source(self, probe):
        source = self.source()
        raw = self.root / "raw"
        with self.assertRaisesRegex(ValueError, "license"):
            prepare_local(self.manifest, "P02_05", source, raw)
        updated, target, evidence = prepare_local(self.manifest, "P02_05", source, raw, acknowledge_license=True)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual(updated["records"][0]["sha256"], evidence["sha256"])
        self.assertEqual(updated["records"][0]["verification_status"], "verified")
        self.assertIsNone(self.record["sha256"])
        with self.assertRaises(FileExistsError):
            prepare_local(self.manifest, "P02_05", source, raw, acknowledge_license=True)

    def test_partial_clip_prepare_fails_instead_of_mislabeling_source(self):
        self.record["clip_interval_seconds"] = [1, 5]
        with self.assertRaisesRegex(ValueError, "full short videos"):
            prepare_local(self.manifest, "P02_05", self.root / "missing.mp4", self.root / "raw", acknowledge_license=True)

    def test_csv_timestamps_strict(self):
        self.assertEqual(timestamp_seconds("01:02:03.125"), 3723.125)
        for value in ("1.25", "NaN", "00:60:01", "00:00:60", "-01:00:00", " 00:00:01", "0:00:01", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                timestamp_seconds(value)

    def test_reference_export_keeps_human_labels_separate_and_rebases(self):
        path = self.action_csv()
        self.record["clip_interval_seconds"] = [2, 5]
        result = read_epic_references(path, self.record, "action")
        self.assertEqual(result["role"], "evaluation_reference_only")
        annotation = result["annotations"][0]
        self.assertEqual(annotation["source_interval_seconds"], [1.25, 3.5])
        self.assertEqual(annotation["clip_interval_seconds"], [0, 1.5])
        self.assertEqual(annotation["original"]["narration"], "synthetic test action")
        self.assertNotIn("transcript", result)
        self.assertEqual(result["source_csv_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_csv_checksum_and_invalid_source_intervals_fail(self):
        path = self.action_csv()
        self.record["reference_annotations"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            read_epic_references(path, self.record, "action")
        self.record["reference_annotations"][0]["sha256"] = None
        self.record["source_duration_seconds"] = 3.0
        with self.assertRaisesRegex(ValueError, "outside source"):
            read_epic_references(path, self.record, "action")

    def test_reference_destination_checks_custom_roots_and_actual_filenames(self):
        config = {"paths": {"processed_dir": str(self.root / "custom-observations"),
                            "memory_dir": str(self.root / "custom-store")}}
        for path in (self.root / "custom-observations" / "labels.json",
                     self.root / "custom-store" / "nested" / "labels.json",
                     self.root / "references" / "context_blocks.json",
                     self.root / "references" / "vision_captions.json",
                     self.root / "references" / "semantic_structure.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_reference_destination(path, config)
        expected = self.root / "references" / "evaluation.json"
        self.assertEqual(check_reference_destination(expected, config), expected)
        (self.root / "alias").symlink_to(self.root / "custom-store", target_is_directory=True)
        with self.assertRaises(ValueError):
            check_reference_destination(self.root / "alias" / "labels.json", config)

    @patch("slm_pipeline.datasets.manifest.probe_video", return_value=10.0)
    def test_failed_copy_leaves_no_partial_pipeline_input(self, probe):
        source = self.source()
        raw = self.root / "raw"
        with patch("slm_pipeline.datasets.manifest.shutil.copyfileobj", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                prepare_local(self.manifest, "P02_05", source, raw, acknowledge_license=True)
        self.assertEqual(list(raw.iterdir()), [])

    def test_cli_refuses_reference_output_in_generated_artifacts(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.manifest))
        result = main(["references", "--manifest", str(manifest), "--video", "P02_05", "--kind", "action",
                       "--csv", str(self.action_csv()), "--output", str(self.root / "processed" / "transcript.json")])
        self.assertEqual(result, 2)
        self.assertFalse((self.root / "processed").exists())


if __name__ == "__main__":
    unittest.main()
