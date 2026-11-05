"""Guards against source substitution, frozen-output overwrite and partial context."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_d01_source import ROOT, anchors_present, validate_output, validate_pdf


class D01AuditTests(unittest.TestCase):
    def test_other_pdf_version_cannot_reuse_pinned_provenance(self):
        original = b"%PDF-synthetic-source-A"
        replacement = b"%PDF-synthetic-source-B"
        version = {"sha256": hashlib.sha256(original).hexdigest(), "bytes": len(original)}
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "fixture.pdf"
            pdf.write_bytes(replacement)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_pdf(pdf, version)
            self.assertEqual(pdf.read_bytes(), replacement)
            pdf.write_bytes(original)
            validate_pdf(pdf, version)

    def test_existing_run_is_rejected_without_touching_evidence(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            frozen = Path(directory) / "audit.json"
            frozen.write_text('{"frozen": true}\n')
            before = frozen.read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                validate_output(directory)
            self.assertEqual(frozen.read_bytes(), before)

    def test_raw_artifacts_cannot_be_written_to_tracked_or_escaped_paths(self):
        for path in (ROOT / "reports/new-audit", ROOT / "outputs", ROOT / "outputs/../data/new-audit"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_output(path)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                validate_output(Path(directory) / "untracked-but-not-output")

    def test_cap_without_attached_exception_does_not_pass_bundle_check(self):
        # Synthetic text: a numeric cap alone cannot stand in for an exception.
        anchors = ["fictional_cap", "fictional_exception"]
        self.assertFalse(anchors_present("fictional_cap only", anchors))
        self.assertTrue(anchors_present("fictional_cap\nfictional_exception", anchors))

    def test_development_fixture_is_bound_to_observed_mirror_not_historical_pdf(self):
        manifest = json.loads((ROOT / "data/development/d01/source_manifest.json").read_text())
        fixture = json.loads((ROOT / "data/development/d01/clauses.json").read_text())
        observed = manifest["versions"][fixture["source_version"]]
        self.assertEqual(fixture["source_sha256"], observed["sha256"])
        self.assertEqual(fixture["source_id"], observed["source_id"])
        self.assertNotEqual(observed["sha256"], manifest["versions"]["historical_jeju"]["sha256"])
        self.assertFalse(fixture["independent_human_approved"])


if __name__ == "__main__":
    unittest.main()
