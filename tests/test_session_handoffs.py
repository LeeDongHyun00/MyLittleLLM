"""Reject incomplete/misattributed session handoffs before GitHub integration."""
from copy import deepcopy
import unittest

from scripts.check_session_handoffs import validate_registry


class SessionHandoffTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"project_id": "MyLittleLLM", "contract_version": "1.1.0",
                         "roles": {"S2": ["R01"], "S3": ["L01"]}}
        self.session = {"id": "test-s2", "role": "S2", "task_id": "R01",
                        "runtime": "conversation_subagent", "branch": "s2/r01-retrieval",
                        "code_baseline": "a" * 40, "contract_version": "1.0.0",
                        "status": "completed", "handoff": "docs/handoffs/test-s2.json",
                        "pull_request": None}
        self.registry = {"schema_version": 1, "project_id": "MyLittleLLM",
                         "event_transport": "github_actions", "external_webhook_configured": False,
                         "historical_contract_versions": ["1.0.0"],
                         "integration_branch": "s2/r01-retrieval", "sessions": [self.session]}
        self.handoff = {"session_role": "S2", **{key: self.session[key] for key in
                        ("task_id", "branch", "code_baseline", "contract_version", "status")}}
        self.validated_versions = []

    def validate(self, branch="s2/r01-retrieval"):
        return validate_registry(self.registry, self.manifest, lambda _: self.handoff,
                                 lambda _, m: self.validated_versions.append(m["contract_version"]), branch)

    def test_archived_handoff_uses_its_declared_version(self):
        self.assertEqual(len(self.validate()), 1)
        self.assertEqual(self.validated_versions, ["1.0.0"])
        self.registry["historical_contract_versions"] = []
        with self.assertRaisesRegex(ValueError, "Undeclared historical"):
            self.validate()

    def test_finished_session_requires_handoff(self):
        self.session["handoff"] = None
        with self.assertRaisesRegex(ValueError, "no handoff"):
            self.validate()
        self.session["status"] = "running"
        self.validate()

    def test_role_task_and_branch_must_match(self):
        self.session["task_id"] = "L01"
        with self.assertRaisesRegex(ValueError, "Role/task mismatch"):
            self.validate()
        self.session["task_id"] = "R01"
        self.session["branch"] = "s3/l01-rule-boundaries"
        with self.assertRaisesRegex(ValueError, "Branch does not match"):
            self.validate()

    def test_sessions_cannot_share_a_branch(self):
        duplicate = deepcopy(self.session)
        duplicate["id"] = "another-s2"
        self.registry["sessions"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Shared session branch"):
            self.validate()

    def test_stale_or_wrong_handoff_is_rejected(self):
        for field, wrong in (("session_role", "S3"), ("task_id", "L01"),
                             ("branch", "s2/other"), ("code_baseline", "b" * 40),
                             ("status", "partial"), ("contract_version", "0.9.0")):
            with self.subTest(field=field):
                original = self.handoff[field]
                self.handoff[field] = wrong
                with self.assertRaisesRegex(ValueError, f"handoff {field} mismatch"):
                    self.validate()
                self.handoff[field] = original

    def test_unregistered_role_branch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.validate("s3/l01-rule-boundaries")
        self.validate("docs/project-contract-and-session-roles")

    def test_path_escape_is_rejected_before_read(self):
        self.session["handoff"] = "docs/handoffs/../../../outside.json"
        with self.assertRaisesRegex(ValueError, "Invalid handoff path"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
