"""Strict identifier and command contract tests for analyst review."""

import unittest
from uuid import UUID

from pydantic import ValidationError

from app.models.audience_review import (
    AUDIENCE_REVIEW_NAMESPACE,
    MAX_PRIVATE_REJECT_NOTE_LENGTH,
    ApproveReviewCommand,
    RejectReasonCode,
    RejectReviewCommand,
    new_command_id,
    new_run_id,
    parse_review_command,
    review_id_for,
    review_thread_id,
)
from app.agent.audience_review_runtime import _payload_digest


class AudienceReviewContractTests(unittest.TestCase):
    def assert_private_input_hidden(
        self,
        error: ValidationError,
        secret: str,
    ) -> None:
        rendered = (
            str(error)
            + repr(error)
            + repr(error.errors())
            + error.json()
        )
        self.assertNotIn(secret, rendered)

    def test_identifiers_have_required_versions_and_stability(self) -> None:
        run_id = new_run_id()
        command_id = new_command_id()
        first = review_id_for(run_id, "cluster-one")

        self.assertEqual(UUID(run_id).version, 4)
        self.assertEqual(UUID(command_id).version, 4)
        self.assertEqual(UUID(first).version, 5)
        self.assertEqual(first, review_id_for(run_id, "cluster-one"))
        self.assertNotEqual(first, review_id_for(run_id, "cluster-two"))
        self.assertEqual(
            review_thread_id(run_id),
            f"wikipulse-review-v1:{run_id}",
        )
        self.assertEqual(
            str(AUDIENCE_REVIEW_NAMESPACE),
            "76d7067d-f0cb-4c7c-b9c6-c14c4691fdd4",
        )

    def test_command_requires_canonical_uuid_versions(self) -> None:
        run_id = new_run_id()
        with self.assertRaises(ValidationError):
            ApproveReviewCommand(
                type="approve",
                run_id=run_id.upper(),
                review_id=review_id_for(run_id, "cluster"),
                cluster_id="cluster",
                expected_version=1,
                command_id=new_command_id(),
            )

    def test_other_rejection_requires_bounded_private_note(self) -> None:
        run_id = new_run_id()
        values = dict(
            type="reject",
            run_id=run_id,
            review_id=review_id_for(run_id, "cluster"),
            cluster_id="cluster",
            expected_version=1,
            command_id=new_command_id(),
            reason_code=RejectReasonCode.OTHER,
        )
        with self.assertRaises(ValidationError):
            RejectReviewCommand(**values)
        with self.assertRaises(ValidationError):
            RejectReviewCommand(
                **values,
                private_note="x" * (MAX_PRIVATE_REJECT_NOTE_LENGTH + 1),
            )
        command = RejectReviewCommand(**values, private_note="  Analyst note  ")
        self.assertEqual(command.private_note, "Analyst note")

    def test_reject_contract_forbids_unknown_fields(self) -> None:
        run_id = new_run_id()
        with self.assertRaises(ValidationError):
            RejectReviewCommand(
                type="reject",
                run_id=run_id,
                review_id=review_id_for(run_id, "cluster"),
                cluster_id="cluster",
                expected_version=1,
                command_id=new_command_id(),
                reason_code=RejectReasonCode.SAFETY_CONCERN,
                prompt="forbidden",
            )

    def test_control_character_note_is_redacted_from_validation_error(self) -> None:
        run_id = new_run_id()
        secret = "SECRET_CONTROL_NOTE_6e12\nprivate"
        with self.assertRaises(ValidationError) as raised:
            RejectReviewCommand(
                type="reject",
                run_id=run_id,
                review_id=review_id_for(run_id, "cluster"),
                cluster_id="cluster",
                expected_version=1,
                command_id=new_command_id(),
                reason_code=RejectReasonCode.OTHER,
                private_note=secret,
            )
        self.assert_private_input_hidden(raised.exception, secret)
        self.assertNotIn("SECRET_CONTROL_NOTE_6e12", raised.exception.json())

    def test_overlength_note_is_redacted_from_validation_error(self) -> None:
        run_id = new_run_id()
        marker = "SECRET_OVERLENGTH_NOTE_35da"
        secret = marker + ("x" * MAX_PRIVATE_REJECT_NOTE_LENGTH)
        with self.assertRaises(ValidationError) as raised:
            RejectReviewCommand(
                type="reject",
                run_id=run_id,
                review_id=review_id_for(run_id, "cluster"),
                cluster_id="cluster",
                expected_version=1,
                command_id=new_command_id(),
                reason_code=RejectReasonCode.OTHER,
                private_note=secret,
            )
        self.assert_private_input_hidden(raised.exception, secret)
        self.assertNotIn(marker, raised.exception.json())

    def test_missing_and_blank_other_notes_are_safely_rejected(self) -> None:
        run_id = new_run_id()
        values = dict(
            type="reject",
            run_id=run_id,
            review_id=review_id_for(run_id, "cluster"),
            cluster_id="cluster",
            expected_version=1,
            command_id=new_command_id(),
            reason_code=RejectReasonCode.OTHER,
        )
        with self.assertRaises(ValidationError) as missing:
            RejectReviewCommand(**values)
        with self.assertRaises(ValidationError) as blank:
            RejectReviewCommand(**values, private_note=" \t\n ")
        self.assertEqual(
            missing.exception.errors()[0]["input"]["private_note"],
            "[redacted]",
        )
        self.assertEqual(
            blank.exception.errors()[0]["input"]["private_note"],
            "[redacted]",
        )

    def test_discriminated_parser_accepts_only_approve_or_reject(self) -> None:
        run_id = new_run_id()
        common = {
            "run_id": run_id,
            "review_id": review_id_for(run_id, "cluster"),
            "cluster_id": "cluster",
            "expected_version": 1,
            "command_id": new_command_id(),
        }
        approve = parse_review_command({"type": "approve", **common})
        reject = parse_review_command(
            {
                "type": "reject",
                **common,
                "command_id": new_command_id(),
                "reason_code": "safety_concern",
            }
        )
        self.assertIsInstance(approve, ApproveReviewCommand)
        self.assertIsInstance(reject, RejectReviewCommand)

        invalid_payloads = (
            {**common},
            {"type": "edit_recommendation", **common},
            {"type": "approve", **common, "prompt": "forbidden"},
            {"type": "approve", **common, "run_id": "not-a-uuid"},
            {"type": "approve", **common, "expected_version": True},
            {"type": "approve", **common, "expected_version": 0},
            {"type": "approve", **common, "expected_version": -1},
            {
                "type": "reject",
                **common,
                "reason_code": "not-a-reason",
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    parse_review_command(payload)

    def test_equivalent_model_and_mapping_have_same_canonical_digest(self) -> None:
        run_id = new_run_id()
        raw = {
            "type": "reject",
            "run_id": run_id,
            "review_id": review_id_for(run_id, "cluster"),
            "cluster_id": "cluster",
            "expected_version": 1,
            "command_id": new_command_id(),
            "reason_code": "other",
            "private_note": "  ordinary   analyst note  ",
        }
        parsed_raw = parse_review_command(raw)
        parsed_model = parse_review_command(parsed_raw)
        raw_payload = parsed_raw.model_dump(mode="json")
        model_payload = parsed_model.model_dump(mode="json")
        self.assertEqual(raw_payload, model_payload)
        self.assertEqual(
            _payload_digest(raw_payload),
            _payload_digest(model_payload),
        )

    def test_all_unicode_control_categories_are_safely_rejected(self) -> None:
        run_id = new_run_id()
        common = {
            "type": "reject",
            "run_id": run_id,
            "review_id": review_id_for(run_id, "cluster"),
            "cluster_id": "cluster",
            "expected_version": 1,
            "command_id": new_command_id(),
            "reason_code": "other",
        }
        for character in ("\n", "\t", "\u007f", "\u0085", "\u200b"):
            secret = f"SECRET_NOTE{character}private"
            with self.subTest(character=repr(character)):
                with self.assertRaises(ValidationError) as raised:
                    parse_review_command({**common, "private_note": secret})
                self.assert_private_input_hidden(raised.exception, secret)
                self.assertNotIn("SECRET_NOTE", raised.exception.json())

    def test_unicode_note_is_accepted_and_whitespace_is_normalized(self) -> None:
        run_id = new_run_id()
        command = parse_review_command(
            {
                "type": "reject",
                "run_id": run_id,
                "review_id": review_id_for(run_id, "cluster"),
                "cluster_id": "cluster",
                "expected_version": 1,
                "command_id": new_command_id(),
                "reason_code": "other",
                "private_note": "  Élégant   evidence — valid!  ",
            }
        )
        self.assertEqual(command.private_note, "Élégant evidence — valid!")


if __name__ == "__main__":
    unittest.main()
