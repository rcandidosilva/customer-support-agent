"""The lint exists because the model's strongest prior is to write to the customer."""

from __future__ import annotations

from conftest import make_packet

from support_agent.handoff_lint import has_errors, lint_packet


def fields(findings) -> set[str]:
    return {f.field.split("[")[0] for f in findings}


def messages(findings) -> str:
    return " ".join(f.message for f in findings)


def test_a_good_packet_is_clean():
    assert lint_packet(make_packet()) == []


def test_catches_customer_voice_in_the_summary():
    findings = lint_packet(
        make_packet(summary="Thanks for reaching out! We're sorry for the inconvenience.")
    )
    assert has_errors(findings)
    assert "summary" in fields(findings)


def test_catches_a_greeting():
    findings = lint_packet(
        make_packet(customer_goal="Hi there, the customer wants a refund.")
    )
    assert has_errors(findings)
    assert "greeting" in messages(findings)


def test_catches_customer_voice_in_a_list_field():
    findings = lint_packet(
        make_packet(
            next_steps=["Refund the charge.", "Let me know if you have any questions!"]
        )
    )
    assert has_errors(findings)
    assert any(f.field.startswith("next_steps[1]") for f in findings)


def test_ignores_customer_voice_in_the_draft_reply():
    """That field is *supposed* to sound like support prose - it is the one we may send."""
    findings = lint_packet(
        make_packet(
            suggested_reply_draft=(
                "Thanks for reaching out - I've refunded the duplicate charge and you "
                "should see it in 5-10 business days. Let me know if you have any "
                "questions."
            )
        )
    )
    assert findings == []


def test_catches_vague_next_steps():
    findings = lint_packet(
        make_packet(next_steps=["Investigate further and follow up as needed."])
    )
    assert has_errors(findings)
    assert "actionable" in messages(findings)


def test_catches_missing_next_steps():
    findings = lint_packet(make_packet(next_steps=[]))
    assert has_errors(findings)
    assert "next_steps" in fields(findings)


def test_requires_a_statement_about_what_was_already_sent():
    findings = lint_packet(make_packet(already_told_customer="   "))
    assert has_errors(findings)
    assert "already_told_customer" in fields(findings)


def test_warns_on_an_unscannable_subject_line():
    findings = lint_packet(make_packet(subject_line="Customer has a billing problem"))
    assert not has_errors(findings)  # a warning, not an error
    assert "subject_line" in fields(findings)


def test_warns_on_an_overlong_subject_line():
    findings = lint_packet(make_packet(subject_line="[Billing/High] " + "x" * 100))
    assert any("too long" in f.message for f in findings)


def test_warns_when_no_finding_cites_the_knowledge_base():
    findings = lint_packet(make_packet(findings=["The customer was charged twice."]))
    assert any("traceable" in f.message for f in findings)


def test_warns_on_a_bare_open_question():
    findings = lint_packet(make_packet(open_questions=["Was this a duplicate charge?"]))
    assert any("bare question" in f.message for f in findings)


def test_a_question_paired_with_a_reason_is_accepted():
    findings = lint_packet(
        make_packet(
            open_questions=[
                "Whether this was a duplicate charge or a seat addition, which the "
                "agent cannot determine without invoice line items."
            ]
        )
    )
    assert findings == []


def test_warns_on_stacked_hedging():
    findings = lint_packet(
        make_packet(why_escalated="It could potentially be outside the refund window.")
    )
    assert any("hedge" in f.message for f in findings)


def test_errors_sort_before_warnings():
    findings = lint_packet(
        make_packet(subject_line="no shape here", next_steps=[])
    )
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: 0 if s == "error" else 1)
