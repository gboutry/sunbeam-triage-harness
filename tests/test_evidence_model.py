from sunbeam_triage.core.evidence_model import (
    EvidenceObservation,
    EvidenceProvenance,
    SourceRef,
    confidence_for_read,
)


def test_evidence_observation_records_provenance_and_targeted_read():
    observation = EvidenceObservation(
        source=SourceRef(
            path="generated/sunbeam/output.log",
            line_start=120,
            line_end=140,
            role="primary-log",
        ),
        category="failure_surface",
        excerpt="wait timed out after 1799.999s",
        provenance=EvidenceProvenance(
            origin="step_profile",
            selector="sunbeam_deploy.cluster_join_timeout",
            read_class="targeted_read",
            tool_name="get_artifact_file",
        ),
    )

    assert observation.confidence == "high"
    assert observation.is_targeted is True
    assert observation.to_prompt_line().startswith(
        "- [failure_surface/high/targeted_read] "
        "generated/sunbeam/output.log:120-140:"
    )


def test_confidence_for_read_keeps_broad_search_lower_than_targeted_read():
    assert confidence_for_read("broad_search") == "low"
    assert confidence_for_read("targeted_search") == "medium"
    assert confidence_for_read("targeted_read") == "high"
