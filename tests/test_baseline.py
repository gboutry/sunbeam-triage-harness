from sunbeam_triage.core.baseline import (
    BaselineRun,
    BaselineSignature,
    compare_signals,
    match_successful_baselines,
)


def test_success_baseline_treats_shared_startup_signal_as_counterevidence():
    signature = BaselineSignature(
        step="sunbeam_deploy",
        sku="fcb-master-microstack-noble",
        addon="sunbeam_2024.1_beta",
        topology="four-node",
        versions=("k8s=1.32.13",),
    )
    runs = [
        BaselineRun(
            uuid="success-1",
            outcome="success",
            signature=signature,
            signals=("cni config not initialized",),
        ),
        BaselineRun(
            uuid="failure-1",
            outcome="failure",
            signature=signature,
            signals=("cni config not initialized",),
        ),
    ]

    matches = match_successful_baselines(signature, runs)
    comparison = compare_signals(["cni config not initialized"], matches)[0]

    assert [run.uuid for run in matches] == ["success-1"]
    assert comparison.observed_in_successes == 1
    assert comparison.interpretation.startswith("counterevidence")


def test_failure_specific_baseline_signal_remains_only_a_candidate():
    comparison = compare_signals(
        ["webhook connection refused"],
        [
            BaselineRun(
                uuid="success-1",
                outcome="success",
                signature=BaselineSignature("deploy", "sku", "addon", "topology"),
                signals=(),
            )
        ],
    )[0]

    assert comparison.observed_in_successes == 0
    assert "does not establish causality" in comparison.interpretation
