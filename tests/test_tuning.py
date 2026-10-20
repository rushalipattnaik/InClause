from eval import run_eval


def rows_and_hits(positive_score, spurious_score, count=8):
    rows, hits = [], []
    for _ in range(count):
        rows.append({"topic": "arbitration"})
        hits.append([("arbitration", positive_score), ("other", positive_score - 0.12)])
    for _ in range(count):
        rows.append({"topic": "none"})
        hits.append([("other", spurious_score)])
    return rows, hits


def test_a_setting_that_hides_almost_everything_cannot_win():
    rows, hits = rows_and_hits(positive_score=0.55, spurious_score=0.30)
    _, threshold, _ = run_eval.best_setting(rows, hits)
    assert 0.30 < threshold <= 0.55
    found, _, empty = run_eval.evaluate_setting(rows, hits, 0.65, 0.15)
    assert found == 0.0 and run_eval.harmonic_mean([found, 1.0, empty]) == 0.0


def test_threshold_follows_where_the_good_matches_sit():
    rows, hits = rows_and_hits(positive_score=0.45, spurious_score=0.22)
    _, threshold, _ = run_eval.best_setting(rows, hits)
    assert 0.22 < threshold <= 0.45


def test_harmonic_mean_punishes_a_collapsed_number():
    assert run_eval.harmonic_mean([0.9, 0.9, 0.9]) > 0.89
    assert run_eval.harmonic_mean([1.0, 1.0, 0.0]) == 0.0
    assert run_eval.harmonic_mean([]) == 0.0
