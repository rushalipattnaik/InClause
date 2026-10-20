import numpy as np
import pytest

from utils import analysis, ml_classifier


def separable_rows():
    safe = [f"This agreement commences on date {i} and the headings are for convenience only" for i in range(12)]
    risky = [f"A tariff surcharge and levy {i} applies to every delayed instalment of the balance" for i in range(12)]
    susp = [f"Everything paid {i} is non-refundable and forfeited in all cases for the member" for i in range(12)]
    return ([{"text": t, "label": "Safe"} for t in safe] + [{"text": t, "label": "Risky"} for t in risky]
            + [{"text": t, "label": "Suspicious"} for t in susp])


@pytest.mark.parametrize("kind", ["tfidf", "stacked"])
def test_word_based_models_learn_without_any_embedding_model(kind):
    model = ml_classifier.TextClassifier(kind, separable_rows())
    probs = model.predict_proba_many(["A tariff surcharge and levy applies to the delayed instalment",
                                      "This agreement commences on the date and the headings are for convenience"])
    assert max(probs[0], key=probs[0].get) == "Risky" and max(probs[1], key=probs[1].get) == "Safe"
    assert all(abs(sum(p.values()) - 1) < 1e-6 for p in probs)


def test_stacked_model_uses_the_rule_signals():
    model = ml_classifier.TextClassifier("stacked", separable_rows())
    probs = model.predict_proba_many(["Your fees are non-refundable and the sum is forfeited for any reason"])[0]
    assert max(probs, key=probs.get) == "Suspicious"


def test_embed_model_works_with_a_fake_embedding(fake_embeddings):
    model = ml_classifier.TextClassifier("embed", separable_rows(), fake_embeddings)
    probs = model.predict_proba_many(["A tariff surcharge and levy applies to the delayed instalment"])[0]
    assert max(probs, key=probs.get) == "Risky"


def test_invalid_use_is_rejected():
    with pytest.raises(ValueError):
        ml_classifier.TextClassifier("magic", separable_rows())
    with pytest.raises(ValueError):
        ml_classifier.TextClassifier("embed", separable_rows())


def test_grouped_splits_never_mix_a_contract_across_train_and_test():
    labels = ["Safe", "Risky", "Suspicious"] * 20
    groups = [f"contract_{i % 4}" for i in range(60)]
    splits = ml_classifier.make_splits(labels, groups, folds=5)
    assert len(splits) == 4
    for train, test in splits:
        assert not {groups[i] for i in train} & {groups[i] for i in test}
        assert len(set(train) & set(test)) == 0
    covered = sorted(i for _, test in splits for i in test)
    assert covered == list(range(60))


def test_without_groups_splits_are_stratified_and_complete():
    labels = ["Safe"] * 10 + ["Risky"] * 10 + ["Suspicious"] * 3
    splits = ml_classifier.make_splits(labels, None, folds=5)
    assert len(splits) == 3
    assert sorted(i for _, test in splits for i in test) == list(range(len(labels)))


def test_cross_val_probs_gives_one_prediction_per_row_and_uses_extra_training_rows():
    rows = separable_rows()
    probs = ml_classifier.cross_val_probs("tfidf", rows, extra_rows=rows[:6], folds=4)
    assert len(probs) == len(rows) and all(p is not None for p in probs)
    correct = sum(max(p, key=p.get) == r["label"] for p, r in zip(probs, rows, strict=True))
    assert correct / len(rows) > 0.9


def test_embedding_cache_avoids_recomputing(fake_embeddings):
    calls = []
    original = fake_embeddings.embed_documents
    fake_embeddings.embed_documents = lambda texts: calls.append(list(texts)) or original(texts)
    ml_classifier.embed(fake_embeddings, ["alpha clause text", "alpha clause text", "beta clause text"])
    ml_classifier.embed(fake_embeddings, ["alpha clause text", "gamma clause text"])
    assert calls == [["alpha clause text", "beta clause text"], ["gamma clause text"]]


def test_the_app_can_run_hybrid_with_a_word_model_and_no_embeddings(monkeypatch):
    monkeypatch.setenv("ML_KIND", "tfidf")
    ml_classifier.reset()
    out = analysis.analyze_all(["The deposit is non-refundable under any circumstances at all."], mode="hybrid")
    assert out[0]["label"] == "Suspicious"
    assert ml_classifier.get_model("tfidf") is not None
    ml_classifier.reset()


def test_unknown_kind_in_the_environment_falls_back(monkeypatch):
    monkeypatch.setenv("ML_KIND", "nonsense")
    assert ml_classifier.kind_from_env() == "auto"
    assert isinstance(np.zeros(1), np.ndarray)


def test_ml_only_mode_still_says_why_for_word_based_models(monkeypatch):
    ml_classifier.reset()
    ml_classifier.set_model(ml_classifier.TextClassifier("tfidf", separable_rows()))
    monkeypatch.setenv("ML_KIND", "tfidf")
    out = analysis.analyze_all(["Everything paid is non-refundable and forfeited in all cases for the member"], mode="ml")[0]
    assert out["label"] == "Suspicious"
    assert out["reasons"][0]["reason"] == "model estimate"
    assert "forfeited" in out["reasons"][1]["matched"] or "refundable" in out["reasons"][1]["matched"]
    ml_classifier.reset()


def test_explain_terms_are_empty_for_embeddings_and_safe_clauses(fake_embeddings):
    embed_model = ml_classifier.TextClassifier("embed", separable_rows(), fake_embeddings)
    assert embed_model.explain_terms(["some clause text"], ["Risky"]) == [[]]
    tfidf = ml_classifier.TextClassifier("tfidf", separable_rows())
    assert tfidf.explain_terms(["a clause"], ["NotALabel"]) == [[]]


def test_two_contracts_are_enough_for_a_strict_new_document_split():
    labels = ["Safe", "Risky"] * 10
    groups = ["rental"] * 10 + ["commercial"] * 10
    splits = ml_classifier.make_splits(labels, groups, folds=5)
    assert len(splits) == 2
    for train, test in splits:
        assert len({groups[i] for i in test}) == 1 and not {groups[i] for i in train} & {groups[i] for i in test}
