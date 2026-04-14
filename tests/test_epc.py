"""
Tests for metrics/epc.py: k-NN contextual EPC.
"""
from metrics.epc import EPCTrainer


def test_epc_train_does_not_raise(merge_history_df):
    trainer = EPCTrainer(k=5)
    trainer.train(merge_history_df)


def test_epc_predict_returns_float(merge_history_df):
    trainer = EPCTrainer(k=5)
    trainer.train(merge_history_df)
    pred, reliability, neighbours = trainer.predict_with_context(0.8, 0.7, 0.75)
    assert isinstance(pred, float)


def test_epc_reliability_between_0_and_1(merge_history_df):
    trainer = EPCTrainer(k=5)
    trainer.train(merge_history_df)
    _, reliability, _ = trainer.predict_with_context(0.8, 0.7, 0.75)
    assert 0.0 <= reliability <= 1.0


def test_epc_returns_correct_number_of_neighbours(merge_history_df):
    trainer = EPCTrainer(k=3)
    trainer.train(merge_history_df)
    _, _, neighbours = trainer.predict_with_context(0.8, 0.75, 0.7)
    assert len(neighbours) == 3


def test_epc_neighbour_has_expected_keys(merge_history_df):
    trainer = EPCTrainer(k=5)
    trainer.train(merge_history_df)
    _, _, neighbours = trainer.predict_with_context(0.8, 0.75, 0.7)
    for key in ("psc", "fsc", "rsc", "improvement", "distance"):
        assert key in neighbours[0]


def test_epc_save_and_load(merge_history_df, tmp_path):
    trainer = EPCTrainer(k=5)
    trainer.train(merge_history_df)
    path = tmp_path / "epc.pkl"
    trainer.save(str(path))
    loaded = EPCTrainer(k=5)
    loaded.load(str(path))
    pred_orig, _, _ = trainer.predict_with_context(0.8, 0.7, 0.75)
    pred_loaded, _, _ = loaded.predict_with_context(0.8, 0.7, 0.75)
    assert abs(pred_orig - pred_loaded) < 1e-9