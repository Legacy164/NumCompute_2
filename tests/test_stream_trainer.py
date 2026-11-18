from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def test_trainer_scores_before_updating():
    pipe = Pipeline([("tree", DecisionTreeClassifier(max_depth=1, buffer_size=2))])
    trainer = StreamTrainer(pipe)

    first = trainer.fit_chunk([[0], [1]], [0, 1])
    assert first["accuracy"] is None

    second = trainer.fit_chunk([[0], [1]], [1, 0])
    assert second["accuracy"] == 0.0
    assert pipe.predict([[0], [1]]).tolist() == [1, 0]

    third = trainer.fit_chunk([[0]], [1])
    assert third["accuracy"] == 1.0
    assert third["cumulative_accuracy"] == 1 / 3
    assert third["buffer_bytes"] > 0

def test_trainer_accepts_a_bare_tree():
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=1, buffer_size=2))

    assert trainer.fit_chunk([[0], [1]], [0, 1])["accuracy"] is None
    entry = trainer.fit_chunk([[0], [1]], [0, 1])
    assert entry["accuracy"] == 1.0
    assert entry["precision"] == entry["recall"] == entry["f1"] == 1.0


def test_score_chunk_returns_accuracy_without_training():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=2)
    trainer = StreamTrainer(tree)
    assert trainer.score_chunk([[0], [1]], [0, 1]) is None

    tree.fit([[0], [1]], [0, 1])
    assert trainer.score_chunk([[0], [1]], [0, 0]) == 0.5
    assert trainer.history == []