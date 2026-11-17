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