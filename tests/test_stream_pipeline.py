from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import OneHotEncoder
from numcompute_stream.tree import DecisionTreeClassifier


def test_pipeline_updates_encoder_and_tree_together():
    pipe = Pipeline([
        ("encode", OneHotEncoder()),
        ("tree", DecisionTreeClassifier(max_depth=1, buffer_size=4)),
    ])

    pipe.partial_fit([["blue"], ["blue"], ["red"], ["red"]], [0, 0, 1, 1])
    assert pipe.predict([["blue"], ["red"]]).tolist() == [0, 1]

    pipe.partial_fit([["amber"], ["amber"]], [1, 1])
    assert pipe.named_steps["encode"].get_feature_names_out() == ["x0_blue", "x0_red"]
    assert pipe.predict([["blue"], ["red"]]).tolist() == [0, 1]

    pipe.partial_fit([["red"], ["red"]], [0, 0])
    assert pipe.named_steps["encode"].get_feature_names_out() == [
        "x0_amber", "x0_blue", "x0_red"
    ]
    assert pipe.predict([["amber"], ["red"]]).tolist() == [1, 0]