import pytest

from gqa.tuning import balanced_accuracy, choose_threshold


def test_separable_scores_pick_the_gap():
    conf = [0.9, 0.8, 0.7, 0.2, 0.1]
    ans = [True, True, True, False, False]
    c = choose_threshold(conf, ans)
    assert c.threshold == 0.7 and c.balanced_accuracy == 1.0
    assert c.refuse_unanswerable == 1.0 and c.answer_answerable == 1.0


def test_ties_go_to_lower_threshold_and_range():
    conf = [1.0, 1.0, 1.0, 1.0]
    ans = [True, False, True, False]
    c = choose_threshold(conf, ans)
    assert c.threshold == 0.0 and c.balanced_accuracy == 0.5
    assert 0.0 <= choose_threshold([1.5, -2.0], [True, False]).threshold <= 1.0


def test_errors():
    with pytest.raises(ValueError):
        balanced_accuracy([0.5], [True], 0.5)
    with pytest.raises(ValueError):
        choose_threshold([0.5], [True, False])


def test_downstream_refusals_lower_the_needed_threshold():
    conf = [0.9, 0.8, 0.6, 0.85, 0.5]
    ans = [True, True, True, False, False]
    # Retrieval alone cannot separate question 4 (an unanswerable question with high confidence).
    alone = choose_threshold(conf, ans)
    # The model refuses question 4 itself, so a low gate catching question 5 suffices.
    joint = choose_threshold(conf, ans, downstream_refused=[False, False, False, True, False])
    assert joint.balanced_accuracy == 1.0 and joint.threshold == 0.6
    assert alone.balanced_accuracy < 1.0
    with pytest.raises(ValueError):
        choose_threshold(conf, ans, downstream_refused=[True])
