import math
from types import SimpleNamespace

import pytest

from evaluation.perplexity import nll_from_prompt_logprobs, perplexity, windows


def test_windows_drop_ragged_tail():
    assert windows(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert windows(list(range(10)), 4, max_windows=1) == [[0, 1, 2, 3]]
    assert windows([1, 2], 4) == []


def test_nll_scores_only_from_start():
    ids = [5, 6, 7, 8]
    lp = [None] + [{t: SimpleNamespace(logprob=math.log(0.5)), 99: SimpleNamespace(logprob=-9)}
                   for t in ids[1:]]
    nll, n = nll_from_prompt_logprobs(ids, lp, start=2)  # completion = last 2 tokens
    assert n == 2 and nll == pytest.approx(2 * math.log(2))
    assert perplexity(nll, n) == pytest.approx(2.0)  # p = 1/2 per token
    with pytest.raises(ValueError):
        nll_from_prompt_logprobs(ids, lp, start=0)
