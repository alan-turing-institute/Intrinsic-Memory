"""What the explorer asks Wikipedia for, given what it has already been told.

A sweep replays one question set under every seed and every memory arm, so a
title reached in one run is reached again in the next ninety-nine. Measured over
the September run: 22,432 FEVER searches covered 1,659 distinct titles and 20,393
HotpotQA searches covered 2,243, and 87% and 80% of them respectively came back
`WikipediaUnavailable`.
"""

from types import SimpleNamespace

import pytest
import wikipedia

from tasks.envs import utils
from tasks.envs.utils import LangChainWiki, WikipediaUnavailable


@pytest.fixture(autouse=True)
def waits(monkeypatch):
    """The seconds the retry would have slept, without spending them."""
    slept = []
    monkeypatch.setattr(utils, 'time', SimpleNamespace(sleep=slept.append))
    return slept


@pytest.fixture
def explorer(tmp_path):
    return LangChainWiki(cache_dir=tmp_path / 'pages')


def another(tmp_path):
    """A second explorer sharing the first's cache, as a second worker would."""
    return LangChainWiki(cache_dir=tmp_path / 'pages')


def counted(monkeypatch, content='Telemundo is a network.\n\nIt mentions zebras.'):
    """`wikipedia.page`, recording the titles it was actually asked for."""
    asked = []

    class Page:
        def __init__(self, title):
            self.content = content
            self.url = f'https://en.wikipedia.org/wiki/{title}'

    def page(title, **kwargs):
        asked.append(title)
        return Page(title)

    monkeypatch.setattr(wikipedia, 'page', page)
    return asked


def test_a_page_already_fetched_is_not_fetched_again(explorer, monkeypatch):
    asked = counted(monkeypatch)

    explorer.search('Telemundo')
    explorer.search('Telemundo')

    assert asked == ['Telemundo'], f'asked Wikipedia {len(asked)} times for one title'


def test_a_page_fetched_by_one_worker_is_not_fetched_by_the_next(tmp_path, monkeypatch):
    """This is the 13.5x: the workers are processes, so only the disk is shared."""
    asked = counted(monkeypatch)
    another(tmp_path).search('Telemundo')

    another(tmp_path).search('Telemundo')

    assert asked == ['Telemundo']


def test_a_cached_page_answers_with_what_the_live_fetch_answered(tmp_path, monkeypatch):
    counted(monkeypatch)
    live = another(tmp_path).search('Telemundo')

    assert another(tmp_path).search('Telemundo') == live


def test_a_cached_page_can_still_be_looked_up_in(tmp_path, monkeypatch):
    counted(monkeypatch)
    another(tmp_path).search('Telemundo')

    cached = another(tmp_path)
    cached.search('Telemundo')

    assert 'zebras' in cached.lookup('zebras')


def test_a_page_that_does_not_exist_is_not_asked_for_twice(explorer, monkeypatch):
    """An absent page costs two requests - the page and the similar titles - and
    the model asks for the same wrong title under every seed."""
    asked = []

    def page(title, **kwargs):
        asked.append(title)
        raise wikipedia.PageError('no such page')

    monkeypatch.setattr(wikipedia, 'page', page)
    monkeypatch.setattr(wikipedia, 'search', lambda title, **kwargs: ['Something else'])

    first = explorer.search('Nonexistent Title')
    second = explorer.search('Nonexistent Title')

    assert asked == ['Nonexistent Title']
    assert first == second
    assert 'Could not find [Nonexistent Title]' in second


def test_a_search_that_could_not_reach_wikipedia_is_not_cached(explorer, monkeypatch):
    """Caching a transport fault would make one bad minute permanent."""
    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: (_ for _ in ()).throw(
        ValueError('Expecting value: line 1 column 1 (char 0)')))

    with pytest.raises(WikipediaUnavailable):
        explorer.search('Telemundo')

    asked = counted(monkeypatch)
    explorer.search('Telemundo')

    assert asked == ['Telemundo'], 'the failure was cached and the page never arrived'


def test_a_disambiguation_is_cached_with_its_alternatives(explorer, monkeypatch):
    asked = []

    def page(title, **kwargs):
        asked.append(title)
        raise wikipedia.DisambiguationError('ambiguous', [])

    monkeypatch.setattr(wikipedia, 'page', page)
    monkeypatch.setattr(wikipedia, 'search', lambda title, **kwargs: ['Mercury (planet)'])

    explorer.search('Mercury')
    second = explorer.search('Mercury')

    assert asked == ['Mercury']
    assert 'Mercury (planet)' in second


def test_a_live_fetch_waits_for_the_shared_window(explorer, monkeypatch, waits):
    counted(monkeypatch)

    explorer.search('Telemundo')
    explorer.search('Anne Rice (author)')

    assert waits, 'nothing waited, so 40 workers would burst'
    assert sum(waits) > 0


def test_a_cached_fetch_does_not_wait(tmp_path, monkeypatch, waits):
    counted(monkeypatch)
    another(tmp_path).search('Telemundo')
    waits.clear()

    another(tmp_path).search('Telemundo')

    assert waits == [], 'a cache hit spent the rate limit it did not need'


def test_a_refusal_defers_the_window_by_what_it_asked_for(explorer, monkeypatch):
    """The 429 names a wait. Before, only the worker that met it waited, and the
    other 39 kept asking, which is what kept the refusal alive for nine hours."""
    refused = utils.WikipediaRefused(429, asked=11.0)
    monkeypatch.setattr(wikipedia, 'page', lambda title, **kwargs: (_ for _ in ()).throw(
        refused))

    with pytest.raises(WikipediaUnavailable):
        explorer.search('Telemundo')

    assert explorer.window.reserve() >= 10.0, 'the wait was not shared'


def test_an_explorer_given_no_cache_still_searches(monkeypatch):
    """The offline suite and the contract tests build one with no arguments."""
    counted(monkeypatch)

    assert 'Telemundo' in LangChainWiki(cache_dir=None).search('Telemundo')
