"""Wikipedia is asked for a page once per sweep, and asked no faster than it allows.

The two datasets that drive live Wikipedia replay one question set under every
seed and every memory arm, so the same titles are fetched over and over, by as
many worker processes as the node has cores. Both halves of that are fixed here:
what has been fetched is kept, and what has not is fetched through a window every
process shares.
"""

import json
import time

import pytest

from tasks.envs.wiki_cache import PageCache, RequestWindow


@pytest.fixture
def cache(tmp_path):
    return PageCache(tmp_path / 'pages')


@pytest.fixture
def window(tmp_path):
    return RequestWindow(tmp_path / 'window', min_interval=0.25)


# ── the cache ─────────────────────────────────────────────────────────────────

def test_a_title_not_yet_fetched_is_a_miss(cache):
    assert cache.get('Telemundo') is None


def test_a_stored_page_is_returned_for_the_same_title(cache):
    cache.put('Telemundo', {'content': 'the page text', 'url': 'https://x/Telemundo'})

    assert cache.get('Telemundo') == {
        'content': 'the page text', 'url': 'https://x/Telemundo'
    }


def test_a_stored_page_is_visible_to_a_second_cache_on_the_same_directory(tmp_path):
    """The workers are separate processes, so nothing in memory is shared."""
    PageCache(tmp_path / 'pages').put('Telemundo', {'content': 'text', 'url': 'u'})

    assert PageCache(tmp_path / 'pages').get('Telemundo') == {
        'content': 'text', 'url': 'u'
    }


def test_titles_differing_only_in_case_or_punctuation_are_separate_entries(cache):
    """Wikipedia titles are case sensitive past the first letter."""
    cache.put('Mercury (planet)', {'content': 'the planet', 'url': 'u'})

    assert cache.get('Mercury') is None
    assert cache.get('mercury (planet)') is None


def test_a_title_that_is_not_a_usable_filename_is_still_stored_and_found(cache):
    """Search terms come from the model, so any character can arrive in one."""
    awkward = 'AC/DC: "Back in Black" \\ 100% ../../etc/passwd\n'
    cache.put(awkward, {'content': 'the page text', 'url': 'u'})

    assert cache.get(awkward) == {'content': 'the page text', 'url': 'u'}


def test_an_entry_written_only_half_way_is_a_miss_rather_than_a_crash(cache, tmp_path):
    """A job killed mid-write leaves the file behind for the next one to read."""
    cache.put('Telemundo', {'content': 'the page text', 'url': 'u'})
    entry = next((tmp_path / 'pages').iterdir())
    entry.write_text('{"content": "the page te', encoding='utf-8')

    assert cache.get('Telemundo') is None


def test_a_write_that_dies_part_way_leaves_the_entry_it_would_have_replaced(
    cache, monkeypatch
):
    """Every worker writes the same directory, so a reader can arrive mid-write.

    An entry is staged and renamed into place rather than written where it will
    be read, so what a reader sees is one whole entry or the last whole one.
    """
    cache.put('Telemundo', {'content': 'the page text', 'url': 'u'})

    def dies(*_, **__):
        raise OSError('no space left on device')

    monkeypatch.setattr(json, 'dump', dies)
    cache.put('Telemundo', {'content': 'a longer page text', 'url': 'u'})

    assert cache.get('Telemundo') == {'content': 'the page text', 'url': 'u'}


def test_a_write_that_dies_part_way_leaves_no_staged_file_behind(cache, tmp_path):
    """A sweep makes tens of thousands of these, and `data/` is not swept up."""
    def dies(*_, **__):
        raise OSError('no space left on device')

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(json, 'dump', dies)
        cache.put('Telemundo', {'content': 'the page text', 'url': 'u'})

    assert list((tmp_path / 'pages').iterdir()) == []


def test_a_cache_directory_that_cannot_be_written_does_not_fail_the_search(tmp_path):
    """A read-only filesystem costs the saving, not the run."""
    unwritable = tmp_path / 'ro'
    unwritable.mkdir()
    unwritable.chmod(0o500)
    try:
        cache = PageCache(unwritable / 'pages')
        cache.put('Telemundo', {'content': 'text', 'url': 'u'})

        assert cache.get('Telemundo') is None
    finally:
        unwritable.chmod(0o700)


# ── the window ────────────────────────────────────────────────────────────────

def test_the_first_request_of_a_run_does_not_wait(window):
    assert window.reserve() == pytest.approx(0.0, abs=0.01)


def test_the_request_after_one_waits_the_interval(window):
    window.reserve()

    assert window.reserve() == pytest.approx(0.25, abs=0.02)


def test_the_waits_of_many_requests_do_not_land_on_the_same_moment(window):
    """Each caller takes its own slot, so 40 workers spread rather than burst."""
    waits = [window.reserve() for _ in range(4)]

    assert waits == pytest.approx([0.0, 0.25, 0.5, 0.75], abs=0.03)


def test_a_window_reopens_once_its_interval_has_passed(tmp_path):
    window = RequestWindow(tmp_path / 'window', min_interval=0.01)
    window.reserve()
    time.sleep(0.05)

    assert window.reserve() == pytest.approx(0.0, abs=0.01)


def test_a_second_window_on_the_same_file_shares_the_slots(tmp_path):
    """The point of the file: the workers are processes and share nothing else."""
    RequestWindow(tmp_path / 'window', min_interval=0.25).reserve()

    assert RequestWindow(tmp_path / 'window', min_interval=0.25).reserve() == (
        pytest.approx(0.25, abs=0.02)
    )


def test_a_refusal_defers_every_process_not_only_the_one_that_met_it(tmp_path):
    """The 429 that broke the sweep: one worker is told to wait 11s, the other
    39 know nothing of it and keep the refusal alive by asking again."""
    met_it = RequestWindow(tmp_path / 'window', min_interval=0.25)
    everyone_else = RequestWindow(tmp_path / 'window', min_interval=0.25)

    met_it.defer(5.0)

    assert everyone_else.reserve() == pytest.approx(5.0, abs=0.05)


def test_a_deferral_shorter_than_the_wait_already_owed_does_not_shorten_it(window):
    window.defer(5.0)

    window.defer(0.1)

    assert window.reserve() == pytest.approx(5.0, abs=0.05)


def test_a_window_file_that_cannot_be_read_costs_the_spacing_not_the_run(tmp_path):
    path = tmp_path / 'window'
    path.write_text('not a number', encoding='utf-8')

    assert RequestWindow(path, min_interval=0.25).reserve() == pytest.approx(0.0, abs=0.01)


def test_a_window_on_an_unwritable_directory_still_answers(tmp_path):
    unwritable = tmp_path / 'ro'
    unwritable.mkdir()
    unwritable.chmod(0o500)
    try:
        assert RequestWindow(unwritable / 'window', min_interval=0.25).reserve() == (
            pytest.approx(0.0, abs=0.01)
        )
    finally:
        unwritable.chmod(0o700)


def test_a_disabled_window_never_waits(tmp_path):
    window = RequestWindow(tmp_path / 'window', min_interval=0)

    assert [window.reserve() for _ in range(3)] == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
