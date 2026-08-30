import pytest

from voice_cloning.text import normalise_script, split_sentences


class TestNormaliseScript:
    def test_collapses_newlines_and_runs_of_whitespace(self):
        assert normalise_script("one\n  two\n\n three") == "one two three"

    def test_strips_surrounding_whitespace(self):
        assert normalise_script("  hello  ") == "hello"

    def test_blank_input_becomes_empty(self):
        assert normalise_script("   \n\n  ") == ""


class TestSplitSentences:
    def test_blank_input_yields_no_sentences(self):
        assert split_sentences("") == []
        assert split_sentences("   \n ") == []

    def test_single_sentence_is_returned_whole(self):
        assert split_sentences("Just one sentence.") == ["Just one sentence."]

    @pytest.mark.parametrize("use_nltk", [True, False])
    def test_splits_on_terminal_punctuation(self, use_nltk):
        sentences = split_sentences(
            "First one. Second one! Third one?", use_nltk=use_nltk
        )
        assert len(sentences) == 3
        assert sentences[0] == "First one."
        assert sentences[-1] == "Third one?"

    def test_regex_fallback_handles_multiline_script(self):
        script = """
        If you are looking to invest, you may wonder about off-market deals.
        They can be a great resource. Do some digging first.
        """
        sentences = split_sentences(script, use_nltk=False)
        assert len(sentences) == 3
        # normalisation must have removed the indentation
        assert all("\n" not in s for s in sentences)
        assert not any(s.startswith(" ") for s in sentences)

    def test_no_empty_strings_in_output(self):
        sentences = split_sentences("A.  B.   C.", use_nltk=False)
        assert all(s.strip() for s in sentences)

    def test_text_without_terminal_punctuation_is_one_sentence(self):
        assert split_sentences("no punctuation here", use_nltk=False) == [
            "no punctuation here"
        ]
