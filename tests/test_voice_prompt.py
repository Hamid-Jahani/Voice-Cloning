import numpy as np
import pytest

from voice_cloning.voice_prompt import (
    COARSE_CODEBOOKS,
    REQUIRED_KEYS,
    load_voice_prompt,
    save_voice_prompt,
)


@pytest.fixture
def codes():
    """An EnCodec-shaped code matrix: 8 codebooks x 50 frames."""
    return np.arange(8 * 50, dtype=np.int64).reshape(8, 50)


@pytest.fixture
def semantic_tokens():
    return np.arange(120, dtype=np.int64)


class TestSaveVoicePrompt:
    def test_writes_all_keys_bark_requires(self, tmp_path, codes, semantic_tokens):
        path = save_voice_prompt(tmp_path / "v.npz", codes=codes, semantic_tokens=semantic_tokens)
        with np.load(path) as data:
            assert set(REQUIRED_KEYS).issubset(data.files)

    def test_coarse_prompt_is_the_first_two_codebooks(self, tmp_path, codes, semantic_tokens):
        path = save_voice_prompt(tmp_path / "v.npz", codes=codes, semantic_tokens=semantic_tokens)
        with np.load(path) as data:
            assert data["coarse_prompt"].shape == (COARSE_CODEBOOKS, 50)
            assert np.array_equal(data["coarse_prompt"], codes[:COARSE_CODEBOOKS, :])

    def test_fine_prompt_is_the_full_code_matrix(self, tmp_path, codes, semantic_tokens):
        path = save_voice_prompt(tmp_path / "v.npz", codes=codes, semantic_tokens=semantic_tokens)
        with np.load(path) as data:
            assert np.array_equal(data["fine_prompt"], codes)

    def test_creates_parent_directories(self, tmp_path, codes, semantic_tokens):
        target = tmp_path / "deep" / "nested" / "v.npz"
        written = save_voice_prompt(target, codes=codes, semantic_tokens=semantic_tokens)
        assert written.exists()

    def test_reports_the_npz_path_when_suffix_omitted(self, tmp_path, codes, semantic_tokens):
        """numpy appends .npz itself; the returned path must reflect that."""
        written = save_voice_prompt(tmp_path / "voice", codes=codes, semantic_tokens=semantic_tokens)
        assert written.suffix == ".npz"
        assert written.exists()

    def test_rejects_one_dimensional_codes(self, tmp_path, semantic_tokens):
        with pytest.raises(ValueError, match="2-D"):
            save_voice_prompt(tmp_path / "v.npz", codes=np.arange(10), semantic_tokens=semantic_tokens)

    def test_rejects_too_few_codebooks(self, tmp_path, semantic_tokens):
        with pytest.raises(ValueError, match="codebooks"):
            save_voice_prompt(
                tmp_path / "v.npz",
                codes=np.zeros((1, 50)),
                semantic_tokens=semantic_tokens,
            )


class TestLoadVoicePrompt:
    def test_roundtrip_preserves_arrays(self, tmp_path, codes, semantic_tokens):
        path = save_voice_prompt(tmp_path / "v.npz", codes=codes, semantic_tokens=semantic_tokens)
        prompt = load_voice_prompt(path)
        assert np.array_equal(prompt.fine_prompt, codes)
        assert np.array_equal(prompt.semantic_prompt, semantic_tokens)
        assert np.array_equal(prompt.coarse_prompt, codes[:COARSE_CODEBOOKS, :])

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_voice_prompt(tmp_path / "absent.npz")

    def test_npz_without_required_keys_is_rejected(self, tmp_path):
        path = tmp_path / "wrong.npz"
        np.savez(path, something_else=np.zeros(3))
        with pytest.raises(ValueError, match="missing array"):
            load_voice_prompt(path)

    def test_error_names_every_missing_key(self, tmp_path, semantic_tokens):
        path = tmp_path / "partial.npz"
        np.savez(path, semantic_prompt=semantic_tokens)
        with pytest.raises(ValueError) as excinfo:
            load_voice_prompt(path)
        assert "coarse_prompt" in str(excinfo.value)
        assert "fine_prompt" in str(excinfo.value)
