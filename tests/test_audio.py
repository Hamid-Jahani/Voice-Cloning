import numpy as np
import pytest

from voice_cloning.audio import make_silence, stitch, write_wav


class TestMakeSilence:
    def test_length_is_seconds_times_rate(self):
        assert len(make_silence(0.15, 24_000)) == 3_600

    def test_length_truncates_toward_zero(self):
        # 0.1 * 24000 = 2400.0000000000005 in binary floating point
        assert len(make_silence(0.1, 24_000)) == 2_400

    def test_all_samples_are_zero(self):
        assert not make_silence(0.05, 24_000).any()

    def test_zero_seconds_gives_empty_array(self):
        assert len(make_silence(0, 24_000)) == 0

    @pytest.mark.parametrize("seconds,rate", [(-0.1, 24_000), (0.1, 0), (0.1, -1)])
    def test_rejects_invalid_arguments(self, seconds, rate):
        with pytest.raises(ValueError):
            make_silence(seconds, rate)


class TestStitch:
    def test_no_clips_gives_empty_array(self):
        assert len(stitch([], make_silence(0.1, 100))) == 0

    def test_single_clip_is_returned_without_padding(self):
        clip = np.ones(10)
        assert np.array_equal(stitch([clip], make_silence(0.1, 100)), clip)

    def test_silence_is_inserted_between_clips(self):
        clip = np.ones(10)
        silence = np.zeros(5)
        result = stitch([clip, clip], silence)
        assert len(result) == 25  # 10 + 5 + 10
        assert not result[10:15].any()

    def test_does_not_end_in_silence(self):
        """The notebook appended silence after every clip, leaving dead air."""
        clip = np.ones(10)
        result = stitch([clip, clip, clip], np.zeros(5))
        assert result[-1] == 1
        assert len(result) == 40  # 3*10 + 2*5, not 3*10 + 3*5

    def test_gap_count_is_one_fewer_than_clip_count(self):
        clips = [np.ones(4) for _ in range(5)]
        result = stitch(clips, np.zeros(3))
        assert len(result) == 5 * 4 + 4 * 3

    def test_accepts_a_generator(self):
        result = stitch((np.ones(2) for _ in range(3)), np.zeros(1))
        assert len(result) == 3 * 2 + 2 * 1


class TestWriteWav:
    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "out.wav"
        written = write_wav(target, np.zeros(100, dtype=np.float32), 24_000)
        assert written.exists()
        assert written == target

    def test_roundtrips_through_scipy(self, tmp_path):
        from scipy.io.wavfile import read

        audio = np.linspace(-0.5, 0.5, 200, dtype=np.float32)
        path = write_wav(tmp_path / "a.wav", audio, 24_000)
        rate, loaded = read(path)
        assert rate == 24_000
        assert np.allclose(loaded, audio, atol=1e-6)
