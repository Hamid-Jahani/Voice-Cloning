import pytest

from voice_cloning.cli import build_parser
from voice_cloning.config import (
    DEFAULT_SAMPLE_RATE,
    ControlledSynthesisConfig,
    LongFormConfig,
    ModelConfig,
    SynthesisConfig,
    sample_rate,
)


class TestConfigDefaults:
    """Defaults must match the values hard-coded in the original notebook."""

    def test_synthesis_defaults(self):
        config = SynthesisConfig()
        assert config.text_temp == 0.7
        assert config.waveform_temp == 0.7

    def test_controlled_defaults(self):
        config = ControlledSynthesisConfig()
        assert config.semantic_temp == 0.7
        assert config.semantic_top_k == 50
        assert config.semantic_top_p == 0.95
        assert config.fine_temp == 0.5

    def test_long_form_defaults(self):
        config = LongFormConfig()
        assert config.gen_temp == 0.6
        assert config.min_eos_p == 0.05
        assert config.silence_seconds == 0.15


class TestConfigValidation:
    @pytest.mark.parametrize("temp", [0.0, -0.1, 1.5])
    def test_rejects_out_of_range_text_temp(self, temp):
        with pytest.raises(ValueError, match="text_temp"):
            SynthesisConfig(text_temp=temp)

    @pytest.mark.parametrize("temp", [0.0, -0.1, 1.5])
    def test_rejects_out_of_range_waveform_temp(self, temp):
        with pytest.raises(ValueError, match="waveform_temp"):
            SynthesisConfig(waveform_temp=temp)

    def test_accepts_temperature_of_one(self):
        assert SynthesisConfig(text_temp=1.0).text_temp == 1.0

    def test_rejects_negative_silence(self):
        with pytest.raises(ValueError, match="silence_seconds"):
            LongFormConfig(silence_seconds=-1)

    def test_rejects_non_positive_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            LongFormConfig(sample_rate=0)

    def test_configs_are_immutable(self):
        with pytest.raises(Exception):
            SynthesisConfig().text_temp = 0.5


class TestSampleRate:
    def test_returns_a_positive_int_without_bark_installed(self):
        rate = sample_rate()
        assert isinstance(rate, int)
        assert rate > 0

    def test_falls_back_to_documented_default(self):
        assert DEFAULT_SAMPLE_RATE == 24_000


class TestModelConfig:
    def test_resolves_a_device_without_raising(self):
        assert ModelConfig().resolve_device() in {"cuda", "cpu"}

    def test_explicit_device_is_respected(self):
        assert ModelConfig(device="cpu").resolve_device() == "cpu"


class TestCli:
    """The parser must work with no models installed."""

    def test_clone_requires_a_reference(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["clone"])

    def test_clone_parses_reference_and_output(self, tmp_path):
        args = build_parser().parse_args(
            ["clone", "--reference", "ref.wav", "--output", "v.npz"]
        )
        assert args.command == "clone"
        assert args.reference.name == "ref.wav"

    def test_speak_defaults_match_synthesis_config(self):
        args = build_parser().parse_args(["speak", "--voice", "v.npz", "--text", "hi"])
        assert args.text_temp == SynthesisConfig().text_temp
        assert args.waveform_temp == SynthesisConfig().waveform_temp
        assert args.controlled is False

    def test_narrate_defaults_match_long_form_config(self):
        args = build_parser().parse_args(
            ["narrate", "--voice", "v.npz", "--script", "s.txt"]
        )
        assert args.gen_temp == LongFormConfig().gen_temp
        assert args.silence == LongFormConfig().silence_seconds

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_command_is_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["frobnicate"])

    @pytest.mark.parametrize("command", ["clone", "speak", "narrate", "score"])
    def test_every_subcommand_has_help(self, command, capsys):
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args([command, "--help"])
        assert excinfo.value.code == 0
        assert capsys.readouterr().out
