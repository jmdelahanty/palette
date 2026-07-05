from fisheye.shared.encoder_tags import parse_encoder_comment


def test_parse_encoder_comment_extracts_resolution_dimensions() -> None:
    parsed = parse_encoder_comment("codec=hevc; res=4512x4512; fps=60")
    assert parsed.get("encoder_res") == "4512x4512"
    assert parsed.get("encoder_res_width") == 4512
    assert parsed.get("encoder_res_height") == 4512


def test_parse_encoder_comment_handles_prefixed_codec_key() -> None:
    parsed = parse_encoder_comment("nvenc codec=hevc; preset=p1; tuning=ll; rc=vbr")
    assert parsed.get("encoder_name") == "nvenc"
    assert parsed.get("encoder_codec") == "hevc"
    assert parsed.get("encoder_preset") == "p1"
    assert parsed.get("encoder_tuning") == "ll"
    assert parsed.get("encoder_rc") == "vbr"
