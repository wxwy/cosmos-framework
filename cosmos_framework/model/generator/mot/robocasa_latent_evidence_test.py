"""B1 的因果 endpoint 与无参数 visual96 CPU 合同。"""

import h5py
import numpy as np
import pytest
import torch

from cosmos_framework.model.generator.mot.robocasa_latent_evidence import (
    LEFT_CAMERA,
    WRIST_CAMERA,
    RoboCasaLatentReader,
    causal_endpoint_index,
    latent_to_visual96,
)


def write_cache(path, frames=67, episode_id="CloseFridge/shard0/episode0"):
    indices = np.arange(0, frames, 4, dtype=np.int64)
    if indices[-1] != frames - 1:
        indices = np.append(indices, frames - 1)
    with h5py.File(path, "w") as cache:
        cache.attrs["episode_id"] = episode_id
        cache.attrs["frame_count"] = frames
        cache.attrs["temporal_compression_factor"] = 4
        cache.attrs["source_frame_to_latent_policy"] = "causal_endpoint"
        for camera in (LEFT_CAMERA, WRIST_CAMERA):
            cache.create_dataset(f"indices/latent_source_frame_indices/{camera}", data=indices)
            cache.create_dataset(f"valid/{camera}", data=np.ones(len(indices), dtype=bool))
            cache.create_dataset(
                f"latents/{camera}",
                data=np.broadcast_to(indices[:, None, None, None], (len(indices), 48, 16, 16)).astype(np.float16),
            )
        cache.create_dataset("robot/action", data=np.zeros((frames, 12), dtype=np.float32))
    return path


def read_cache(path, frames=67, episode_id="CloseFridge/shard0/episode0"):
    return RoboCasaLatentReader(
        path,
        expected_episode_id=episode_id,
        expected_source_frames=frames,
    )


@pytest.mark.parametrize(
    "source,endpoint", [(0, 0), (1, 0), (3, 0), (4, 4), (5, 4), (7, 4), (8, 8), (15, 12), (16, 16)]
)
def test_exact_causal_mapping(source, endpoint):
    endpoints = tuple(range(0, 40, 4))
    selected = endpoints[causal_endpoint_index(endpoints, source)]
    assert selected == endpoint
    assert selected <= source


@pytest.mark.parametrize("endpoints", [(), (4, 8), (0, 4, 4), (0, 8, 4), (0, 4.0, 8)])
def test_invalid_endpoint_order_or_stride_rejected(endpoints):
    with pytest.raises(ValueError):
        causal_endpoint_index(endpoints, 5)


@pytest.mark.parametrize("source", [-1, 1.5, True])
def test_invalid_source_step_rejected(source):
    with pytest.raises(ValueError):
        causal_endpoint_index((0, 4, 8), source)


def test_visual96_exact_mean_rms_formula():
    latent = torch.arange(48 * 16 * 16).reshape(48, 16, 16).remainder(101).half()
    wrist = -latent.flip(-1) * 0.5
    actual = latent_to_visual96(latent, wrist)
    value = torch.cat((latent.float(), wrist.float()), dim=-1)
    expected = torch.cat((value.mean((1, 2)), (value.square().mean((1, 2)) + 1e-6).sqrt()))
    assert actual.shape == (96,)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_visual96_accumulates_in_fp32_and_has_fixed_epsilon():
    latent = torch.full((48, 16, 16), 60000.0, dtype=torch.float16)
    summary = latent_to_visual96(latent, latent)
    assert torch.isfinite(summary).all()
    torch.testing.assert_close(summary, torch.full((96,), 60000.0))
    zero = latent_to_visual96(torch.zeros_like(latent), torch.zeros_like(latent))
    assert torch.equal(zero[:48], torch.zeros(48))
    torch.testing.assert_close(zero[48:], torch.full((48,), 0.001))


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "inf"])
@pytest.mark.parametrize("view", [0, 1])
def test_invalid_latent_rejected(kind, view):
    latent = torch.zeros(48, 16, 16, dtype=torch.float16)
    if kind == "shape":
        latent = latent[:, :, :8]
    elif kind == "dtype":
        latent = latent.float()
    else:
        latent[0, 0, 0] = float(kind)
    with pytest.raises(ValueError):
        views = [torch.zeros(48, 16, 16, dtype=torch.float16)] * 2
        views[view] = latent
        latent_to_visual96(*views)


def test_future_endpoint_change_cannot_affect_earlier_visual():
    endpoints = (0, 4, 8, 12, 16)
    latents = torch.arange(5).half()[:, None, None, None].expand(5, 48, 16, 16).clone()
    index = causal_endpoint_index(endpoints, 7)
    before = latent_to_visual96(latents[index], latents[index])
    latents[2:] = 60000
    after = latent_to_visual96(latents[causal_endpoint_index(endpoints, 7)], latents[index])
    assert torch.equal(before, after)


def test_h5_reader_endpoints_and_copy_isolation(tmp_path):
    reader = read_cache(write_cache(tmp_path / "episode.h5"))
    assert reader.endpoint_indices == (*range(0, 67, 4), 66)
    endpoint, summary = reader.visual_summary(7)
    assert endpoint == 4
    torch.testing.assert_close(summary[:48], torch.full((48,), 4.0))
    summary.fill_(999)
    assert reader.visual_summary(7)[1][0] == 4
    with pytest.raises(ValueError):
        reader.visual_summary(67)


@pytest.mark.parametrize("camera", [LEFT_CAMERA, WRIST_CAMERA])
def test_future_sensitive_h5_cache(tmp_path, camera):
    path = write_cache(tmp_path / "episode.h5")
    before = read_cache(path)
    with h5py.File(path, "r+") as cache:
        cache[f"latents/{camera}"][2:] = 60000
    after = read_cache(path)
    for source in range(8):
        assert torch.equal(before.visual_summary(source)[1], after.visual_summary(source)[1])
    assert not torch.equal(before.visual_summary(8)[1], after.visual_summary(8)[1])


@pytest.mark.parametrize(
    "key", ["episode_id", "frame_count", "temporal_compression_factor", "source_frame_to_latent_policy"]
)
def test_missing_metadata_fail_closed(tmp_path, key):
    path = write_cache(tmp_path / "episode.h5")
    with h5py.File(path, "r+") as cache:
        del cache.attrs[key]
    with pytest.raises(KeyError):
        read_cache(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("episode_id", "other/shard/episode0"),
        ("frame_count", 68),
        ("frame_count", 67.0),
        ("temporal_compression_factor", 8),
        ("temporal_compression_factor", "4"),
        ("source_frame_to_latent_policy", "nearest"),
        ("source_frame_to_latent_policy", "ceil"),
    ],
)
def test_wrong_metadata_fail_closed(tmp_path, key, value):
    path = write_cache(tmp_path / "episode.h5")
    with h5py.File(path, "r+") as cache:
        del cache.attrs[key]
        cache.attrs[key] = value
    with pytest.raises(ValueError):
        read_cache(path)


@pytest.mark.parametrize(
    "kind", ["shape", "dtype", "nan", "inf", "order", "duplicate", "outside", "index_dtype", "length"]
)
@pytest.mark.parametrize("camera", [LEFT_CAMERA, WRIST_CAMERA])
def test_malformed_h5_data_fail_closed(tmp_path, kind, camera):
    path = write_cache(tmp_path / "episode.h5")
    with h5py.File(path, "r+") as cache:
        latent_key = f"latents/{camera}"
        endpoint_key = f"indices/latent_source_frame_indices/{camera}"
        latent = cache[latent_key]
        endpoints = cache[endpoint_key]
        if kind in ("shape", "dtype"):
            data = latent[:]
            del cache[latent_key]
            cache.create_dataset(latent_key, data=data[:, :, :, :8] if kind == "shape" else data.astype(np.float32))
        elif kind in ("nan", "inf"):
            latent[-1, 0, 0, 0] = float(kind)
        elif kind == "order":
            endpoints[1:3] = [8, 4]
        elif kind == "duplicate":
            endpoints[1] = 0
        elif kind == "outside":
            endpoints[-1] = 68
        else:
            data = endpoints[:]
            del cache[endpoint_key]
            cache.create_dataset(endpoint_key, data=data.astype(np.float32) if kind == "index_dtype" else data[:-1])
    with pytest.raises(ValueError):
        read_cache(path)


def test_missing_cache_or_key_has_no_fallback(tmp_path):
    with pytest.raises(OSError):
        read_cache(tmp_path / "missing.h5")
    path = write_cache(tmp_path / "episode.h5")
    with h5py.File(path, "r+") as cache:
        del cache[f"latents/{WRIST_CAMERA}"]
    with pytest.raises(KeyError):
        read_cache(path)


@pytest.mark.parametrize(
    "frames,tail", [(313, (304, 308, 312)), (299, (292, 296, 298)), (280, (272, 276, 279)), (1, (0,))]
)
def test_exact_terminal_vectors_and_causal_lookup(tmp_path, frames, tail):
    reader = read_cache(write_cache(tmp_path / "episode.h5", frames), frames)
    assert reader.endpoint_indices[-len(tail) :] == tail
    expected = list(range(0, frames, 4))
    if expected[-1] != frames - 1:
        expected.append(frames - 1)
    assert reader.endpoint_indices == tuple(expected)
    for source in range(max(0, frames - 8), frames):
        endpoint, _ = reader.visual_summary(source)
        assert endpoint == max(value for value in expected if value <= source)


def test_only_root_frame_count_is_authority(tmp_path):
    path = write_cache(tmp_path / "episode.h5")
    with h5py.File(path, "r+") as cache:
        assert "source_video_frames" not in cache.attrs
        cache.attrs["source_video_frames"] = 999
    assert read_cache(path).source_frames == 67
    with h5py.File(path, "r+") as cache:
        del cache.attrs["frame_count"]
        cache.require_group("metadata").attrs["frame_count"] = 67
    with pytest.raises(KeyError):
        read_cache(path)


@pytest.mark.parametrize("camera", [LEFT_CAMERA, WRIST_CAMERA])
@pytest.mark.parametrize("kind", ["false", "dtype", "shape", "missing"])
def test_selected_valid_mask_is_required(tmp_path, camera, kind):
    path = write_cache(tmp_path / "episode.h5")
    with h5py.File(path, "r+") as cache:
        key = f"valid/{camera}"
        if kind == "false":
            cache[key][-1] = False
        else:
            data = cache[key][:]
            del cache[key]
            if kind != "missing":
                cache.create_dataset(key, data=data.astype(np.uint8) if kind == "dtype" else data[:-1])
    with pytest.raises((ValueError, KeyError)):
        read_cache(path)


def test_missing_terminal_or_off_grid_interior_rejected(tmp_path):
    for bad_endpoints in ((0, 4, 8), (0, 3, 8, 10)):
        path = write_cache(tmp_path / "episode.h5", frames=11)
        with h5py.File(path, "r+") as cache:
            for camera in (LEFT_CAMERA, WRIST_CAMERA):
                key = f"indices/latent_source_frame_indices/{camera}"
                del cache[key]
                cache.create_dataset(key, data=bad_endpoints)
        with pytest.raises(ValueError):
            read_cache(path, frames=11)


def test_two_view_formula_and_right_camera_has_no_authority(tmp_path):
    path = write_cache(tmp_path / "episode.h5")
    with h5py.File(path, "r+") as cache:
        cache[f"latents/{LEFT_CAMERA}"][:] = 2
        cache[f"latents/{WRIST_CAMERA}"][:] = 6
        cache.create_dataset(
            "latents/observation.images.robot0_agentview_right", data=np.zeros((18, 48, 16, 16), dtype=np.float16)
        )
    before = read_cache(path).visual_summary(7)[1]
    expected = torch.cat((torch.full((48,), 4.0), torch.full((48,), 20.0 + 1e-6).sqrt()))
    torch.testing.assert_close(before, expected, rtol=0, atol=0)
    with h5py.File(path, "r+") as cache:
        cache["latents/observation.images.robot0_agentview_right"][:] = float("nan")
    assert torch.equal(before, read_cache(path).visual_summary(7)[1])


@pytest.mark.parametrize("keyword", ["latent_key", "endpoint_key", "metadata_group", "camera"])
def test_arbitrary_camera_control_surface_rejected(tmp_path, keyword):
    with pytest.raises(TypeError):
        RoboCasaLatentReader(
            tmp_path / "unused.h5", expected_episode_id="episode", expected_source_frames=67, **{keyword: "right"}
        )
