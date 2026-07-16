"""Unit tests for services.zenodo.build_zenodo_metadata (Zenodo deposition metadata)."""
from donadataset.services.zenodo import build_zenodo_metadata, get_zenodo_communities


def _base_config(**zenodo_overrides) -> dict:
    return {
        "zenodo": {
            "creators": [{"name": "Test Author"}],
            **zenodo_overrides,
        },
    }


def test_get_zenodo_communities_parses_configured_list():
    config = _base_config(communities=["camera-traps", "biodiversity"])
    assert get_zenodo_communities(config) == ["camera-traps", "biodiversity"]


def test_get_zenodo_communities_defaults_to_empty_list_when_unset():
    config = _base_config()
    assert get_zenodo_communities(config) == []


def test_build_zenodo_metadata_omits_communities_key_when_none_configured():
    metadata = build_zenodo_metadata(_base_config())
    assert "communities" not in metadata


def test_build_zenodo_metadata_includes_communities_as_identifier_dicts():
    config = _base_config(communities=["camera-traps", "biodiversity"])
    metadata = build_zenodo_metadata(config)
    assert metadata["communities"] == [
        {"identifier": "camera-traps"},
        {"identifier": "biodiversity"},
    ]
