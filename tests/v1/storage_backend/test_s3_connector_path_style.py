# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for S3 path-style addressing support.

Tests the path/Host-building logic and config validation in isolation
(no network calls, no real S3Client instantiation).
"""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector.s3_connector import _format_s3_path


class TestFormatS3Path:
    """Unit tests for the pure function ``_format_s3_path``."""

    # ------------------------------------------------------------------
    # Virtual-hosted (default) mode — regression guards
    # ------------------------------------------------------------------

    def test_default_virtual_hosted_simple_key(self):
        path = _format_s3_path("hello")
        assert path == "/hello"

    def test_default_virtual_hosted_key_with_slashes(self):
        path = _format_s3_path("a/b/c")
        assert path == "/a_b_c"

    def test_default_virtual_hosted_key_with_special_chars(self):
        path = _format_s3_path("foo bar")
        assert path == "/foo%20bar"

    def test_default_virtual_hosted_key_with_at_sign(self):
        path = _format_s3_path("model@001@chunk")
        assert path == "/model%40001%40chunk"

    def test_explicit_virtual_hosted_false(self):
        path = _format_s3_path("hello", use_path_style=False, bucket=None)
        assert path == "/hello"

    # ------------------------------------------------------------------
    # Path-style mode
    # ------------------------------------------------------------------

    def test_path_style_simple_key(self):
        path = _format_s3_path("hello", use_path_style=True, bucket="mybucket")
        assert path == "/mybucket/hello"

    def test_path_style_key_with_slashes(self):
        path = _format_s3_path("a/b/c", use_path_style=True, bucket="mybucket")
        assert path == "/mybucket/a_b_c"

    def test_path_style_key_with_special_chars(self):
        path = _format_s3_path("foo bar", use_path_style=True, bucket="b")
        assert path == "/b/foo%20bar"

    def test_path_style_key_with_at_sign(self):
        path = _format_s3_path("model@001", use_path_style=True, bucket="b")
        assert path == "/b/model%40001"

    # ------------------------------------------------------------------
    # Bucket name is never flattened (only the key is)
    # ------------------------------------------------------------------

    def test_bucket_never_flattened(self):
        path = _format_s3_path("key", use_path_style=True, bucket="my.bucket/name")
        assert path == "/my.bucket/name/key"

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_key(self):
        path = _format_s3_path("", use_path_style=True, bucket="b")
        assert path == "/b/"

    def test_empty_key_virtual_hosted(self):
        path = _format_s3_path("")
        assert path == "/"

    def test_bucket_with_special_chars_path_style(self):
        path = _format_s3_path("key", use_path_style=True, bucket="my-bucket-123")
        assert path == "/my-bucket-123/key"

    def test_long_key_path_style(self):
        key = "a" * 100
        path = _format_s3_path(key, use_path_style=True, bucket="b")
        assert path == "/b/" + "a" * 100


class TestS3ConnectorPathStyleInitValidation:
    """Test that ``S3Connector.__init__`` enforces the config contract."""

    @pytest.fixture
    def mock_deps(self):
        """Return minimal mocks that let S3Connector.__init__ reach the
        validation checks without touching awscrt for real."""
        loop = MagicMock()
        config = MagicMock()
        config.chunk_size = 256
        meta = MagicMock()
        meta.get_shapes.return_value = []
        meta.get_dtypes.return_value = []
        return loop, config, meta

    def _construct(self, loop, config, meta, **kwargs):
        """Build an S3Connector with mocked base + awscrt internals."""
        # First Party
        from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
        from lmcache.v1.storage_backend.connector.s3_connector import S3Connector

        # Patch the base class __init__ so it doesn't touch metadata methods
        _orig_base_init = RemoteConnector.__init__

        def _fake_base_init(self, cfg, md):
            self.full_chunk_size_bytes = 256
            self.metadata = md
            self.config = cfg
            self.save_chunk_meta = False

        with (
            patch.object(RemoteConnector, "__init__", _fake_base_init),
            patch("lmcache.v1.storage_backend.connector.s3_connector.io"),
            patch("lmcache.v1.storage_backend.connector.s3_connector.auth"),
            patch("lmcache.v1.storage_backend.connector.s3_connector.s3"),
            patch("lmcache.v1.storage_backend.connector.s3_connector.ClientTlsContext"),
            patch("lmcache.v1.storage_backend.connector.s3_connector.AsyncPQExecutor"),
        ):
            return S3Connector(
                s3_endpoint=kwargs.pop("s3_endpoint", "s3://localhost:9000"),
                loop=kwargs.pop("loop", loop),
                local_cpu_backend=kwargs.pop(
                    "local_cpu_backend", MagicMock(config=config, metadata=meta)
                ),
                s3_num_io_threads=kwargs.pop("s3_num_io_threads", 1),
                s3_prefer_http2=kwargs.pop("s3_prefer_http2", False),
                s3_region=kwargs.pop("s3_region", "us-east-1"),
                s3_enable_s3express=kwargs.pop("s3_enable_s3express", False),
                disable_tls=kwargs.pop("disable_tls", True),
                aws_access_key_id=kwargs.pop("aws_access_key_id", None),
                aws_secret_access_key=kwargs.pop("aws_secret_access_key", None),
                s3_use_path_style=kwargs.pop("s3_use_path_style", False),
                s3_bucket=kwargs.pop("s3_bucket", None),
            )

    def test_path_style_on_without_bucket_raises(self, mock_deps):
        loop, config, meta = mock_deps
        with pytest.raises(ValueError) as exc:
            self._construct(loop, config, meta, s3_use_path_style=True, s3_bucket=None)
        assert "s3_bucket" in str(exc.value)

    def test_path_style_off_without_bucket_ok(self, mock_deps):
        loop, config, meta = mock_deps
        conn = self._construct(
            loop, config, meta, s3_use_path_style=False, s3_bucket=None
        )
        assert conn.s3_use_path_style is False
        assert conn.s3_bucket is None

    def test_path_style_on_with_bucket_ok(self, mock_deps):
        loop, config, meta = mock_deps
        conn = self._construct(
            loop, config, meta, s3_use_path_style=True, s3_bucket="my-bucket"
        )
        assert conn.s3_use_path_style is True
        assert conn.s3_bucket == "my-bucket"

    def test_path_style_off_with_bucket_ok(self, mock_deps):
        loop, config, meta = mock_deps
        conn = self._construct(
            loop, config, meta, s3_use_path_style=False, s3_bucket="extra-bucket"
        )
        assert conn.s3_use_path_style is False
        assert conn.s3_bucket == "extra-bucket"


class TestS3L2AdapterConfigPathStyle:
    """Test that S3L2AdapterConfig.from_dict validates path-style fields."""

    def test_path_style_on_requires_bucket(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
            S3L2AdapterConfig,
        )

        with pytest.raises(ValueError, match="s3_bucket is required"):
            S3L2AdapterConfig.from_dict(
                {
                    "s3_endpoint": "s3://localhost:9000",
                    "s3_region": "us-east-1",
                    "s3_use_path_style": True,
                }
            )

    def test_path_style_on_with_bucket(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
            S3L2AdapterConfig,
        )

        cfg = S3L2AdapterConfig.from_dict(
            {
                "s3_endpoint": "s3://localhost:9000",
                "s3_region": "us-east-1",
                "s3_use_path_style": True,
                "s3_bucket": "my-bucket",
            }
        )
        assert cfg.s3_use_path_style is True
        assert cfg.s3_bucket == "my-bucket"

    def test_path_style_off_default(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
            S3L2AdapterConfig,
        )

        cfg = S3L2AdapterConfig.from_dict(
            {
                "s3_endpoint": "s3://bucket.us-east-1",
                "s3_region": "us-east-1",
            }
        )
        assert cfg.s3_use_path_style is False
        assert cfg.s3_bucket is None

    def test_path_style_must_be_bool(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
            S3L2AdapterConfig,
        )

        with pytest.raises(ValueError, match="s3_use_path_style must be a boolean"):
            S3L2AdapterConfig.from_dict(
                {
                    "s3_endpoint": "s3://b",
                    "s3_region": "us-east-1",
                    "s3_use_path_style": "yes",
                }
            )

    def test_bucket_must_be_str(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.s3_l2_adapter import (
            S3L2AdapterConfig,
        )

        with pytest.raises(ValueError, match="s3_bucket must be a string"):
            S3L2AdapterConfig.from_dict(
                {
                    "s3_endpoint": "s3://b",
                    "s3_region": "us-east-1",
                    "s3_bucket": 42,
                }
            )
