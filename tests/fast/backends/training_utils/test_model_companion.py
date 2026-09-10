import pytest
import torch

from miles.backends.training_utils.model_companion import ModelCompanion, ModelCompanionUtils, TrainingSampleIdentity


def _identity(source_sample_index: int, row_index: int, row_count: int) -> TrainingSampleIdentity:
    return TrainingSampleIdentity(
        source_sample_index=source_sample_index,
        row_index=row_index,
        row_count=row_count,
    )


class TestModelCompanion:
    def test_install_uses_the_intra_cell_dp_replica_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FT cells describe the same checkpoint shard despite different alive ranks."""
        parallel = type(
            "Parallel",
            (),
            {
                "pp": type("Group", (), {"rank": 2})(),
                "tp": type("Group", (), {"rank": 3})(),
                "cp": type("Group", (), {"rank": 4})(),
                "intra_dp": type("Group", (), {"rank": 5})(),
                "effective_dp": type("Group", (), {"rank": 6})(),
            },
        )()
        monkeypatch.setattr(
            "miles.backends.training_utils.parallel.get_parallel_state",
            lambda: parallel,
        )
        model = torch.nn.Module()

        ModelCompanionUtils.install(model, chunk_index=1)

        assert model.model_companion.replica_id == (3, 4, 5)

    def test_hide_excludes_witness_from_pretrained_weight_load(self) -> None:
        """Pretrained weights can load while the training-only module is hidden."""
        model = torch.nn.Module()
        model.add_module("model_companion", ModelCompanion())

        with ModelCompanionUtils.hide([model]):
            assert "model_companion" not in dict(model.named_children())

        assert "model_companion" in dict(model.named_children())

    def test_counts_every_occurrence(self) -> None:
        """Repeated sample identities remain visible as repeated consumption."""
        witness = ModelCompanion()

        witness.record([_identity(7, 0, 1), _identity(8, 0, 1), _identity(7, 0, 1)])

        assert witness.snapshot() == {_identity(7, 0, 1): 2, _identity(8, 0, 1): 1}

    def test_state_dict_round_trip_resizes_and_replaces_buffers(self) -> None:
        """Checkpoint restore resizes dynamic buffers and replaces both outcomes."""
        source = ModelCompanion()
        source.record([_identity(7, 0, 2), _identity(7, 1, 2), _identity(7, 0, 2)])
        source.record_skipped_nonfinite([_identity(8, 0, 1)])
        target = ModelCompanion()
        target.record([_identity(9, 0, 1)])
        target.record_skipped_nonfinite([_identity(10, 0, 1)])

        target.load_state_dict(source.state_dict())

        assert target.snapshot() == {_identity(7, 0, 2): 2, _identity(7, 1, 2): 1}
        assert target.snapshot_skipped_nonfinite() == {_identity(8, 0, 1): 1}

    def test_parent_device_move_keeps_buffers_on_cpu(self) -> None:
        """The CPU-only companion ignores parent model device transforms."""
        witness = ModelCompanion()
        witness.record([_identity(7, 0, 1)])

        model = torch.nn.Module()
        model.add_module("companion", witness)
        model.to(device="meta")

        assert witness.trained_rows.device.type == "cpu"

    def test_record_preserves_registered_buffer_identity(self) -> None:
        """Backuper references remain valid while the witness grows."""
        witness = ModelCompanion()
        buffer = witness.trained_rows

        witness.record([_identity(7, 0, 1)])
        witness.record([_identity(8, 0, 1)])

        assert witness.trained_rows is buffer
        assert buffer.shape == (2, 4)

    def test_checkpoint_requires_both_outcomes(self) -> None:
        """A truncated state cannot silently discard skipped outcomes."""
        witness = ModelCompanion()

        with pytest.raises(KeyError, match="skipped_nonfinite_rows"):
            witness.load_state_dict({"trained_rows": torch.empty((0, 4), dtype=torch.int64)})

    def test_model_chunks_receive_identical_updates(self) -> None:
        """Virtual pipeline chunks retain identical witness mirrors."""
        chunks = [torch.nn.Module(), torch.nn.Module()]
        for chunk in chunks:
            chunk.add_module("model_companion", ModelCompanion())

        ModelCompanionUtils.record(chunks, [_identity(7, 0, 2), _identity(7, 1, 2)])
        ModelCompanionUtils.record_nonfinite_skip(chunks, [_identity(8, 0, 1)])

        assert ModelCompanionUtils.snapshot(chunks) == {_identity(7, 0, 2): 1, _identity(7, 1, 2): 1}
        assert ModelCompanionUtils.snapshot_nonfinite_skip(chunks) == {_identity(8, 0, 1): 1}

    def test_snapshot_rejects_divergent_model_chunks(self) -> None:
        """A snapshot cannot silently choose between divergent chunk mirrors."""
        chunks = [torch.nn.Module(), torch.nn.Module()]
        for chunk in chunks:
            chunk.add_module("model_companion", ModelCompanion())
        chunks[0].model_companion.record([_identity(7, 0, 1)])
        chunks[1].model_companion.record([_identity(8, 0, 1)])

        with pytest.raises(AssertionError, match="chunks diverged"):
            ModelCompanionUtils.snapshot(chunks)
