from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TrainingSampleIdentity:
    source_sample_index: int
    row_index: int
    row_count: int


class ModelCompanion(torch.nn.Module):
    def __init__(self, *, pipeline_rank: int = 0, chunk_index: int = 0, replica_id: tuple[int, ...] = (0,)) -> None:
        super().__init__()
        self.pipeline_rank = pipeline_rank
        self.chunk_index = chunk_index
        self.replica_id = replica_id
        self.register_buffer("weight_version", torch.zeros((), dtype=torch.int64, device="cpu"))
        self.register_buffer("trained_rows", torch.empty((0, 4), dtype=torch.int64, device="cpu"))
        self.register_buffer("skipped_nonfinite_rows", torch.empty((0, 4), dtype=torch.int64, device="cpu"))

    def record(self, samples: Iterable[TrainingSampleIdentity]) -> None:
        counts = _decode_rows(self.trained_rows)
        counts.update(samples)
        _write_rows(target=self.trained_rows, counts=counts)

    def snapshot(self) -> dict[TrainingSampleIdentity, int]:
        return dict(_decode_rows(self.trained_rows))

    def record_skipped_nonfinite(self, samples: Iterable[TrainingSampleIdentity]) -> None:
        counts = _decode_rows(self.skipped_nonfinite_rows)
        counts.update(samples)
        _write_rows(target=self.skipped_nonfinite_rows, counts=counts)

    def snapshot_skipped_nonfinite(self) -> dict[TrainingSampleIdentity, int]:
        return dict(_decode_rows(self.skipped_nonfinite_rows))

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> dict[str, Any]:
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        return {
            f"{prefix}{name}": ShardedObject(
                key=f"{prefix}pp{self.pipeline_rank}.chunk{self.chunk_index}.{name}",
                data=buffer.clone(),
                global_shape=(1,),
                global_offset=(0,),
                replica_id=self.replica_id,
            )
            for name, buffer in self.named_buffers()
        }

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "ModelCompanion":
        return self

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for name, buffer in self.named_buffers():
            incoming = state_dict[f"{prefix}{name}"]
            assert incoming.dtype == torch.int64
            if name == "weight_version":
                assert incoming.ndim == 0 and incoming.item() >= 0
            else:
                assert incoming.ndim == 2 and incoming.shape[1] == 4
            buffer.resize_(incoming.shape)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class ModelCompanionUtils:
    @staticmethod
    def install(model: torch.nn.Module, *, chunk_index: int) -> None:
        from miles.backends.training_utils.parallel import get_parallel_state

        parallel = get_parallel_state()
        model.add_module(
            "model_companion",
            ModelCompanion(
                pipeline_rank=parallel.pp.rank,
                chunk_index=chunk_index,
                replica_id=(parallel.tp.rank, parallel.cp.rank, parallel.intra_dp.rank),
            ),
        )

    @staticmethod
    def weight_version(model: Sequence[torch.nn.Module]) -> int:
        companions = _model_companions(model)
        assert companions, "Model is missing its companion"
        return ModelCompanionUtils.version_from_buffers(ModelCompanionUtils.named_buffers(model))

    @staticmethod
    def version_from_buffers(buffers: Iterable[tuple[str, torch.Tensor]]) -> int:
        versions = {int(buffer.item()) for name, buffer in buffers if name.endswith(".weight_version")}
        assert len(versions) == 1, f"Model companion versions diverged: {versions}"
        return versions.pop()

    @staticmethod
    def bump_weight_version(model: Sequence[torch.nn.Module]) -> None:
        companions = _model_companions(model)
        if companions:
            ModelCompanionUtils.weight_version(model)
        for companion in companions:
            companion.weight_version.add_(1)

    @staticmethod
    def record(model: Sequence[torch.nn.Module], samples: Iterable[TrainingSampleIdentity]) -> None:
        identities = list(samples)
        for companion in _model_companions(model):
            companion.record(identities)

    @staticmethod
    def snapshot(model: Sequence[torch.nn.Module]) -> dict[TrainingSampleIdentity, int]:
        return _snapshot(model=model, skipped_nonfinite=False)

    @staticmethod
    def record_nonfinite_skip(model: Sequence[torch.nn.Module], samples: Iterable[TrainingSampleIdentity]) -> None:
        identities = list(samples)
        for companion in _model_companions(model):
            companion.record_skipped_nonfinite(identities)

    @staticmethod
    def snapshot_nonfinite_skip(model: Sequence[torch.nn.Module]) -> dict[TrainingSampleIdentity, int]:
        return _snapshot(model=model, skipped_nonfinite=True)

    @staticmethod
    def clear(model: Sequence[torch.nn.Module]) -> None:
        for companion in _model_companions(model):
            companion.weight_version.zero_()
            companion.trained_rows.resize_((0, 4))
            companion.skipped_nonfinite_rows.resize_((0, 4))

    @staticmethod
    def named_buffers(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
        for chunk_index, companion in enumerate(_model_companions(model)):
            for name, buffer in companion.named_buffers():
                yield f"model_companion.{chunk_index}.{name}", buffer

    @staticmethod
    def is_buffer_name(name: str) -> bool:
        return name.startswith("model_companion.")

    @staticmethod
    @contextmanager
    def hide(model: Sequence[torch.nn.Module]) -> Iterator[None]:
        children = [
            (parent, name, child)
            for chunk in model
            for parent in list(chunk.modules())
            for name, child in list(parent.named_children())
            if isinstance(child, ModelCompanion)
        ]
        for parent, name, _ in children:
            delattr(parent, name)
        try:
            yield
        finally:
            for parent, name, child in children:
                parent.add_module(name, child)


def _model_companions(model: Sequence[torch.nn.Module]) -> list[ModelCompanion]:
    return [module for chunk in model for module in chunk.modules() if isinstance(module, ModelCompanion)]


def _snapshot(model: Sequence[torch.nn.Module], *, skipped_nonfinite: bool) -> dict[TrainingSampleIdentity, int]:
    companions = _model_companions(model)
    assert companions, "Model is missing its CPU training witness"
    snapshots = [
        companion.snapshot_skipped_nonfinite() if skipped_nonfinite else companion.snapshot()
        for companion in companions
    ]
    assert all(snapshot == snapshots[0] for snapshot in snapshots[1:]), "CPU witness model chunks diverged"
    return snapshots[0]


def _decode_rows(rows: torch.Tensor) -> Counter[TrainingSampleIdentity]:
    return Counter(
        {
            TrainingSampleIdentity(source_sample_index=source, row_index=row, row_count=row_count): count
            for source, row, row_count, count in rows.tolist()
        }
    )


def _write_rows(*, target: torch.Tensor, counts: Counter[TrainingSampleIdentity]) -> None:
    values = [
        [identity.source_sample_index, identity.row_index, identity.row_count, count]
        for identity, count in counts.items()
    ]
    source = torch.tensor(values, dtype=torch.int64, device="cpu").reshape((-1, 4))
    target.resize_(source.shape)
    target.copy_(source)
