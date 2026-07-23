"""Owner- and generation-guarded activation for selector-addressed run groups."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
import uuid

from fisheye.shared.archive_identity import ArchiveIdentity, archive_identity
from fisheye.shared.coordinate_reference import canonical_node_path


class SelectorActivationError(ValueError):
    """Raised when one selector activation loses exact ownership."""


def _fail(message: str) -> None:
    raise SelectorActivationError(message)


@dataclass(frozen=True)
class DeferredSelectorActivation:
    """Process-local proof for one leased, selected, still-ineligible child."""

    root: Any = field(repr=False, compare=False)
    expected_archive: ArchiveIdentity
    parent_path: str
    run_path: str
    run_name: str
    owner_attr: str
    owner_uuid: str
    policy_attr: str
    generation_attr: str
    lease_attr: str
    lease: Mapping[str, Any]
    base_generation: int
    next_generation: int
    selectors: tuple[str, ...]
    expected_parent_state: Mapping[str, tuple[bool, Any]] = field(
        repr=False,
        compare=False,
    )
    mutations: tuple[tuple[str, tuple[bool, Any], Any], ...] = field(
        repr=False,
        compare=False,
    )
    baseline_proof: Any = field(repr=False, compare=False)
    proof_loader: Callable[[], Any] = field(repr=False, compare=False)
    attr_writer: Callable[[Any, str, Any], None] = field(
        repr=False,
        compare=False,
    )


def write_activation_attr(attrs: Any, name: str, value: Any) -> None:
    """Perform one injectable activation write."""

    attrs[name] = value


def _restore_attr(attrs: Any, name: str, previous: tuple[bool, Any]) -> None:
    existed, value = previous
    if existed:
        attrs[name] = copy.deepcopy(value)
    elif name in attrs:
        del attrs[name]


def _fresh_group(
    root: Any,
    path: str,
    *,
    expected_archive: ArchiveIdentity,
    label: str,
) -> Any:
    try:
        group = root[path]
    except Exception as exc:
        _fail(f"{label} disappeared during activation: {exc}.")
    if canonical_node_path(group) != path or archive_identity(group) != expected_archive:
        _fail(f"{label} changed archive or canonical path during activation.")
    return group


def _generation(
    snapshot: Mapping[str, tuple[bool, Any]],
    generation_attr: str,
) -> int:
    present, value = snapshot[generation_attr]
    if not present:
        return 0
    if type(value) is not int or value < 0:
        _fail("Publication generation must be one nonnegative integer.")
    return value


def _require_parent_state(
    parent: Any,
    names: Sequence[str],
    snapshot: Mapping[str, tuple[bool, Any]],
    overrides: Mapping[str, tuple[bool, Any]],
) -> None:
    attrs = parent.attrs
    for name in names:
        present, value = overrides.get(name, snapshot[name])
        if (name in attrs) is not present or (present and attrs.get(name) != value):
            _fail(f"Concurrent parent mutation observed for {name!r}.")


def _owned_epoch(
    parent: Any,
    *,
    lease_attr: str,
    generation_attr: str,
    lease: Mapping[str, Any],
    base_generation: int,
    next_generation: int,
) -> bool:
    generation = parent.attrs.get(generation_attr, 0)
    return (
        parent.attrs.get(lease_attr) == dict(lease)
        and type(generation) is int
        and generation in {base_generation, next_generation}
    )


def _valid_owner_uuid(value: Any) -> str | None:
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def _prior_lease_selectors(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    raw = value.get("selector_attrs", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(str(item) for item in raw)


def _validate_prior_epoch(
    root: Any,
    *,
    expected_archive: ArchiveIdentity,
    normalized_parent: str,
    owner_attr: str,
    policy: str,
    lease_schema_id: str,
    lease: Any,
    generation: int,
    parent: Any,
) -> None:
    """Require a closed, committed predecessor before advancing its epoch."""

    required_keys = {
        "schema_id",
        "schema_version",
        "policy",
        "owner_uuid",
        "publication_owner",
        "run_path",
        "run_name",
        "base_generation",
        "next_generation",
        "selector_attrs",
    }
    if not isinstance(lease, Mapping) or set(lease) != required_keys:
        _fail("Runs parent has a malformed prior publication lease.")
    owner = _valid_owner_uuid(lease.get("owner_uuid"))
    selector_attrs = _prior_lease_selectors(lease)
    run_name = str(lease.get("run_name") or "")
    run_path = str(lease.get("run_path") or "").strip("/")
    if (
        lease.get("schema_id") != str(lease_schema_id)
        or lease.get("schema_version") != 1
        or lease.get("policy") != policy
        or owner is None
        or lease.get("publication_owner") != owner
        or type(lease.get("base_generation")) is not int
        or lease.get("base_generation") != generation - 1
        or type(lease.get("next_generation")) is not int
        or lease.get("next_generation") != generation
        or not run_name
        or "/" in run_name
        or run_path != f"{normalized_parent}/{run_name}"
        or not selector_attrs
        or len(selector_attrs) != len(set(selector_attrs))
        or any(not selector or "/" in selector for selector in selector_attrs)
        or any(parent.attrs.get(selector) != run_name for selector in selector_attrs)
    ):
        _fail("Runs parent prior publication lease is not one exact committed epoch.")
    prior = _fresh_group(
        root,
        run_path,
        expected_archive=expected_archive,
        label="Prior activation candidate",
    )
    if (
        prior.attrs.get(owner_attr) != owner
        or prior.attrs.get("palette_run_completion_status") != "complete"
        or prior.attrs.get("stage_selector_eligible") is not True
    ):
        _fail("Runs parent prior publication lease names an uncommitted child.")


def _rollback_activation_attempt(
    root: Any,
    *,
    expected_archive: ArchiveIdentity,
    parent_path: str,
    run_path: str,
    owner_attr: str,
    owner_uuid: str,
    lease_attr: str,
    generation_attr: str,
    lease: Mapping[str, Any],
    base_generation: int,
    next_generation: int,
    selectors: Sequence[str],
    mutations: Sequence[tuple[str, tuple[bool, Any], Any]],
) -> list[str]:
    """Conditionally undo only values that still belong to this attempt."""

    failures: list[str] = []
    try:
        candidate = _fresh_group(
            root,
            run_path,
            expected_archive=expected_archive,
            label="Activation candidate",
        )
        if (
            candidate.attrs.get(owner_attr) == owner_uuid
            and candidate.attrs.get("stage_selector_eligible") is not False
        ):
            candidate.attrs["stage_selector_eligible"] = False
    except BaseException as exc:  # pragma: no cover - hostile store
        failures.append(f"candidate eligibility: {exc}")

    selector_set = frozenset(selectors)
    # Selectors are safe to restore by exact attempted value even after lease
    # loss: the immutable run name is unique to this publication attempt.
    for attr, previous, written in reversed(tuple(mutations)):
        if attr not in selector_set:
            continue
        try:
            parent = _fresh_group(
                root,
                parent_path,
                expected_archive=expected_archive,
                label="Activation parent",
            )
            if parent.attrs.get(attr) == written:
                _restore_attr(parent.attrs, attr, previous)
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"{attr}: {exc}")

    # Epoch metadata can be shared with or superseded by another writer. Only
    # restore it while the exact lease and one of our two generation states
    # remain owned; otherwise leave the foreign epoch untouched.
    try:
        parent = _fresh_group(
            root,
            parent_path,
            expected_archive=expected_archive,
            label="Activation parent",
        )
        owned = _owned_epoch(
            parent,
            lease_attr=lease_attr,
            generation_attr=generation_attr,
            lease=lease,
            base_generation=base_generation,
            next_generation=next_generation,
        )
    except BaseException as exc:  # pragma: no cover - hostile store
        failures.append(f"ownership check: {exc}")
        owned = False
    if owned:
        for attr, previous, written in reversed(tuple(mutations)):
            if attr in selector_set:
                continue
            try:
                parent = _fresh_group(
                    root,
                    parent_path,
                    expected_archive=expected_archive,
                    label="Activation parent",
                )
                if not _owned_epoch(
                    parent,
                    lease_attr=lease_attr,
                    generation_attr=generation_attr,
                    lease=lease,
                    base_generation=base_generation,
                    next_generation=next_generation,
                ):
                    break
                if parent.attrs.get(attr) == written:
                    _restore_attr(parent.attrs, attr, previous)
            except BaseException as exc:  # pragma: no cover - hostile store
                failures.append(f"{attr}: {exc}")
    return failures


def activate_selector_eligible_run(
    root: Any,
    parent_group: Any,
    run_group: Any,
    *,
    parent_path: str,
    run_path: str,
    run_name: str,
    owner_attr: str,
    expected_owner_uuid: str,
    policy_attr: str,
    generation_attr: str,
    lease_attr: str,
    policy: str,
    lease_schema_id: str,
    proof_loader: Callable[[], Any],
    selector_attrs: Sequence[str] = ("latest_complete", "latest"),
    parent_guard_attrs: Sequence[str] = (
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ),
    attr_writer: Callable[[Any, str, Any], None] = write_activation_attr,
    defer_eligibility: bool = False,
) -> DeferredSelectorActivation | None:
    """Lease one parent generation and expose the exact child as the final write.

    ``proof_loader`` must freshly validate the complete, selector-ineligible child
    and return a stable equality-comparable proof token (normally digest tuples).
    """

    name = str(run_name).strip()
    normalized_parent = str(parent_path).strip("/")
    normalized_run = str(run_path).strip("/")
    expected_archive = archive_identity(root)
    owner_uuid = _valid_owner_uuid(expected_owner_uuid)
    if owner_uuid is None or str(expected_owner_uuid) != owner_uuid:
        _fail("Expected candidate publication owner must be one canonical UUID.")
    selectors = tuple(dict.fromkeys(str(value) for value in selector_attrs))
    discovered_prior_selectors = _prior_lease_selectors(
        parent_group.attrs.get(lease_attr)
    )
    tracked = tuple(
        dict.fromkeys(
            (
                *selectors,
                *discovered_prior_selectors,
                *(str(value) for value in parent_guard_attrs),
                policy_attr,
                generation_attr,
                lease_attr,
            )
        )
    )
    if (
        not name
        or "/" in name
        or not selectors
        or any(not value or "/" in value for value in tracked)
        or canonical_node_path(parent_group) != normalized_parent
        or canonical_node_path(run_group) != normalized_run
        or normalized_run != f"{normalized_parent}/{name}"
        or archive_identity(parent_group) != expected_archive
        or archive_identity(run_group) != expected_archive
    ):
        _fail("Activation requires one exact owned selector-ineligible run path.")

    # The caller supplies the owner it minted when creating the candidate.  A
    # run handle can outlive deletion/recreation of its canonical child, so it
    # is path evidence only; activation authority comes from this fresh lookup.
    candidate = _fresh_group(
        root,
        normalized_run,
        expected_archive=expected_archive,
        label="Activation candidate",
    )
    if (
        candidate.attrs.get(owner_attr) != owner_uuid
        or candidate.attrs.get("stage_selector_eligible") is not False
    ):
        _fail(
            "Activation candidate does not match the expected publication "
            "owner or selector-ineligible state."
        )

    snapshot = {
        attr: (attr in parent_group.attrs, copy.deepcopy(parent_group.attrs.get(attr)))
        for attr in tracked
    }
    base_generation = _generation(snapshot, generation_attr)
    next_generation = base_generation + 1
    policy_present, prior_policy = snapshot[policy_attr]
    lease_present, prior_lease = snapshot[lease_attr]
    if policy_present and prior_policy != policy:
        _fail("Runs parent uses an unsupported publication policy.")
    if base_generation > 0:
        if not policy_present:
            _fail("Runs parent generation lacks its publication policy.")
        if not lease_present:
            _fail("Runs parent has an invalid prior publication lease.")
        if set(_prior_lease_selectors(prior_lease)) - set(tracked):
            _fail("Concurrent prior publication lease mutation was observed.")
        _validate_prior_epoch(
            root,
            expected_archive=expected_archive,
            normalized_parent=normalized_parent,
            owner_attr=owner_attr,
            policy=policy,
            lease_schema_id=lease_schema_id,
            lease=prior_lease,
            generation=base_generation,
            parent=parent_group,
        )
    elif lease_present or policy_present:
        _fail(
            "Generation-zero legacy adoption requires no publication policy or lease."
        )

    lease = {
        "schema_id": str(lease_schema_id),
        "schema_version": 1,
        "policy": policy,
        "owner_uuid": owner_uuid,
        "publication_owner": owner_uuid,
        "run_path": normalized_run,
        "run_name": name,
        "base_generation": base_generation,
        "next_generation": next_generation,
        "selector_attrs": list(selectors),
    }
    overrides: dict[str, tuple[bool, Any]] = {}
    mutations: list[tuple[str, tuple[bool, Any], Any]] = []

    def write_parent(attr: str, value: Any) -> None:
        parent = _fresh_group(
            root,
            normalized_parent,
            expected_archive=expected_archive,
            label="Activation parent",
        )
        _require_parent_state(parent, tracked, snapshot, overrides)
        previous = overrides.get(attr, snapshot[attr])
        written = copy.deepcopy(value)
        mutations.append((attr, previous, written))
        attr_writer(parent.attrs, attr, written)
        overrides[attr] = (True, written)
        parent = _fresh_group(
            root,
            normalized_parent,
            expected_archive=expected_archive,
            label="Activation parent",
        )
        _require_parent_state(parent, tracked, snapshot, overrides)

    baseline_proof = proof_loader()
    final_write_attempted = False
    try:
        write_parent(lease_attr, lease)
        if proof_loader() != baseline_proof:
            _fail("Candidate publication changed after lease acquisition.")
        for selector in selectors:
            write_parent(selector, name)
        write_parent(policy_attr, policy)
        write_parent(generation_attr, next_generation)

        parent = _fresh_group(
            root,
            normalized_parent,
            expected_archive=expected_archive,
            label="Activation parent",
        )
        candidate = _fresh_group(
            root,
            normalized_run,
            expected_archive=expected_archive,
            label="Activation candidate",
        )
        if (
            not _owned_epoch(
                parent,
                lease_attr=lease_attr,
                generation_attr=generation_attr,
                lease=lease,
                base_generation=base_generation,
                next_generation=next_generation,
            )
            or any(parent.attrs.get(selector) != name for selector in selectors)
            or candidate.attrs.get(owner_attr) != owner_uuid
            or candidate.attrs.get("stage_selector_eligible") is not False
            or proof_loader() != baseline_proof
        ):
            _fail("Owned selector epoch or exact candidate did not persist.")
        if defer_eligibility:
            return DeferredSelectorActivation(
                root=root,
                expected_archive=expected_archive,
                parent_path=normalized_parent,
                run_path=normalized_run,
                run_name=name,
                owner_attr=owner_attr,
                owner_uuid=owner_uuid,
                policy_attr=policy_attr,
                generation_attr=generation_attr,
                lease_attr=lease_attr,
                lease=copy.deepcopy(lease),
                base_generation=base_generation,
                next_generation=next_generation,
                selectors=selectors,
                expected_parent_state={
                    attr: copy.deepcopy(overrides.get(attr, snapshot[attr]))
                    for attr in tracked
                },
                mutations=tuple(copy.deepcopy(mutations)),
                baseline_proof=baseline_proof,
                proof_loader=proof_loader,
                attr_writer=attr_writer,
            )
        # Literal commit point. No fallible read or metadata write follows.
        final_write_attempted = True
        attr_writer(candidate.attrs, "stage_selector_eligible", True)
    except BaseException as exc:
        committed = False
        if final_write_attempted:
            try:
                candidate = _fresh_group(
                    root,
                    normalized_run,
                    expected_archive=expected_archive,
                    label="Activation candidate",
                )
                committed = (
                    candidate.attrs.get(owner_attr) == owner_uuid
                    and candidate.attrs.get("stage_selector_eligible") is True
                )
            except BaseException:
                committed = False
        if committed:
            return
        failures = _rollback_activation_attempt(
            root,
            expected_archive=expected_archive,
            parent_path=normalized_parent,
            run_path=normalized_run,
            owner_attr=owner_attr,
            owner_uuid=owner_uuid,
            lease_attr=lease_attr,
            generation_attr=generation_attr,
            lease=lease,
            base_generation=base_generation,
            next_generation=next_generation,
            selectors=selectors,
            mutations=mutations,
        )
        if failures:
            raise SelectorActivationError(
                f"Owned activation rollback was incomplete: {failures!r}."
            ) from exc
        raise

    return None


def commit_deferred_selector_activation(
    activation: DeferredSelectorActivation,
    *,
    root: Any,
    parent_group: Any,
    run_group: Any,
    proof_loader: Callable[[], Any],
) -> None:
    """Rebind a deferred lease to fresh handles, then perform its commit write.

    Atomic publishers can write child metadata after the receipt is created.
    The final eligibility mutation therefore must use the freshly reopened
    root/parent/run supplied at the literal commit point, never the receipt's
    potentially stale group handles or proof closure.
    """

    if archive_identity(root) != activation.expected_archive:
        _fail("Deferred activation fresh root changed archives/stores.")
    if (
        canonical_node_path(parent_group) != activation.parent_path
        or canonical_node_path(run_group) != activation.run_path
        or archive_identity(parent_group) != activation.expected_archive
        or archive_identity(run_group) != activation.expected_archive
    ):
        _fail("Deferred activation received invalid fresh parent/run handles.")

    final_write_attempted = False
    try:
        parent = _fresh_group(
            root,
            activation.parent_path,
            expected_archive=activation.expected_archive,
            label="Activation parent",
        )
        _require_parent_state(
            parent,
            tuple(activation.expected_parent_state),
            activation.expected_parent_state,
            {},
        )
        candidate = _fresh_group(
            root,
            activation.run_path,
            expected_archive=activation.expected_archive,
            label="Activation candidate",
        )
        if (
            not _owned_epoch(
                parent,
                lease_attr=activation.lease_attr,
                generation_attr=activation.generation_attr,
                lease=activation.lease,
                base_generation=activation.base_generation,
                next_generation=activation.next_generation,
            )
            or parent.attrs.get(activation.generation_attr)
            != activation.next_generation
            or any(
                parent.attrs.get(selector) != activation.run_name
                for selector in activation.selectors
            )
            or candidate.attrs.get(activation.owner_attr) != activation.owner_uuid
            or run_group.attrs.get(activation.owner_attr) != activation.owner_uuid
            or candidate.attrs.get("palette_run_completion_status") != "complete"
            or candidate.attrs.get("stage_selector_eligible") is not False
            or run_group.attrs.get("stage_selector_eligible") is not False
            or proof_loader() != activation.baseline_proof
        ):
            _fail("Deferred activation lost its exact owned selector epoch.")
        # Literal commit point. No fallible read or metadata write follows.
        final_write_attempted = True
        activation.attr_writer(candidate.attrs, "stage_selector_eligible", True)
    except BaseException as exc:
        committed = False
        if final_write_attempted:
            try:
                candidate = _fresh_group(
                    root,
                    activation.run_path,
                    expected_archive=activation.expected_archive,
                    label="Activation candidate",
                )
                committed = (
                    candidate.attrs.get(activation.owner_attr)
                    == activation.owner_uuid
                    and candidate.attrs.get("stage_selector_eligible") is True
                )
            except BaseException:
                committed = False
        if committed:
            return
        failures = _rollback_activation_attempt(
            root,
            expected_archive=activation.expected_archive,
            parent_path=activation.parent_path,
            run_path=activation.run_path,
            owner_attr=activation.owner_attr,
            owner_uuid=activation.owner_uuid,
            lease_attr=activation.lease_attr,
            generation_attr=activation.generation_attr,
            lease=activation.lease,
            base_generation=activation.base_generation,
            next_generation=activation.next_generation,
            selectors=activation.selectors,
            mutations=activation.mutations,
        )
        if failures:
            raise SelectorActivationError(
                f"Deferred activation rollback was incomplete: {failures!r}."
            ) from exc
        raise


def rollback_deferred_selector_activation(
    activation: DeferredSelectorActivation,
) -> None:
    """Conditionally cancel one staged activation without touching alien state."""

    failures = _rollback_activation_attempt(
        activation.root,
        expected_archive=activation.expected_archive,
        parent_path=activation.parent_path,
        run_path=activation.run_path,
        owner_attr=activation.owner_attr,
        owner_uuid=activation.owner_uuid,
        lease_attr=activation.lease_attr,
        generation_attr=activation.generation_attr,
        lease=activation.lease,
        base_generation=activation.base_generation,
        next_generation=activation.next_generation,
        selectors=activation.selectors,
        mutations=activation.mutations,
    )
    if failures:
        raise SelectorActivationError(
            f"Deferred activation rollback was incomplete: {failures!r}."
        )


__all__ = [
    "DeferredSelectorActivation",
    "SelectorActivationError",
    "activate_selector_eligible_run",
    "commit_deferred_selector_activation",
    "rollback_deferred_selector_activation",
    "write_activation_attr",
]
