"""Eliminate single-use measurement results without enumerating outcome paths.

Each matched measurement and correction becomes its unconditional quantum
channel. This preserves correlations with later measurements. Only the cost
of this correction uses its marginal probability; no independence is assumed.
"""

from dataclasses import dataclass

from qasmviz_expected import Event, Path, probability_one


class UnsupportedStream(Exception):
    """Use the bounded classical executor for this control-flow structure."""


@dataclass
class Quantum:
    events: tuple
    counts: dict

    @classmethod
    def capture(cls, path):
        result = cls(tuple(reversed(tuple(reversed(path.history)))), path.counts)
        path.history, path.counts = (), {}
        return result


@dataclass
class Sample:
    wire: int
    key: object


@dataclass
class Choice:
    key: object
    branches: tuple  # false, true


def plan(items, budget):
    """Check all result lifetimes before doing any probability work."""
    versions, uses, origins = {}, {}, {}
    for index, item in enumerate(items):
        budget.spend()
        if isinstance(item, Sample):
            if item.key is not None:
                versions[item.key] = index
        elif isinstance(item, Choice):
            origin = versions.get(item.key)
            if origin is None:
                raise UnsupportedStream()
            origins[index] = origin
            uses[origin] = uses.get(origin, 0) + 1
    output, index = [], 0
    while index < len(items):
        budget.spend()
        item = items[index]
        if isinstance(item, Quantum):
            output.append(item)
            index += 1
            continue
        if not isinstance(item, Sample) or uses.get(index, 0) > 1:
            raise UnsupportedStream()
        if not uses.get(index, 0):
            output.append(Quantum((Event('discard', (item.wire,)),), {'measurements': 1}))
            index += 1
            continue
        next_index, common = index + 1, Quantum((), {})
        if next_index < len(items) and isinstance(items[next_index], Quantum):
            candidate = items[next_index]
            if candidate.events == (Event('reset', (item.wire,)),):
                common = candidate
                next_index += 1
        if origins.get(next_index) != index:
            raise UnsupportedStream()
        output.append((item.wire, common, items[next_index].branches))
        index = next_index + 1
    return output


def _count(path, counts, probability=1):
    if not probability:
        return
    for name, count in counts.items():
        path.count(name, None if count is None else probability * count)


def execute(items, budget):
    path = Path()
    for item in items:
        budget.spend()
        if isinstance(item, Quantum):
            _count(path, item.counts)
            for event in item.events:
                path.record(event, budget)
            continue
        wire, common, branches = item
        probability = path.facts.probability(wire) if path.facts is not None else None
        if probability is None:
            probability = probability_one(path.history, wire, budget)
        path.count('measurements')
        _count(path, common.counts)
        for outcome, branch in enumerate(branches):
            _count(path, branch.counts, probability if outcome else 1 - probability)
        touched = {q for event in common.events for q in event.wires}
        touched.update(q for branch in branches for event in branch.events for q in event.wires)
        path.record(Event('instrument', (wire,) + tuple(sorted(touched - {wire})),
                          (common.events, tuple(branch.events for branch in branches))), budget)
    return [path]
