"""Bounded exact expected counts; no state vectors or floating-point proofs.

Measurement observables are pulled backwards as sparse sums of Hermitian
Paulis with coefficients in Q(sqrt(2)). All work limits are shared by one
analysis, including classical execution and every probability query.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache


class NotEasy(ValueError):
    """This analysis cannot establish the expectation within its limits."""


class SymbolicPhase(NotEasy):
    """A parameterized gate might still admit a zero-expectation proof."""


@dataclass(frozen=True)
class Limits:
    steps: int = 10000
    paths: int = 64
    history: int = 512
    terms: int = 32
    qubits: int = 4096
    iterations: int = 64
    depth: int = 32
    integer_bits: int = 4096
    coefficient_bits: int = 256


@dataclass
class Budget:
    limits: Limits = field(default_factory=Limits)
    steps: int = 0
    paths: int = 1

    def spend(self, count=1):
        if self.steps + count > self.limits.steps:
            raise NotEasy('analysis work limit reached')
        self.steps += count

    def fork(self):
        if self.paths >= self.limits.paths:
            raise NotEasy('classical path limit reached')
        self.paths += 1
        self.spend()

    def depth(self, depth):
        if depth > self.limits.depth:
            raise NotEasy('analysis nesting limit reached')


@dataclass(frozen=True)
class Event:
    kind: str
    wires: tuple
    value: object = None


@dataclass
class Path:
    probability: Fraction = Fraction(1)
    values: dict = field(default_factory=dict)
    scopes: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    history: tuple = ()
    exit: str = 'normal'

    def copy(self):
        return Path(self.probability, dict(self.values), [dict(s) for s in self.scopes],
                    dict(self.counts), self.history, self.exit)

    def count(self, name, value=1):
        old = self.counts.get(name, 0)
        self.counts[name] = None if value is None or old is None else old + value

    def record(self, event, budget):
        budget.spend()
        if len(self.history) >= budget.limits.history:
            raise NotEasy('quantum history limit reached')
        self.history += (event,)


ONE = (Fraction(1), Fraction(0))
ZERO = (Fraction(0), Fraction(0))


def _add_term(terms, word, coefficient, budget):
    old = terms.get(word, ZERO)
    value = old[0] + coefficient[0], old[1] + coefficient[1]
    if value == ZERO:
        terms.pop(word, None)
    else:
        if word not in terms and len(terms) >= budget.limits.terms:
            raise NotEasy('measurement expression limit reached')
        if any(max(abs(v.numerator).bit_length(), v.denominator.bit_length())
               > budget.limits.coefficient_bits for v in value):
            raise NotEasy('exact arithmetic size limit reached')
        terms[word] = value


def _scale(value, factor):
    return value[0] * factor, value[1] * factor


@lru_cache(maxsize=32)
def _clifford_action(name, arity):
    # The existing named-gate tables contain signed, exact tableau actions.
    # Invert the forward permutation for U^dagger P U.
    from qasmviz_basis import _action, _GATES
    if name not in _GATES or _GATES[name].num_qubits != arity:
        return None
    forward = _action(name, tuple(range(arity)), arity)
    inverse = [0] * len(forward)
    for before, after in enumerate(forward):
        inverse[after] = before
    return inverse


_CCX = (
    ('h', (2,)), ('cx', (1, 2)), ('tdg', (2,)), ('cx', (0, 2)),
    ('t', (2,)), ('cx', (1, 2)), ('tdg', (2,)), ('cx', (0, 2)),
    ('t', (1,)), ('t', (2,)), ('h', (2,)), ('cx', (0, 1)),
    ('t', (0,)), ('tdg', (1,)), ('cx', (0, 1)),
)
_DIAGONAL = frozenset(('diagonal', 'rz', 'p', 'u1', 'cp', 'cu1', 'crz', 'cs', 'csdg', 'ccz'))
_AXES = {'rx': 'x', 'ry': 'y', 'rxx': 'xx', 'ryy': 'yy', 'rzz': 'zz', 'rzx': 'zx'}


def _pull_gate(terms, name, wires, budget):
    mask = sum(1 << q for q in wires)
    output = {}
    arity = len(wires)
    for (x, z), coefficient in terms.items():
        budget.spend()
        if not (x | z) & mask:
            _add_term(output, (x, z), coefficient, budget)
            continue
        if name in ('t', 'tdg'):
            bit = 1 << wires[0]
            if not x & bit:
                _add_term(output, (x, z), coefficient, budget)
                continue
            half_root = coefficient[1], coefficient[0] / 2
            _add_term(output, (x, z), half_root, budget)
            sign = (1 if z & bit else -1) * (1 if name == 't' else -1)
            _add_term(output, (x, z ^ bit), _scale(half_root, sign), budget)
            continue
        # Arbitrary angles are allowed only when commutation proves they have
        # no effect on this observable. Never round an angle to a Clifford.
        if name in _DIAGONAL and not x & mask:
            _add_term(output, (x, z), coefficient, budget)
            continue
        if name in _AXES:
            axes = _AXES[name]
            ax = sum(1 << q for q, axis in zip(wires, axes) if axis in 'xy')
            az = sum(1 << q for q, axis in zip(wires, axes) if axis in 'yz')
            if ((x & az).bit_count() + (z & ax).bit_count()) % 2 == 0:
                _add_term(output, (x, z), coefficient, budget)
                continue
        if name in _DIAGONAL or name in _AXES:
            raise SymbolicPhase(f'unsupported gate in measurement dependencies: {name}')
        if name == 'ccx':
            controls, target = mask ^ (1 << wires[-1]), 1 << wires[-1]
            if not x & controls and not z & target:
                _add_term(output, (x, z), coefficient, budget)
                continue
            expanded = {(x, z): coefficient}
            for gate, positions in reversed(_CCX):
                expanded = _pull_gate(expanded, gate, tuple(wires[i] for i in positions), budget)
            for word, value in expanded.items():
                _add_term(output, word, value, budget)
            continue
        action = _clifford_action(name, arity) if arity <= 2 else None
        if action is None:
            raise NotEasy(f'unsupported gate in measurement dependencies: {name}')
        code = sum(((x >> q) & 1) << i | ((z >> q) & 1) << (arity + i)
                   for i, q in enumerate(wires))
        code = action[code]
        nx = (x & ~mask) | sum(((code >> i) & 1) << q for i, q in enumerate(wires))
        nz = (z & ~mask) | sum(((code >> (arity + i)) & 1) << q for i, q in enumerate(wires))
        sign = -1 if code >> (2 * arity) else 1
        _add_term(output, (nx, nz), _scale(coefficient, sign), budget)
    return output


def _rational_probability(history, wire, budget):
    terms = {(0, 1 << wire): ONE}
    constant = ZERO
    for event in reversed(history):
        identity = terms.pop((0, 0), ZERO)
        constant = constant[0] + identity[0], constant[1] + identity[1]
        if not terms:
            break
        if event.kind == 'gate':
            name, = event.value
            terms = _pull_gate(terms, name, event.wires, budget)
            continue
        output = {}
        bit = 1 << event.wires[0]
        for (x, z), coefficient in terms.items():
            budget.spend()
            if x & bit:
                continue  # Projection/reset removes X and Y components.
            if event.kind == 'reset':
                _add_term(output, (x, z & ~bit), coefficient, budget)
            else:
                outcome, probability = event.value
                value = _scale(coefficient, 1 / (2 * probability))
                _add_term(output, (x, z), value, budget)
                _add_term(output, (x, z ^ bit), _scale(value, -1 if outcome else 1), budget)
        terms = output
    # Every declared qubit starts in |0>. X/Y terms have zero expectation.
    expectation = constant
    for (x, _), value in terms.items():
        budget.spend()
        if not x:
            expectation = expectation[0] + value[0], expectation[1] + value[1]
    if expectation[1]:
        raise NotEasy('measurement probability is not rational')
    probability = (1 - expectation[0]) / 2
    if not 0 <= probability <= 1:
        raise RuntimeError('inconsistent exact measurement probability')
    return probability


def _zero_expectation(history, wire, budget):
    """Prove zero for every possible coefficient of unknown diagonal phases.

    Overapproximate Pauli support only: a proof succeeds when every surviving
    word has zero expectation in |0...0>. No cancellation or approximate angle
    is assumed. This covers function ancillas even with arbitrary phase gates.
    """
    words = {(0, 1 << wire)}
    touched, prefix_masks = 0, []
    for event in history:
        budget.spend(len(event.wires))
        touched |= sum(1 << q for q in event.wires)
        prefix_masks.append(touched)
    for index in range(len(history) - 1, -1, -1):
        event = history[index]
        output = set()
        for x, z in words:
            budget.spend()
            # An untouched input wire is still an independent |0>. Its X/Y
            # factor proves zero regardless of operations on the other wires.
            if x & ~prefix_masks[index]:
                continue
            if event.kind == 'gate':
                name, = event.value
                mask = sum(1 << q for q in event.wires)
                if name in _DIAGONAL and x & mask:
                    if len(event.wires) >= budget.limits.terms.bit_length():
                        raise NotEasy('measurement expression limit reached')
                    images = ((x, z ^ sum(1 << q for i, q in enumerate(event.wires) if (subset >> i) & 1))
                              for subset in range(1 << len(event.wires)))
                elif name in _AXES:
                    axes = _AXES[name]
                    ax = sum(1 << q for q, axis in zip(event.wires, axes) if axis in 'xy')
                    az = sum(1 << q for q, axis in zip(event.wires, axes) if axis in 'yz')
                    images = [(x, z)]
                    if ((x & az).bit_count() + (z & ax).bit_count()) % 2:
                        images.append((x ^ ax, z ^ az))
                else:
                    images = _pull_gate({(x, z): ONE}, name, event.wires, budget)
            else:
                bit = 1 << event.wires[0]
                images = [] if x & bit else ([(x, z & ~bit)] if event.kind == 'reset'
                                             else [(x, z), (x, z ^ bit)])
            for word in images:
                budget.spend()
                if word == (0, 0):
                    return False
                output.add(word)
                if len(output) > budget.limits.terms:
                    raise NotEasy('measurement expression limit reached')
        words = output
        if not words:
            return True
    return all(x for x, _ in words)


def probability_one(history, wire, budget):
    """Return an exact rational Born probability, or stop this analysis."""
    try:
        return _rational_probability(history, wire, budget)
    except SymbolicPhase:
        if _zero_expectation(history, wire, budget):
            return Fraction(1, 2)
        raise


def measure(path, wire, budget):
    probability = probability_one(path.history, wire, budget)
    outcomes = [(bit, probability if bit else 1 - probability) for bit in (0, 1)]
    outcomes = [(bit, p) for bit, p in outcomes if p]
    if len(outcomes) == 2:
        budget.fork()
    result = []
    for bit, p in outcomes:
        if len(outcomes) == 1:
            branch = path
        else:
            budget.spend(len(path.values) + len(path.counts) + sum(map(len, path.scopes)))
            branch = path.copy()
        branch.probability *= p
        branch.count('measurements')
        if p != 1:  # An already certain measurement does not change the state.
            branch.record(Event('measure', (wire,), (bit, p)), budget)
        result.append((branch, bit))
    return result


def gate(path, name, wires, budget, t_cost, *, count=True):
    budget.spend(len(wires))
    if len(set(wires)) != len(wires):
        raise NotEasy('overlapping gate operands')
    if count:
        path.count('gate-count')
        path.count('gate:' + name)
        path.count('t-count', t_cost)
        if len(wires) == 2:
            path.count('2q-count')
        if name == 'cx':
            path.count('cx-count')
    path.record(Event('gate', tuple(wires), (name,)), budget)


def attach_expected_counts(result, execute, *, limits=None):
    """Add expectations without changing conservative bounds or depths.

If execution stops early, discard all partial execution totals. Only exact
point bounds from the existing analysis remain usable as expectations.
"""
    metrics = {key: value for key, value in result.items()
               if isinstance(value, dict) and set(value) == {'lower', 'upper'}}
    metrics.update({'gate:' + name: bound for name, bound in result['gate-counts'].items()})
    expected = {key: Fraction(b['lower']) if b['lower'] == b['upper'] else None
                for key, b in metrics.items()}
    reason = None
    unknown = {key for key, value in expected.items() if value is None}
    if unknown == {'t-count'} and metrics['t-count']['upper'] is None:
        # All named gate counts are fixed, so probabilities cannot resolve an
        # unknown synthesis cost for a gate that is certainly executed.
        reason = 'some gate costs are unknown'
    elif unknown:
        budget = Budget(limits or Limits())
        try:
            if result['width'] > budget.limits.qubits:
                raise NotEasy('qubit limit reached')
            if result.get('clbits', 0) > budget.limits.integer_bits:
                raise NotEasy('classical size limit reached')
            paths = execute(budget)
            if sum(path.probability for path in paths) != 1:
                raise RuntimeError('incomplete expected-cost execution')
            for key in expected:
                values = [path.counts.get(key, 0) for path in paths]
                if all(value is not None for value in values):
                    expected[key] = sum((path.probability * value for path, value in zip(paths, values)), Fraction())
            if any(value is None for value in expected.values()):
                reason = 'some gate costs are unknown'
        except NotEasy as exc:
            reason = str(exc)
    def encode(value):
        return None if value is None else {'numerator': value.numerator, 'denominator': value.denominator}
    report = {key: encode(value) for key, value in expected.items() if not key.startswith('gate:')}
    report['gate-counts'] = {key[5:]: encode(value) for key, value in expected.items() if key.startswith('gate:')}
    report['status'] = ('exact' if all(v is not None for v in expected.values()) else
                        'partial' if any(v is not None for v in expected.values()) else 'unavailable')
    report['initial-state'] = 'zero'
    if reason is not None:
        report['reason'] = reason
    result['expected-counts'] = report
    return result
