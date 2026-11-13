"""Small forward facts that keep uncomputation proofs local.

Affine expressions describe computational-basis support, not amplitudes.
Function facts have physical witness wires and are invalidated when those
wires change. Bounded local Pauli proofs recognize computed function ancillas.
"""

from fractions import Fraction


_DIAGONAL = frozenset(('id', 'z', 's', 'sdg', 't', 'tdg', 'rz', 'p', 'u1', 'cp',
                       'cu1', 'cz', 'cs', 'csdg', 'crz', 'ccz', 'diagonal'))


def writes(event):
    if event.kind != 'gate':
        return set(event.wires)
    name, = event.value
    if name in _DIAGONAL:
        return set()
    if name in ('cx', 'cy', 'ccx'):
        return {event.wires[-1]}
    return set(event.wires)


class BasisFacts:
    def __init__(self):
        self.expressions = {}
        self.atoms = {}
        self.functions = {}
        self.users = {}
        self.candidates = {}
        self.fair = set()
        self.unknown = self.ones = self.next_variable = 0

    def copy(self):
        other = BasisFacts()
        other.expressions = dict(self.expressions)
        other.atoms = dict(self.atoms)
        other.functions = dict(self.functions)
        other.users = {q: set(users) for q, users in self.users.items()}
        other.candidates = dict(self.candidates)
        other.fair = set(self.fair)
        other.unknown, other.ones, other.next_variable = self.unknown, self.ones, self.next_variable
        return other

    def copy_size(self):
        return (len(self.expressions) + len(self.atoms) + len(self.functions) + len(self.candidates)
                + len(self.fair) + sum(map(len, self.users.values())))

    def expression(self, q):
        return self.expressions.get(q, frozenset())

    def function(self, q):
        return self.functions.get(q) if self.unknown & (1 << q) else frozenset()

    def probability(self, q):
        if not self.unknown & (1 << q):
            return Fraction(bool(self.ones & (1 << q)))
        return Fraction(1, 2) if q in self.fair else None

    def _remove(self, q, budget):
        for witness in self.functions.pop(q, ()):
            budget.spend()
            self.users[witness].discard(q)

    def _invalidate(self, q, budget):
        for dependent in tuple(self.users.get(q, ())):
            self._remove(dependent, budget)
        self._remove(q, budget)
        self.fair.discard(q)

    def _function(self, q, witnesses, budget):
        if witnesses is None or q in witnesses or len(witnesses) > 32:
            return
        self.functions[q] = frozenset(witnesses)
        for witness in witnesses:
            budget.spend()
            self.users.setdefault(witness, set()).add(q)

    def _fresh(self):
        self.next_variable += 1
        return frozenset((self.next_variable,))

    def _atom(self, expression):
        # A control contributes its current value as one opaque symbol. Do not
        # expand a ripple carry's entire XOR ancestry on every use. Equal
        # expressions share a symbol, so compute/uncompute cancellation stays
        # exact. Symbols may be correlated; we never assume independence.
        if len(expression) <= 1:
            return expression
        if expression not in self.atoms:
            self.atoms[expression] = self._fresh()
        return self.atoms[expression]

    def _set(self, q, expression):
        self.expressions[q] = expression
        bit = 1 << q
        if expression <= {0}:
            self.unknown &= ~bit
            self.ones = (self.ones | bit) if expression else (self.ones & ~bit)
        else:
            self.unknown |= bit
            self.ones &= ~bit

    def _change(self, q, expression, witnesses, budget):
        self._invalidate(q, budget)
        self._set(q, expression if len(expression) <= 32 else self._atom(expression))
        self._function(q, witnesses, budget)

    def apply(self, event, history, budget):
        if event.kind == 'reset':
            self.candidates.pop(event.wires[0], None)
            self._change(event.wires[0], frozenset(), frozenset(), budget)
            return
        if event.kind == 'measure':
            outcome, _ = event.value
            self.fair.clear()  # Conditioning may change other marginal probabilities.
            self.candidates.pop(event.wires[0], None)
            self._change(event.wires[0], frozenset((0,)) if outcome else frozenset(), frozenset(), budget)
            return
        if event.kind == 'discard':
            return  # Unconditional Z measurement preserves basis support.
        if event.kind == 'instrument':
            common, branches = event.value
            for item in common:
                self.apply(item, history, budget)
            changed = set()
            for branch in branches:
                for item in branch:
                    budget.spend()
                    changed.update(writes(item))
            for q in changed:
                self._change(q, self._fresh(), None, budget)
            return
        name, = event.value
        if name in _DIAGONAL:
            return
        wires = event.wires
        if name in ('x', 'y'):
            q = wires[0]
            self._change(q, self.expression(q) ^ {0}, self.function(q), budget)
        elif name in ('cx', 'cy'):
            c, q = wires
            old = self.function(q)
            budget.spend(len(self.expression(c)) + len(self.expression(q)))
            control, target = self._atom(self.expression(c)), self.expression(q)
            value = frozenset() if control == self._atom(target) else target ^ control
            self._change(q, value,
                         None if old is None else old | {c}, budget)
        elif name == 'ccx':
            a, b, q = wires
            old = self.function(q)
            ae, be, qe = self._atom(self.expression(a)), self._atom(self.expression(b)), self.expression(q)
            value = qe if not ae or not be else (qe ^ be if ae == {0} else qe ^ ae if be == {0} else self._fresh())
            self._change(q, value, None if old is None else old | {a, b}, budget)
        elif name == 'h':
            q = wires[0]
            is_function = self.function(q) is not None
            was_known = not self.unknown & (1 << q)
            candidate = self.candidates.get(q)
            before = (history, (self.unknown, self.ones))
            self._change(q, self._fresh(), None, budget)
            self.candidates.pop(q, None)
            if is_function:
                self.fair.add(q)
            if candidate is not None:
                relation = self._recognize(q, event, history, candidate, budget)
                if relation is not None:
                    witnesses, constant = relation
                    self._function(q, witnesses, budget)
                    if constant is not None:
                        self._set(q, frozenset((0,)) if constant else frozenset())
                    self.fair.discard(q)
            if was_known:
                self.candidates[q] = before
        else:
            for q in wires:
                self._change(q, self._fresh(), None, budget)

    @staticmethod
    def _recognize(q, closing, history, candidate, budget):
        from qasmviz_expected import History, NotEasy, ONE, _pull_gate, reduce_known
        boundary, initial = candidate
        if len(history) - len(boundary) > 32:
            return None
        events = [closing]
        node = history
        while node is not boundary:
            if not isinstance(node, History) or node.event.kind != 'gate':
                return None
            events.append(node.event)
            node = node.previous
        # Scheduling may interleave preparation of a later input/scratch wire
        # with this gadget. Move operations disjoint from the prefix before
        # the opening H; they commute with that entire prefix. Their recorded
        # basis constants then belong to the local proof's initial boundary.
        active, local = {q}, []
        unknown, ones = initial
        for event in reversed(events):
            budget.spend()
            if active.intersection(event.wires):
                active.update(event.wires)
                local.append(event)
            else:
                if event.known is None:
                    return None
                mask = sum(1 << wire for wire in event.wires)
                unknown = (unknown & ~mask) | (event.known[0] & mask)
                ones = (ones & ~mask) | (event.known[1] & mask)
        events, initial = list(reversed(local)), (unknown, ones)
        touched = set(q for event in events for q in event.wires)
        if len(touched) > 5:
            return None
        terms = {(0, 1 << q): ONE}
        try:
            for event in events:
                terms = _pull_gate(terms, event.value[0], event.wires, budget)
            terms = reduce_known(terms, initial, budget)
        except NotEasy as exc:
            # Optional local recognition must not swallow a global work limit.
            if 'work limit' in str(exc):
                raise
            return None
        if any(x for x, _ in terms):
            return None
        support = 0
        for _, z in terms:
            support |= z
        witnesses = [wire for wire in touched if support & (1 << wire)]
        if len(witnesses) > 3 or any(set(witnesses) & writes(event) for event in events):
            return None
        truth = []
        for assignment in range(1 << len(witnesses)):
            bits = sum(1 << wire for i, wire in enumerate(witnesses) if (assignment >> i) & 1)
            a = b = Fraction()
            for (_, z), value in terms.items():
                budget.spend()
                sign = -1 if (z & bits).bit_count() % 2 else 1
                a += sign * value[0]
                b += sign * value[1]
            if b or a not in (-1, 1):
                return None
            truth.append(int(a == -1))
        return frozenset(witnesses), truth[0] if len(set(truth)) == 1 else None
