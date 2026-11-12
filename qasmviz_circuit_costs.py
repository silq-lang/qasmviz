"""Conservative execution counts for Qiskit circuits with control flow."""

from qiskit.circuit import (
    BoxOp, BreakLoopOp, CASE_DEFAULT, ContinueLoopOp, ControlFlowOp,
    ForLoopOp, IfElseOp, SwitchCaseOp, WhileLoopOp,
)
from qiskit.circuit.classical import expr

from qasmviz_dynamic_costs import Analyzer, Bound, add, merge, total


def _constant(value):
    # Do not infer runtime classical values from the initial register state.
    return value.value if isinstance(value, expr.Value) else None


class CircuitAnalyzer:
    def __init__(self, t_cost):
        self.t_cost = t_cost

    def block(self, circuit):
        flow = {'normal': {}}
        for instruction in circuit.data:
            prefix = flow.pop('normal', None)
            if prefix is None:
                break
            for exit_kind, cost in self.operation(instruction.operation).items():
                merge(flow, exit_kind, add(prefix, cost))
        return flow

    def operation(self, operation):
        if isinstance(operation, IfElseOp):
            known = _constant(operation.condition)
            flow = {}
            for take_true in (True, False):
                if known is not None and bool(known) != take_true:
                    continue
                index = 0 if take_true else 1
                branch = self.block(operation.blocks[index]) if index < len(operation.blocks) else {'normal': {}}
                for exit_kind, cost in branch.items():
                    merge(flow, exit_kind, cost)
            return flow
        if isinstance(operation, ForLoopOp):
            indexset = operation.params[0]
            if isinstance(indexset, range):
                span = indexset.stop - indexset.start
                count = 0 if span * indexset.step <= 0 else (abs(span) - 1) // abs(indexset.step) + 1
            else:
                count = len(indexset)
            # Counts of parameter-dependent rotations may remain unknown;
            # fixed gate counts do not require expansion of large loops.
            return Analyzer.repeat(self.block(operation.blocks[0]), count)
        if isinstance(operation, WhileLoopOp):
            known = _constant(operation.condition)
            if known is not None and not known:
                return {'normal': {}}
            return Analyzer.repeat(self.block(operation.blocks[0]), None, known is not None and bool(known))
        if isinstance(operation, SwitchCaseOp):
            known = _constant(operation.target)
            flow, default = {}, None
            for labels, block in operation.cases_specifier():
                if CASE_DEFAULT in labels:
                    default = block
                if known is not None and known in labels:
                    return self.block(block)
                if known is None:
                    for exit_kind, cost in self.block(block).items():
                        merge(flow, exit_kind, cost)
            if known is not None:
                return self.block(default) if default is not None else {'normal': {}}
            if default is None:
                merge(flow, 'normal', {})
            return flow
        if isinstance(operation, BoxOp):
            return self.block(operation.blocks[0])
        if isinstance(operation, BreakLoopOp):
            return {'break': {}}
        if isinstance(operation, ContinueLoopOp):
            return {'continue': {}}
        if isinstance(operation, ControlFlowOp):
            # A new backend construct must not silently acquire a zero cost.
            return {'normal': {key: Bound(0, None) for key in (
                'gate-count', 't-count', 'cx-count', '2q-count', 'measurements', 'resets')}}
        if operation.name in ('measure', 'reset'):
            key = 'measurements' if operation.name == 'measure' else 'resets'
            return {'normal': {key: Bound(operation.num_qubits, operation.num_qubits)}}
        if operation.name in ('barrier', 'delay', 'snapshot') or not operation.num_qubits:
            return {'normal': {}}
        cost = {'gate-count': Bound(1, 1), 'gate:' + operation.name: Bound(1, 1)}
        if operation.num_qubits == 2:
            cost['2q-count'] = Bound(1, 1)
        if operation.name == 'cx':
            cost['cx-count'] = Bound(1, 1)
        t = self.t_cost(operation)
        cost['t-count'] = Bound(0, None) if t is None else Bound(t, t)
        return {'normal': cost}


def collect_circuit_dynamic_costs(circuit, t_cost):
    counts = total(CircuitAnalyzer(t_cost).block(circuit))
    result = {
        'analysis': 'dynamic', 'bounds': 'conservative', 'width': circuit.num_qubits,
        'clbits': circuit.num_clbits, 'depth': None, 't-depth': None, '2q-depth': None,
    }
    for key in ('gate-count', 't-count', 'cx-count', '2q-count', 'measurements', 'resets'):
        result[key] = counts.get(key, Bound()).json()
    result['gate-counts'] = {key[5:]: value.json() for key, value in sorted(counts.items())
                           if key.startswith('gate:') and value.upper != 0}
    from qasmviz_expected import attach_expected_counts
    from qasmviz_expected_circuit import execute_circuit
    return attach_expected_counts(result, lambda budget: execute_circuit(circuit, t_cost, budget))
