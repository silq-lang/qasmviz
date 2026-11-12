"""Strict OpenQASM 2 export without relying on Qiskit's extended qelib1."""

import math
import re

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit import ControlFlowOp, Gate
from qiskit.circuit.library import (
    CPhaseGate, CU1Gate, GlobalPhaseGate, PhaseGate, U1Gate, U3Gate, UGate,
    get_standard_gate_name_mapping,
)


# The qelib1.inc shipped with the OpenQASM 2 specification. Qiskit's exporter
# also assumes names such as cp, sx and rxx exist; strict readers do not.
_QELIB = frozenset('u3 u2 u1 cx id x y z h s sdg t tdg rx ry rz cz cy ch ccx crz cu1 cu3'.split())
_STANDARD = get_standard_gate_name_mapping()
_KEYWORDS = {'OPENQASM', 'qreg', 'creg', 'include', 'gate', 'opaque', 'U', 'CX',
             'measure', 'reset', 'if', 'barrier'}


class _Converter:
    def __init__(self, circuit):
        self.used = _QELIB | _KEYWORDS | {reg.name for reg in circuit.qregs + circuit.cregs}
        self.used = set(self.used)
        self.aliases = {}
        self.gates = []

    def unique(self, name):
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if not name or not 'a' <= name[0] <= 'z':
            name = 'gate_' + name
        stem, counter = name, 0
        while name in self.used:
            counter += 1
            name = f'{stem}_{counter}'
        self.used.add(name)
        return name

    def gate(self, operation):
        if operation.base_class is PhaseGate:
            return U1Gate(*operation.params)
        if operation.base_class is UGate:
            return U3Gate(*operation.params)
        if operation.base_class is CPhaseGate and operation.ctrl_state == 1:
            return CU1Gate(*operation.params)
        standard = _STANDARD.get(operation.name)
        if operation.name in _QELIB and standard is not None and operation.base_class is standard.base_class:
            return operation
        for previous, prepared in self.gates:
            if previous == operation or prepared == operation:
                return prepared
        name = self.unique(operation.name)
        # Temporary names force Qiskit to emit definitions even for names it
        # wrongly assumes to be in qelib1. Rename identifier tokens once, after
        # export; no angle substitution or textual gate lowering is involved.
        alias = self.unique('qasmviz_export_gate')
        self.aliases[alias] = name
        definition = operation.definition
        prepared = Gate(alias, operation.num_qubits,
                        operation.params if definition is None else [])
        if definition is not None:
            prepared.definition = self.block(definition)
        self.gates.append((operation, prepared))
        return prepared

    def block(self, circuit):
        result = circuit.copy_empty_like()
        result._layout = None
        result.global_phase = 0
        phase = float(circuit.global_phase)
        for instruction in circuit.data:
            operation = instruction.operation
            if isinstance(operation, GlobalPhaseGate):
                phase += float(operation.params[0])
                continue
            if isinstance(operation, ControlFlowOp):
                blocks = []
                for block in operation.blocks:
                    prepared = self.block(block)
                    # OQ2 permits one conditional operation, not a block. A
                    # unitary body can be one defined gate, without repeatedly
                    # evaluating a condition that a measurement might modify.
                    if len(prepared.data) > 1 and prepared.num_qubits and all(
                        isinstance(i.operation, Gate) and not i.clbits for i in prepared.data
                    ):
                        body = QuantumCircuit(prepared.num_qubits)
                        for i in prepared.data:
                            body.append(i.operation, [prepared.find_bit(q).index for q in i.qubits])
                        single = prepared.copy_empty_like()
                        single.append(self.gate(body.to_gate(label='conditional_body')), single.qubits)
                        prepared = single
                    blocks.append(prepared)
                operation = operation.replace_blocks(blocks)
            elif isinstance(operation, Gate):
                operation = self.gate(operation)
            result.append(operation, instruction.qubits, instruction.clbits)
        phase = math.remainder(phase, math.tau)
        if phase:
            if not result.num_qubits:
                raise qasm2.QASM2ExportError('OpenQASM 2 cannot encode a nonzero phase without a qubit')
            # U1(a) X U1(a) X = exp(i*a) I. Encode phases in definitions too:
            # dropping one there is observable when that gate is controlled.
            result.append(U1Gate(phase), [0])
            result.x(0)
            result.append(U1Gate(phase), [0])
            result.x(0)
        return result


def dumps_qasm2(circuit):
    """Export a bound circuit, preserving widths, definitions and global phase."""
    if circuit.parameters:
        raise qasm2.QASM2ExportError('OpenQASM 2 requires bound gate parameters')
    converter = _Converter(circuit)
    source = qasm2.dumps(converter.block(circuit))
    if converter.aliases:
        pattern = r'\b(?:' + '|'.join(map(re.escape, converter.aliases)) + r')\b'
        source = re.sub(pattern, lambda match: converter.aliases[match.group()], source)
    return source


def flat_registers(circuit):
    """Give text-based backend adapters a stable numeric wire/bit order."""
    result = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    result.compose(circuit, inplace=True)
    return result
