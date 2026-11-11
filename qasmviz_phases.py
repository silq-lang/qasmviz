"""Give imported controlled global phases a recognizable phase-gate form."""

import cmath
import math

from qiskit.circuit import ControlFlowOp, ControlledGate, Gate, ParameterExpression
from qiskit.circuit.library import GlobalPhaseGate, PhaseGate, UnitaryGate


def _phase_only(gate):
    if gate.num_qubits or gate.num_clbits:
        return None
    if isinstance(gate, GlobalPhaseGate):
        return gate.params[0]
    if isinstance(gate, UnitaryGate):
        # pow(...) can produce a 1x1 unitary. Asking Qiskit 2.5 for its
        # definition instead can fail inside its unitary decomposer.
        return cmath.phase(gate.to_matrix()[0, 0])
    if type(gate) is Gate:
        definition = gate.definition
        if definition is not None and not definition.data:
            return definition.global_phase
    return None


def _principal_angle(angle):
    if isinstance(angle, ParameterExpression) and angle.parameters:
        return angle
    angle = math.remainder(float(angle), math.tau)
    return math.pi if angle == -math.pi else angle


def normalize_controlled_phases(circuit):
    """Replace zero-target phase wrappers, also inside classical control flow.

    Qiskit's QASM3 importer represents gphase by an empty, zero-qubit gate
    named circuit-N. Controlling it hides the phase from drawing and costing.
    Recognize its structure, not its generated name; leave other gates intact.
    """
    result = circuit.copy_empty_like()
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, ControlFlowOp):
            operation = operation.replace_blocks(
                [normalize_controlled_phases(block) for block in operation.blocks]
            )
        elif isinstance(operation, ControlledGate):
            angle = _phase_only(operation.base_gate)
            if angle is not None:
                angle = _principal_angle(angle)
                qubits = instruction.qubits
                controls = operation.num_ctrl_qubits
                state = operation.ctrl_state
                if controls == 1:
                    if state == 0:
                        # diag(exp(i*a), 1) = exp(i*a) P(-a). Keep this
                        # phase in the current block, never hoist it out.
                        result.global_phase += angle
                        angle = _principal_angle(-angle)
                    result.append(PhaseGate(angle, label=operation.label), qubits)
                else:
                    # Use the last control as P's target. An open target
                    # needs X conjugation; the other control states remain.
                    open_target = not (state & (1 << (controls - 1)))
                    if open_target:
                        result.x(qubits[-1])
                    phase = PhaseGate(angle, label=operation.label).control(
                        controls - 1, ctrl_state=state & ((1 << (controls - 1)) - 1),
                        annotated=False,
                    )
                    result.append(phase, qubits)
                    if open_target:
                        result.x(qubits[-1])
                continue
        result.append(operation, instruction.qubits, instruction.clbits)
    return result
