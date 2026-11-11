import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import openqasm3
from openqasm3 import ast
from qiskit import QuantumCircuit
from qiskit.circuit import CASE_DEFAULT
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Clifford, Operator, random_clifford
from qiskit.transpiler import PassManager
from qiskit_qasm3_import import parse as import_quantum

from qasmviz_basis import (
    CLIFFORD_GATES, EXTENDED_CLIFFORD_T_BASIS, CliffordBasisTranslation, clifford_decomposition,
    parse_extended_basis, restore_logical_wires, validate_clifford_basis,
)
from qasmviz_clifford_qasm import clifford_definitions, dumps_clifford_t
from qasmviz_clifford_t import CLIFFORD_T_BASIS, clifford_t_pass_manager
from qasmviz_dynamic_costs import collect_dynamic_costs
from qasmviz_qasm3 import compile_clifford_t_qasm3, compile_clifford_t_source_with_ast


ROOT = Path(__file__).resolve().parents[1]
HEADER = 'OPENQASM 3.0; include "stdgates.inc";\n'
CUSTOM = ('[h,s,cx]', '[h,sdg,cz]', '[s,sx,cy]', '[sdg,sxdg,ecr]', '[s,sx,iswap]', '[h,s,dcx]')


def gates_in(statements):
    for statement in statements:
        if isinstance(statement, ast.QuantumGate):
            yield statement.name.name
        elif isinstance(statement, ast.BranchingStatement):
            yield from gates_in(statement.if_block)
            yield from gates_in(statement.else_block)
        elif isinstance(statement, ast.CompoundStatement):
            yield from gates_in(statement.statements)
        elif isinstance(statement, (ast.ForInLoop, ast.WhileLoop)):
            yield from gates_in(statement.block)


def chosen_branch(program, take_true):
    """Extract the fixtures' selected quantum path, including gate definitions."""
    result = []
    for statement in program.statements:
        if isinstance(statement, (ast.Include, ast.QubitDeclaration, ast.QuantumGateDefinition,
                                  ast.QuantumGate, ast.QuantumPhase)):
            result.append(statement)
        elif isinstance(statement, ast.BranchingStatement):
            block = statement.if_block if take_true else statement.else_block
            result.extend(chosen_branch(ast.Program(block), take_true).statements)
    return ast.Program(result, version='3.0')


class CliffordBasisTests(unittest.TestCase):
    def test_all_named_cliffords_are_discrete_clifford_gates(self):
        mapping = get_standard_gate_name_mapping()
        for name in CLIFFORD_GATES:
            gate = mapping[name]
            self.assertFalse(gate.params)
            Clifford(gate)
        self.assertEqual(set(parse_extended_basis('all')), set(CLIFFORD_GATES) | {'t', 'tdg'})

    def test_complete_alternative_generators_are_accepted(self):
        for specification in CUSTOM:
            with self.subTest(specification=specification):
                basis = parse_extended_basis(specification)
                self.assertEqual(basis[-2:], ('t', 'tdg'))
                for name in ('h', 's', 'cx'):
                    circuit = clifford_decomposition(name, basis)
                    self.assertLessEqual(set(circuit.count_ops()), set(basis) - {'t', 'tdg'})
                    np.testing.assert_allclose(Operator(circuit).data,
                                               get_standard_gate_name_mapping()[name].to_matrix(), atol=1e-12)

    def test_incomplete_groups_are_rejected_even_with_t_available(self):
        for specification in ('[]', '[cx]', '[h,cx]', '[h,x,cx]', '[x,y,z,cx]',
                              '[h,s]', '[h,s,swap]', '[h,s,t,tdg,swap]'):
            with self.subTest(specification=specification), self.assertRaisesRegex(ValueError, 'incomplete'):
                parse_extended_basis(specification)

    def test_non_cliffords_unknown_and_parameterized_gates_are_rejected(self):
        for name in ('ch', 'cs', 'ccx', 'rx', 'rz', 'u', 'made_up', 'H'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'unsupported generators'):
                parse_extended_basis(f'[h,s,cx,{name}]')

    def test_list_syntax_and_duplicates(self):
        expected = ('h', 's', 'cx', 't', 'tdg')
        for value in ('[h,s,cx]', '["h", "s", "cx"]', "['h','s','cx']", '[ h, s, cx, ]', '[h,s,cx,h,t,tdg]'):
            self.assertEqual(parse_extended_basis(value), expected)
        for value in ('h,s,cx', '', '[h,,s,cx]', '[h,s,cx', "[h,s,'cx]", '[h,s,cx()]'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_extended_basis(value)

    def test_decompositions_preserve_global_phase_for_every_named_gate(self):
        for specification in CUSTOM:
            basis = parse_extended_basis(specification)
            for name in CLIFFORD_GATES:
                with self.subTest(basis=specification, gate=name):
                    circuit = clifford_decomposition(name, basis)
                    self.assertLessEqual(set(circuit.count_ops()), set(basis) - {'t', 'tdg'})
                    np.testing.assert_allclose(Operator(circuit).data,
                                               get_standard_gate_name_mapping()[name].to_matrix(), atol=1e-12, rtol=0)

    def test_returned_decompositions_do_not_mutate_the_cache(self):
        basis = parse_extended_basis('[s,sx,cz]')
        before = clifford_decomposition('h', basis)
        modified = clifford_decomposition('h', basis)
        modified.x(0)
        self.assertEqual(clifford_decomposition('h', basis), before)


class ExtendedCompilerTests(unittest.TestCase):
    def compile(self, circuit, basis, **kwargs):
        output = clifford_t_pass_manager(basis_gates=basis, qubits_initially_zero=False, **kwargs).run(circuit)
        output = restore_logical_wires(output, basis)
        self.assertLessEqual(set(output.count_ops()), set(basis))
        return output

    def test_default_extended_basis_keeps_named_cliffords(self):
        mapping = get_standard_gate_name_mapping()
        for name in CLIFFORD_GATES:
            gate = mapping[name]
            circuit = QuantumCircuit(gate.num_qubits)
            circuit.append(gate, circuit.qubits)
            output = self.compile(circuit, EXTENDED_CLIFFORD_T_BASIS, optimization_level=0)
            self.assertEqual(output.count_ops().get(name), 1)
            np.testing.assert_allclose(Operator(output).data, gate.to_matrix(), atol=1e-12, rtol=0)

    def test_random_cliffords_stay_t_free_and_match_on_all_input_states(self):
        for specification, level in itertools.product(CUSTOM, (0, 3)):
            with self.subTest(specification=specification, level=level):
                circuit = random_clifford(3, seed=4781).to_circuit()
                circuit.global_phase += 0.37
                basis = parse_extended_basis(specification)
                output = self.compile(circuit, basis, optimization_level=level)
                self.assertFalse({'t', 'tdg'} & set(output.count_ops()))
                np.testing.assert_allclose(Operator(output).data, Operator(circuit).data, atol=1e-11, rtol=0)

    def test_synthesis_error_budget_with_alternative_single_qubit_generators(self):
        circuit = QuantumCircuit(1)
        circuit.rz(0.123456789, 0)
        for specification, epsilon in itertools.product(('[h,sdg,cz]', '[s,sx,iswap]'), (1e-3, 1e-6)):
            basis = parse_extended_basis(specification)
            output = self.compile(circuit, basis, epsilon=epsilon, optimization_level=1)
            self.assertLessEqual(np.linalg.norm(Operator(output).data - Operator(circuit).data, 2), epsilon + 1e-12)

    def test_static_export_definitions_phase_and_final_permutation(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.sxdg(1)
        circuit.ecr(0, 1)
        circuit.swap(0, 1)
        circuit.global_phase += 0.7
        for specification in ('all', '[s,sx,ecr]', '[h,sdg,cz]'):
            basis = parse_extended_basis(specification)
            output = self.compile(circuit, basis, optimization_level=3)
            self.assertIsNone(output.layout)
            reparsed = import_quantum(dumps_clifford_t(output))
            np.testing.assert_allclose(Operator(reparsed).data, Operator(circuit).data, atol=1e-11, rtol=0)

    def test_each_nonstandard_definition_is_phase_correct(self):
        for definition in clifford_definitions(CLIFFORD_GATES):
            name = definition.name.name
            gate = get_standard_gate_name_mapping()[name]
            qubits = [ast.IndexedIdentifier(ast.Identifier('q'), [[ast.IntegerLiteral(i)]]) for i in range(gate.num_qubits)]
            program = ast.Program([
                ast.Include('stdgates.inc'), definition,
                ast.QubitDeclaration(ast.Identifier('q'), ast.IntegerLiteral(gate.num_qubits)),
                ast.QuantumGate([], ast.Identifier(name), [], qubits),
            ], version='3.0')
            circuit = import_quantum(openqasm3.dumps(program))
            np.testing.assert_allclose(Operator(circuit).data, gate.to_matrix(), atol=1e-12, rtol=0)

    def test_old_default_basis_is_unchanged(self):
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.cz(0, 1)
        output = clifford_t_pass_manager(qubits_initially_zero=False).run(circuit)
        self.assertLessEqual(set(output.count_ops()), set(CLIFFORD_T_BASIS))

    def test_control_flow_translation_and_exported_block_phases(self):
        circuit = QuantumCircuit(2, 1)
        with circuit.if_test((circuit.clbits[0], True)) as otherwise:
            circuit.h(0)
            circuit.sxdg(1)
            circuit.ecr(0, 1)
        with otherwise:
            circuit.sxdg(0)
        basis = parse_extended_basis('[s,sx,cz]')
        output = PassManager([CliffordBasisTranslation(basis)]).run(circuit)
        output = restore_logical_wires(output, basis)
        reimported = import_quantum(dumps_clifford_t(output))
        for original, compiled, exported in zip(circuit.data[0].operation.blocks,
                                                 output.data[0].operation.blocks,
                                                 reimported.data[0].operation.blocks):
            self.assertLessEqual(set(compiled.count_ops()), set(basis))
            np.testing.assert_allclose(Operator(compiled).data, Operator(original).data, atol=1e-12, rtol=0)
            np.testing.assert_allclose(Operator(exported).data, Operator(original).data, atol=1e-12, rtol=0)

    def test_exported_switch_phases_follow_case_labels(self):
        circuit = QuantumCircuit(1, 2)
        default = QuantumCircuit(1, 2, global_phase=0.2)
        zero = QuantumCircuit(1, 2, global_phase=0.4)
        circuit.switch(circuit.cregs[0], [(0, zero), (CASE_DEFAULT, default)], circuit.qubits, circuit.clbits)
        program = openqasm3.parse(dumps_clifford_t(circuit))
        switch = next(s for s in program.statements if isinstance(s, ast.SwitchStatement))
        self.assertAlmostEqual(switch.default.statements[0].argument.value, 0.2)
        self.assertAlmostEqual(switch.cases[0][1].statements[0].argument.value, 0.4)


class ExtendedSourceTests(unittest.TestCase):
    def test_dynamic_regions_and_restored_swaps_use_only_allowed_generators(self):
        source = HEADER + '''qubit[2] q; bit b;
if(b) { h q[0]; s q[1]; swap q[0],q[1]; } else { x q[1]; }
if(b) { cx q[0],q[1]; } else { y q[0]; }
'''
        original = openqasm3.parse(source)
        for specification, level in itertools.product(('all', *CUSTOM), range(4)):
            with self.subTest(basis=specification, level=level):
                basis = parse_extended_basis(specification)
                output, program = compile_clifford_t_source_with_ast(source, basis_gates=basis, optimization_level=level)
                reparsed = openqasm3.parse(output)
                self.assertLessEqual(set(gates_in(reparsed.statements)), set(basis) - {'t', 'tdg'})
                self.assertEqual(collect_dynamic_costs(program)['t-count'], {'lower': 0, 'upper': 0})
                for take_true in (False, True):
                    before = import_quantum(openqasm3.dumps(chosen_branch(original, take_true)))
                    after = import_quantum(openqasm3.dumps(chosen_branch(reparsed, take_true)))
                    np.testing.assert_allclose(Operator(after).data, Operator(before).data, atol=1e-11, rtol=0)

    def test_added_definitions_do_not_change_classical_literal_spelling(self):
        literal = '0.1234567890123456789012345678901234'
        source = HEADER + f'qubit[2] q; float[128] v={literal}; // retain this comment\nif(true) {{ cx q[0],q[1]; }}'
        output, _ = compile_clifford_t_source_with_ast(source, basis_gates=parse_extended_basis('[s,sx,ecr]'))
        self.assertIn('gate ecr ', output)
        self.assertIn(literal, output)
        self.assertIn('// retain this comment', output)

    def test_ast_api_adds_definitions_without_mutating_the_input(self):
        source = HEADER + 'qubit[2] q; if(true) { cx q[0],q[1]; }'
        original = openqasm3.parse(source)
        before = copy.deepcopy(original)
        program = compile_clifford_t_qasm3(original, basis_gates=parse_extended_basis('[s,sx,ecr]'))
        self.assertEqual(original, before)
        self.assertTrue(any(isinstance(s, ast.QuantumGateDefinition) and s.name.name == 'ecr' for s in program.statements))
        import_quantum(openqasm3.dumps(chosen_branch(program, True)))

    def test_inlined_subroutine_and_loop_use_selected_basis(self):
        source = HEADER + '''def f(qubit p)->bit { x p; return measure p; }
qubit[2] q; for int i in [0:1] { if(f(q[i])) { t q[i]; } }
'''
        basis = parse_extended_basis('[s,sx,cz]')
        output, program = compile_clifford_t_source_with_ast(source, basis_gates=basis)
        self.assertLessEqual(set(gates_in(openqasm3.parse(output).statements)), set(basis))
        self.assertEqual(collect_dynamic_costs(program)['measurements'], {'lower': 2, 'upper': 2})


class ExtendedCliTests(unittest.TestCase):
    def cli(self, *arguments, source=None, cwd=ROOT):
        source = source if source is not None else HEADER + 'qubit[2] q; h q[0]; s q[1]; cx q[0],q[1];'
        return subprocess.run([sys.executable, str(ROOT / 'qasmviz.py'), *arguments], input=source,
                              text=True, capture_output=True, cwd=cwd, timeout=30)

    def test_bare_flag_before_filename_and_explicit_end_of_options(self):
        source = HEADER + 'qubit q; x q;'
        with tempfile.TemporaryDirectory() as directory:
            for name in ('circuit.qasm', '--clifford-t-ext'):
                path = Path(directory) / name
                path.write_text(source)
            result = self.cli('--clifford-t-ext', 'circuit.qasm', '--dump', cwd=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('x ', result.stdout)
            result = self.cli('--clifford-t-ext', '--dump', '--', '--clifford-t-ext', cwd=directory)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_custom_basis_dump_and_clifford_cost(self):
        for source in (HEADER + 'qubit[2] q; h q[0]; cx q[0],q[1];',
                       HEADER + 'qubit[2] q; bit b; if(b) { h q[0]; cx q[0],q[1]; }'):
            result = self.cli('--clifford-t-ext=[s,sx,cz]', '--dump', source=source)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertLessEqual(set(gates_in(openqasm3.parse(result.stdout).statements)), {'s', 'sx', 'cz', 't', 'tdg'})
            result = self.cli('--clifford-t-ext=[s,sx,cz]', '--json', source=source)
            self.assertEqual(result.returncode, 0, result.stderr)
            count = json.loads(result.stdout).get('t-count', 0)
            self.assertIn(count, (0, {'lower': 0, 'upper': 0}))

    def test_incomplete_or_invalid_lists_fail_before_reading_input(self):
        for specification, message in (('[h,s,swap]', 'incomplete'), ('[h,cx]', 'incomplete'),
                                        ('[h,s,cx,ccx]', 'unsupported'), ('h,s,cx', 'bracketed'), ('[]', 'incomplete')):
            result = self.cli('--clifford-t-ext=' + specification, '--dump', source='')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_mutually_exclusive_targets(self):
        for other in ('--clifford-t', '--cx-u', '--fez'):
            result = self.cli('--clifford-t-ext', other, '--dump')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('at most one target', result.stderr)

    def test_existing_compilation_options_accept_extended_target(self):
        result = self.cli('--clifford-t-ext=[s,sx,cz]', '--eps', '1e-6', '--width', '2',
                          '--max-loop-variants', '5', '--max-inline-statements', '100', '--dump',
                          source=HEADER + 'qubit[2] q; for int i in [0:1] { rz(pi*i/7) q[i]; }')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLessEqual(set(gates_in(openqasm3.parse(result.stdout).statements)), {'s', 'sx', 'cz', 't', 'tdg'})

    def test_qasm2_input_and_output_and_simulation(self):
        source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; x q[0]; cx q[0],q[1];'
        result = self.cli('--clifford-t-ext=[s,sx,cz]', '--dump2', source=source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('OPENQASM 2.0;', result.stdout)
        result = self.cli('--clifford-t-ext', '--run', '--json', source=source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['statevector'][0]['basis'], '11')


if __name__ == '__main__':
    unittest.main()
