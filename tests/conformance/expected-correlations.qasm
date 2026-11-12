include "stdgates.inc";
qubit[2] q;
bit a;
bit b;
h q[0];
cx q[0], q[1];
h q[0];
h q[1];
a = measure q[0];
b = measure q[1];
if (a && b) { t q[0]; }
