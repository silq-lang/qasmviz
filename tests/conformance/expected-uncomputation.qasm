include "stdgates.inc";
qubit[3] q;
bit m;
h q[0];
h q[1];
ccx q[0], q[1], q[2];
h q[2];
m = measure q[2];
reset q[2];
if (m) { cz q[0], q[1]; }
