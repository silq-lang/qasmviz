include "stdgates.inc";
qubit[2] q;
bit b;
cx q[0],q[1];
b=measure q[0];
if(b) { h q[0]; h q[1]; t q[1]; }
