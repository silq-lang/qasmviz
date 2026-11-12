include "stdgates.inc";
qubit[2] q;
p(pi/4) q[0];
ctrl @ p(pi/2) q[0],q[1];
