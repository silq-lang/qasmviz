include "stdgates.inc";
bit m0;
bit m1;
int[64] v0;
int[64] v1;
qubit[1] q;
h q[0];
m0 = measure q[0];
reset q[0];
if (m0 == true) {
  h q[0];
  m1 = measure q[0];
  reset q[0];
  if (m1 == true) { v0 = 3; } else { v0 = 2; }
}
if (m0 == true) { v1 = v0; } else { v1 = 3; }
if (2 < v1) { h q[0]; }
if (!(2 < v1)) { x q[0]; }
