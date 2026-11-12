include "stdgates.inc";
def read_one(qubit p) -> bit { x p; return measure p; }
qubit[2] q;
bit[2] m;
for int i in [0:1] { m[i] = read_one(q[i]); }
