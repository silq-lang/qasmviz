// QQ_adder.qasm
include "stdgates.inc";
bit m0;
bit m1;
bit m2;
qubit[8] q;

x q[2];
h q[3];
p(0.7853981633974483) q[3];
x q[3];
negctrl @ gphase(-0.7853981633974483) q[3];
h q[3];
p(1.5707963267948966) q[3];
h q[4];
negctrl @ x q[3], q[5];
cx q[4], q[5];
p(-1.5707963267948966) q[5];
p(1.5707963267948966) q[4];
negctrl @ x q[3], q[5];
cx q[4], q[5];
h q[4];
p(1.5707963267948966) q[4];
cx q[3], q[5];
cx q[4], q[5];
h q[6];
cx q[5], q[7];
cx q[6], q[7];
p(-1.5707963267948966) q[7];
p(1.5707963267948966) q[6];
cx q[5], q[7];
cx q[6], q[7];
h q[6];
p(1.5707963267948966) q[6];
cx q[5], q[7];
cx q[6], q[7];
h q[6];
m0 = measure q[6];
reset q[6];
if ((m0==true)) {
    p(3.141592653589793) q[3];
    p(3.141592653589793) q[4];
}
h q[4];
m1 = measure q[4];
reset q[4];
if ((m1==true)) {
    negctrl @ gphase(3.141592653589793) q[3];
}
cx q[3], q[4];
h q[4];
m2 = measure q[4];
reset q[4];
x q[4];
swap q[1], q[4];
swap q[2], q[6];
swap q[3], q[4];
swap q[4], q[6];
swap q[5], q[6];
