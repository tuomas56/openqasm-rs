OPENQASM 2.0;

include "qelib1.inc";

gate swap a, b {
    cx a, b;
    cx b, a;
    cx a, b;
}

qreg q[4];
creg out[2];

h q;
cx q[0], q[1];
cx q[1], q[2];
swap q[0], q[1];
cz q[1], q[3];
ccx q[0], q[1], q[2];
barrier q[1], q[2];
tdg q[2];
measure q[2] -> out;
if (out[0] == 0) if (out[1] == 1) ccx q[2], q[1], q[0];
U(0, pi/4, pi/8) q[3];
crz(pi) q[0], q[2];
reset q[2];
cx q[2], q[1];
cx q[1], q[0];
h q;