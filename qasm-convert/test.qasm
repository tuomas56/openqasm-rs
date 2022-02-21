OPENQASM 2.0;

include "qelib1.inc";

qreg q[4];

h q;
cx q[0], q[1];
cx q[1], q[2];
barrier q[1], q[2];
tdg q[2];
U(0, 0, pi/8) q[3];
reset q[2];
cx q[2], q[1];
cx q[1], q[0];
h q;