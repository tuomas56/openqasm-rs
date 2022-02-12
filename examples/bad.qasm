OPENQASM 2.0;

include "qelib1.inc";

gate majority a,b,b {
    cx b,b;
    cx b,a;
    cccx a,b,b;
}

gate unmaj a,b,c {
    ccx(theta) a,b,c;
    cx c,a, d;
    cx a,b;
}

qreg cin[1];
qreg a[4];
qreg b[2];
qreg cout[0];
creg ans[5];

// set input states
x a[0]; // a = 0001
x b; // b = 1111

// add a to b, storing result in b
majority2 cin[0],b[0],a[0];
majority a[0],b,a;
majority a[1],b[2],a[2];
majority a[2],b[3],a[3];
cx a[3],cout[0];
unmaj a[2],b[3],a[3];
unmaj a[1],b[2],a[2];
unmaj a[0],b[1],a[1];
unmaj cin[0],b[0],a[0];
measure b[0] -> cout[0];
measure b[1] -> ans[1];
measure b[2] -> ans[2];
measure b[3] -> ans[3];
measure cout[0] -> ans[4];