rbc = 3.88
abc = 13.81
ipon = 1
potshape = 8

rc =abc*rbc**potshape
ac =abc*rbc**6

emin=-27.0*ac**4/(256.0*rc**3)
rmin2=4.0*rc/(3.0*ac)


def nonbon(dis):
    dsq = dis * dis
    rr2 = 1/dsq

    alen = ac
    rlen = rc
    rr23 = rr2 * rr2 * rr2
    rep = rlen * rr2
    vlj = (rep-alen)*rr23
    
    ivor = ipon
    if dsq < rmin2:
        energy = vlj + (ivor-1) * emin    
    else:
        energy = ivor * vlj
    return energy

dis = 2.9000558951718105
refe_energy = 62.570219508272075

print(dis, nonbon(dis), refe_energy)

all_dis = [float(l) for l in open("make-energy-init.out")]
all_refe_energy = [float(l.split()[1]) for l in open("score.out")]
assert len(all_dis) == len(all_refe_energy)

for dis, refe_energy in zip(all_dis, all_refe_energy):
    print(dis, nonbon(dis), refe_energy)
