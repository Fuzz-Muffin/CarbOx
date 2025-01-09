import numpy as np
import datetime
import re

def sort_nicely(l):
    # Sort the given list in the way that humans expect.
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def write_xyz_old(natoms, atypes, apos, box, outfile):
    with open(outfile,'w') as fo:
        fo.write(f'{natoms}\n')
        fo.write(f'Lattice="{box[0]} 0.0 0.0  0.0 {box[1]} 0.0  0.0 0.0 {box[2]}" Properties=species:S:1:pos:R:3\n')
        for t,p in zip(atypes,apos):
            fo.write(f'{t}  {p[0]}  {p[1]}  {p[2]}\n')
    return

def write_xyz(natoms, atypes, apos, box, outfile, extra=None):
    commentline = f'Lattice="{box[0]} 0.0 0.0  0.0 {box[1]} 0.0  0.0 0.0 {box[2]}" '
    commentline += 'Properties=species:S:1:pos:R:3'
    if extra:
        for i,arr in enumerate(extra):
            if arr.dtype == np.float64:
                if len(arr.shape) == 1:
                    dim = 1
                else:
                    dim = arr[0].shape[0]

                commentline += f':extra{i}:R:{dim}'

    commentline += '\n'

    print(commentline)
    with open(outfile,'w') as fo:
        fo.write(f'{natoms}\n')
        fo.write(commentline)
        outlist = [atypes, apos[:,0], apos[:,1], apos[:,2]] + extra
        for vals in zip(*outlist):
            for val in vals:
                fo.write(str(val) + ' ')

            fo.write('\n')
    return

def read_xyz(infile):
    # get the file extension so we can guess how to get the data out
    ext = infile.split('.')[-1]

    # xyz file format, use extended xyz definition and look for column details in the header
    # otherwise we assume the following column format: <type> <x> <y> <z>
    if ext == 'xyz':
        with open(infile,'r') as fi:
            natoms = int(fi.readline().strip())
            comment_line = fi.readline().strip()
            # check if it is in extended xyz format
            if ('Properties' in comment_line) or ('Lattice' in comment_line):
                is_extended_format = True
                comment_line = comment_line.split('"')
                header = comment_line[-1]
                box = np.array([float(i) for i in comment_line[comment_line.index('Lattice=')+1].split()])
                box = box.reshape(3,3).diagonal()
            else:
                is_extended_format = False
                id_ind = None
                type_ind = 0
                pos_ind = [1,2,3]
                box = np.array([float(i) for i in comment_line.split()])

            data = [next(fi) for n in range(natoms)]

        if is_extended_format:
            props = header.split('Properties=')[1].split(':')
            col_labels = [sss.lower() for sss in props[::3]]
            if 'id' not in col_labels:
                id_ind = None
            else:
                id_ind = col_labels.index('id')

            ii = col_labels.index('pos')
            pos_ind = [ii, ii+1, ii+2]
            type_ind = col_labels.index('species')

        # atom ids are 1-based
        if not id_ind:
            aids = np.arange(1,natoms+1,dtype=int)
        else:
            aids = np.zeroes(natoms,dtype=int)

        atypelabels = []
        atypes = np.zeros(natoms,dtype=int)
        apos = np.zeros((natoms,3))
        for i,row in enumerate(data):
            blah = row.strip().split()
            atypelabels.append(blah[type_ind])
            apos[i,:] = np.array([float(blah[jj]) for jj in pos_ind])
            if id_ind:
                aids[i] = blah[id_ind]

        unique_atypelabels = list(set(atypelabels))
        sort_nicely(unique_atypelabels)
        atypelabels = np.array(atypelabels, dtype='O')
        unique_types = [int(i) for i in range(1,len(unique_atypelabels) + 1)]
        for j,lab in zip(unique_types,unique_atypelabels):
            atypes[atypelabels==lab] = j

    # all atom properties, and the box dimensions are numpy arrays
    # however, lists of unique types and labels are LISTS
    return natoms, box, aids, atypes, atypelabels, apos, unique_types, unique_atypelabels

def makearray(inlist):
    outarray = np.array([[float(i) for i in row.split()] for row in inlist])
    return outarray

def trimcomments(inlist):
    outlist = [txt.split('#')[0].rstrip(' ') for txt in inlist]
    return outlist

def parse_frontmatter(frontmatter,natomtype,nbondtype,nangletype,ndihedtype,nimproptype):
    frontlist = frontmatter.split('\n')
    massdata = np.array([])
    paircoeffs = np.array([])
    bondcoeffs = np.array([])
    anglecoeffs = np.array([])
    dihedcoeffs = np.array([])
    impropcoeffs = np.array([])
    for i,val in enumerate(frontlist):
        if 'Masses' in val:
            lst = trimcomments(frontlist[i+2:i+2+natomtype])
            massdata = makearray(lst)
        if 'Pair Coeffs' in val:
            lst = trimcomments(frontlist[i+2:i+2+natomtype])
            paircoeffs = makearray(lst)
        if 'Bond Coeffs' in val:
            lst = trimcomments(frontlist[i+2:i+2+nbondtype])
            bondcoeffs = makearray(lst)
        if 'Angle Coeffs' in val:
            lst = trimcomments(frontlist[i+2:i+2+nangletype])
            anglecoeffs = makearray(lst)
        if 'Dihedral Coeffs' in val:
            lst = trimcomments(frontlist[i+2:i+2+ndihedtype])
            dihedcoeffs = makearray(lst)
        if 'Improper Coeffs' in val:
            lst = trimcomments(frontlist[i+2:i+2+nimproptype])
            impropcoeffs = makearray(lst)

    return massdata, paircoeffs, bondcoeffs, anglecoeffs, dihedcoeffs, impropcoeffs

def format_frontmatter(massdata, paircoeffs, bondcoeffs, anglecoeffs, dihedcoeffs, impropcoeffs):
    outstring = 'Masses\n\n'
    for row in massdata:
        for i,col in enumerate(row):
            if i==0:
                outstring += str(int(col)) + ' '
            else:
                outstring += str(col) + ' '
        outstring = outstring + '\n'
    outstring = outstring + '\n'

    outstring = outstring + 'Pair Coeffs\n\n'
    for row in paircoeffs:
        for i,col in enumerate(row):
            if i==0:
                outstring += str(int(col)) + ' '
            else:
                outstring += str(col) + ' '
        outstring = outstring + '\n'
    outstring = outstring + '\n'

    outstring = outstring + 'Bond Coeffs\n\n'
    for row in bondcoeffs:
        for i,col in enumerate(row):
            if i==0:
                outstring += str(int(col)) + ' '
            else:
                outstring += str(col) + ' '
        outstring = outstring + '\n'
    outstring = outstring + '\n'

    outstring = outstring + 'Angle Coeffs\n\n'
    for row in anglecoeffs:
        for i,col in enumerate(row):
            if i==0:
                outstring += str(int(col)) + ' '
            else:
                outstring += str(col) + ' '
        outstring = outstring + '\n'
    outstring = outstring + '\n'

    outstring = outstring + 'Dihedral Coeffs\n\n'
    for row in dihedcoeffs:
        for i,col in enumerate(row):
            if i==0 or i==2 or i==3:
                outstring += str(int(col)) + ' '
            else:
                outstring += str(col) + ' '
        outstring = outstring + '\n'
    outstring = outstring + '\n'

    outstring = outstring + 'Improper Coeffs\n\n'
    for row in impropcoeffs:
        for i,col in enumerate(row):
            if i==0:
                outstring += str(int(col)) + ' '
            else:
                outstring += str(col) + ' '
        outstring = outstring + '\n'
    outstring = outstring + '\n'

    return outstring

def write_lammps_data_full(frontmatter, atom_mass_list, natoms, natomtypes, atom_id, atom_mol, atom_type, atom_q, atom_pos, atom_image, vel_id, vel, nbonds, nbondtypes, bond_id, bond_type, bond_atoms, nangles, nangletypes, angle_id, angle_type, angle_atoms, ndiheds, ndihedtypes, dihed_id, dihed_type, dihed_atoms, nimprops, nimproptypes, improp_id, improp_type, improp_atoms, xlo, xhi, ylo, yhi, zlo, zhi, box, outfile):
    # get the date and time
    now = datetime.datetime.now()

    with open(outfile, 'w') as outframe:
        header = f'#LAMMPS data file written by FV script {now.strftime("%H:%M:%S %d %b %Y")}'
        outframe.write(header + '\n\n')

        outframe.write(str(natoms) + ' atoms\n')
        outframe.write(str(natomtypes) + ' atom types\n')

        outframe.write(str(nbonds) + ' bonds\n')
        outframe.write(str(nbondtypes) + ' bond types\n')

        outframe.write(str(nangles) + ' angles\n')
        outframe.write(str(nangletypes) + ' angle types\n')

        outframe.write(str(ndiheds) + ' dihedrals\n')
        outframe.write(str(ndihedtypes) + ' dihedral types\n')

        outframe.write(str(nimprops) + ' impropers\n')
        outframe.write(str(nimproptypes) + ' improper types\n\n')

        outframe.write(str(xlo) + ' ' + str(xhi) + ' xlo xhi\n')
        outframe.write(str(ylo) + ' ' + str(yhi) + ' ylo yhi\n')
        outframe.write(str(zlo) + ' ' + str(zhi) + ' zlo zhi\n\n')

        outframe.write(frontmatter)

        outframe.write('Atoms # full\n\n')
        for i, atom in enumerate(atom_id):
            outstring = str(atom_id[i]) + ' ' + str(atom_mol[i]) + ' ' + str(atom_type[i]) + ' ' + str(atom_q[i]) + ' ' + ' '.join(map(str,atom_pos[i,:])) + ' ' + ' '.join(map(str,atom_image[i,:]))+ ' \n'
            outframe.write(outstring)
        outframe.write('\n')

        if np.sum(vel[0])!=0:
            outframe.write('Velocities\n\n')
            for i, atom in enumerate(atom_id):
                outstring = str(atom_id[i]) + ' ' + ' '.join(map(str,vel[i,:])) + ' \n'
                outframe.write(outstring)
            outframe.write('\n')

        outframe.write('Bonds\n\n')
        for i,bond in enumerate(bond_id):
            outstring = str(bond_id[i]) + ' ' + str(bond_type[i]) + ' ' + ' '.join(map(str,bond_atoms[i,:]))+ ' \n'
            outframe.write(outstring)
        outframe.write('\n')

        outframe.write('Angles\n\n')
        for i,angle in enumerate(angle_id):
            outstring = str(angle_id[i]) + ' ' + str(angle_type[i]) + ' ' + ' '.join(map(str,angle_atoms[i,:]))+ ' \n'
            outframe.write(outstring)
        outframe.write('\n')

        if ndiheds>0:
            outframe.write('Dihedrals\n\n')
            for i,dihed in enumerate(dihed_id):
                outstring = str(dihed_id[i]) + ' ' + str(dihed_type[i]) + ' ' + ' '.join(map(str,dihed_atoms[i,:]))+ ' \n'
                outframe.write(outstring)
            outframe.write('\n')

        if nimprops>0:
            outframe.write('Impropers\n\n')
            for i,improp in enumerate(improp_id):
                outstring = str(improp_id[i]) + ' ' + str(improp_type[i]) + ' ' + ' '.join(map(str,improp_atoms[i,:]))+ ' \n'
                outframe.write(outstring)



