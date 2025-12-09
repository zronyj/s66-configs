import sys
import os
import numpy as np

from ase import Atoms
import ase.constraints as constr
from ase.io.trajectory import Trajectory
from ase.neighborlist import natural_cutoffs, NeighborList

def split_into_molecules(atoms: Atoms, mult: float = 1.1):
    """
    Split an ASE Atoms object into molecular fragments using
    covalent-radius-based connectivity.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure (can contain many molecules).
    mult : float
        Global scaling of covalent radii. Increase slightly if
        bonds are missed; decrease if different molecules get
        spuriously connected.

    Returns
    -------
    components : list of list of int
        List of the indices of the atoms in each molecule.
    """
    # Per-atom cutoffs from covalent radii
    cutoffs = natural_cutoffs(atoms, mult=mult)

    # Build neighbor list: two atoms are neighbors if their
    # covalent-radius spheres overlap.
    nl = NeighborList(cutoffs,
                      self_interaction=False,
                      bothways=True)
    nl.update(atoms)

    # Build adjacency list (graph of bonds)
    n = len(atoms)
    adj = [[] for _ in range(n)]
    for i in range(n):
        indices, offsets = nl.get_neighbors(i)
        for j in indices:
            adj[i].append(j)

    # Find connected components by DFS/BFS
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        comp = []
        visited[start] = True
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in adj[i]:
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
        components.append(sorted(comp))

    return components

def get_centers_of_mass(atoms: Atoms, components: list):
    """
    Calculate centers of mass for molecular fragments.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure (can contain many molecules).
    components : list of list of int
        List of the indices of the atoms in each molecule.

    Returns
    -------
    coms : list of np.ndarray
        List of center-of-mass coordinates for each molecule.
    """
    coms = []
    for comp in components:
        sub_atoms = atoms[comp]
        mass = sub_atoms.get_masses().sum()
        com = np.sum(sub_atoms.get_positions() * sub_atoms.get_masses()[:, np.newaxis], axis=0) / mass
        coms.append(com)
    return coms

def get_coms(atoms: Atoms, positions: np.ndarray, components: list):
    """
    Calculate centers of mass for molecular fragments
    at given positions.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure (can contain many molecules).
    positions : np.ndarray
        Positions to use for the calculation.
    components : list of list of int
        List of the indices of the atoms in each molecule.

    Returns
    -------
    coms : list of np.ndarray
        List of center-of-mass coordinates for each molecule.
    """
    coms = []
    masses = atoms.get_masses()
    for comp in components:
        mass = masses[comp].sum()
        com = np.sum(positions[comp] * masses[comp][:, np.newaxis], axis=0) / mass
        coms.append(com)
    return coms

def append_to_trajectory_file(traj_path: str, atoms: Atoms, step: int):
    """
    Append an ASE Atoms object to a trajectory file.

    Parameters
    ----------
    traj_path : str
        Path to the trajectory file in xyz format.
    atoms : ase.Atoms
        The Atoms object to append.
    """
    content = f"{len(atoms)}\nStep: {step}\n"
    for atom in atoms:
        content += f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"

    with open(traj_path, 'a') as f:
        f.write(content)

class FixCOMs():
    """
    Constraint to fix the center of mass of all molecules
    defined by a list of lists of atom indices.
    """
    def __init__(self, indices, coms):
        self.idxs = indices
        self.coms = coms
    
    def index_shuffle(self, atoms: Atoms, ind):
        pass
    
    def get_removed_dof(self, atoms):
        return 3 * len(self.idxs)

    def adjust_positions(self, atoms, newpositions):
        current_coms = get_coms(atoms, newpositions, self.idxs)

        diffs = []
        for i, idx in enumerate(self.idxs):
            diff = current_coms[i] - self.coms[i]
            diffs.append(diff)

            for a in idx:
                newpositions[a] -= diff

    def adjust_forces(self, atoms, forces):
        pass

    def todict(self):
        return {'name': 'FixCOMs',
                'kwargs': {'indices': self.idxs,
                           'coms': self.coms}}

if __name__ == '__main__':

    # Get the current directory
    here = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(here, '..', 'md')

    # Get the path of the XYZ files
    traj_files = [f for f in os.listdir(md_path) if f.endswith('.traj')]
    traj_files.sort()

    nb_trajs = len(traj_files)

    # Get the path of the already existing trajectories
    already_done = [f for f in os.listdir(md_path) if f.startswith('aligned_') and f.endswith('.xyz')]

    # Register the FixCOMs constraint in ASE
    constr.__all__.append('FixCOMs')
    constr_dict = sys.modules['ase.constraints'].__dict__
    constr_dict['FixCOMs'] = FixCOMs

    # Loop over all trajectory files
    for t, traj in enumerate(traj_files):

        # Skip if already done
        if f'aligned_{traj.replace(".traj", ".xyz")}' in already_done:
            print(f'[{t:0>3}/{nb_trajs}] Trajectory for {traj} already exists. Skipping.')
            continue

        # Read the trajectory
        traj_path = os.path.join(md_path, traj)
        print(f'[{t:0>3}/{nb_trajs}] Reading trajectory file: {traj_path}', flush=True)
        trajectory = Trajectory(traj_path, 'r')

        # Extract the inter-com distance
        inter_COM_distance = -1

        # Prepare output trajectory file
        output_traj_path = os.path.join(here, f'aligned_{traj.replace(".traj", ".xyz")}')

        components = []

        # Loop over the trajectory steps
        for step in range(len(trajectory)):

            # Split into molecules if not already done
            if len(components) == 0:
                components = split_into_molecules(trajectory[step], mult=1.1)

                mult = 1.1

                while len(components) < 2 and mult > 0.5:
                    mult -= 0.05
                    components = split_into_molecules(trajectory[step], mult=mult)

                if len(components) > 2 or mult <= 0.5:
                    print('- Could not split trajectory into two molecules.')
                    os.remove(output_traj_path)
                    break

            # Calculate centers of mass
            coms = get_centers_of_mass(trajectory[step], components)

            # Calculate the inter-center-of-mass distance
            inter_com_distance = np.linalg.norm(coms[1] - coms[0])

            if step == 0:
                inter_COM_distance = inter_com_distance

            # Check if the inter-center-of-mass distance is within the threshold
            if abs(inter_com_distance - inter_COM_distance) > 0.5:
                print(f'- Inter-COM distance {inter_com_distance:.2f} Å at step {step} '
                        f'differs from expected {inter_COM_distance:.2f} Å by more than 0.5 Å.')
                os.remove(output_traj_path)
                break

            # Shift positions to the first molecule's COM
            ref_com = coms[0]
            shifted_positions = trajectory[step].get_positions() - ref_com

            # Rotate centers of mass to align both molecules along x-axis
            vec = coms[1] - coms[0]
            
            # Compute the Euler angles for rotation
            theta = np.arctan2(vec[1], vec[0])
            phi = np.arccos(vec[2] / np.linalg.norm(vec))

            # Compute the rotation matrix
            R = np.array([[np.cos(theta), -np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi)],
                          [np.sin(theta), np.cos(theta) * np.sin(phi), -np.cos(theta) * np.cos(phi)],
                          [0, np.sin(phi), np.cos(phi)]])
            
            # Rotate the positions
            rotated_positions = np.dot(R, shifted_positions.T).T

            # Update the positions in the trajectory
            trajectory[step].set_positions(rotated_positions)

            # Save into a new XYZ trajectory file
            append_to_trajectory_file(output_traj_path, trajectory[step], step)
