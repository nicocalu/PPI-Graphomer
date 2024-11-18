from Bio import PDB
from Bio.PDB import PDBIO
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

class PDBStandardizer:
    def __init__(self, pdb_filename):
        self.pdb_filename = pdb_filename
        self.parser = PDB.PDBParser(QUIET=True)
        self.structure = self.parser.get_structure('X', self.pdb_filename)

    def renumber_residues(self, chain):
        """Renumber residues in a chain to be continuous."""
        idx = 1
        for res in chain.get_residues():
            if is_aa(res, standard=True):
                res.id = (' ', idx, ' ')
                idx += 1
        return chain

    def standardize_met(self, res):
        """Standardize MET-related residues."""
        if res.resname in ['MSE']:
            res.resname = 'MET'
        elif res.resname.startswith('A'):
            res.resname = 'MET'
        return res

    def fill_gaps_with_glycine(self, chain):
        """Fill gaps with Glycine residues."""
        residues = [res for res in chain.get_residues() if is_aa(res, standard=True)]
        filled_chain = chain.copy()
        idx = 1
        prev_res = None

        for res in residues:
            if not prev_res:
                prev_res = res
                continue

            prev_id = prev_res.id[1]
            current_id = res.id[1]

            for i in range(prev_id + 1, current_id):
                # Create a new Glycine residue
                new_residue = PDB.Residue.Residue((' ', i, ' '), 'GLY', res.segid)
                filled_chain.add(new_residue)

            prev_res = res

        return filled_chain

    def standardize_structure(self):
        for model in self.structure:
            for chain in model:
                # Renumber residues
                chain = self.renumber_residues(chain)

                # Standardize MET residues
                for res in chain:
                    res = self.standardize_met(res)

                # Fill gaps with Glycine
                chain = self.fill_gaps_with_glycine(chain)

        return self.structure

    def save_structure(self, output_filename):
        io = PDBIO()
        io.set_structure(self.structure)
        io.save(output_filename)

# 使用示例
pdb_filename = '/public/mxp/xiejun/py_project/PP/3zeu.ent.pdb'
output_filename = '/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/all/3zeu.ent.pdb'

std = PDBStandardizer(pdb_filename)
std.standardize_structure()
std.save_structure(output_filename)