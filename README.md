# PPI-Graphomer

## abstract

Protein-protein interactions (PPIs) refer to the phenomenon of protein binding through non-covalent bonds to execute biological functions. These interactions are critical for understanding biological mechanisms and drug research. Among these, the protein binding interface is a critical region involved in protein-protein interactions, particularly the hot spot residues on it that play a key role in protein interactions. Current deep learning methods trained on large-scale data can characterize proteins to a certain extent, but they often struggle to adequately capture information about protein binding interfaces. To address this limitation, we propose the PPI-Graphomer module, which integrates pretrained features from large-scale language models and inverse folding models. This approach enhances the characterization of protein binding interfaces by defining edge relationships and interface masks based on molecular interaction information. Our model outperforms existing methods across multiple benchmark datasets and demonstrates strong generalization capabilities.

## Usage

### install environment
pip install requirement.txt

### run python script
python inference.py --pdb "your_pdb_path"
