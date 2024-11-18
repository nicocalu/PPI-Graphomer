import os
import subprocess
from Bio import PDB
import shutil

import os
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1
from subprocess import run
import tqdm
import subprocess
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def extract_sequences(pdb_folder, output_fasta):
    sequences = {}
    pdb_parser = PDBParser(QUIET=True)

    with open(output_fasta, 'w') as fasta_file:
            for filename in tqdm.tqdm(os.listdir(pdb_folder)):
                if filename.endswith(".pdb"):
                    filepath = os.path.join(pdb_folder, filename)
                    try:
                        structure = pdb_parser.get_structure(filename, filepath)
                        for model in structure:
                            seq_all = []
                            for chain in model:
                                chain_sequence = []
                                for residue in chain:
                                    if residue.id[0] == " " and residue.resname not in ["HOH"]:
                                        try:
                                            chain_sequence.append(seq1(residue.resname))  # Convert three-letter code to one-letter code
                                        except KeyError:
                                            pass  # Handle rare cases where the residue name is unknown
                                seq_all.append("".join(chain_sequence))
                            seq = "".join(seq_all)
                            seq_id = filename
                            sequences[seq_id] = seq
                            fasta_file.write(f">{seq_id}\n{seq}\n")
                    except:
                        print(filename)

    return sequences


def parse_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        pdb_name = None
        sequence_lines = []

        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if pdb_name is not None:
                    sequences[pdb_name] = ''.join(sequence_lines)
                pdb_name = line[1:]
                sequence_lines = []
            else:
                sequence_lines.append(line)

        # Add the last sequence to the dictionary
        if pdb_name is not None:
            sequences[pdb_name] = ''.join(sequence_lines)

    return sequences

def run_blast(fasta_file_source, fasta_file_target,blast_output):
    # 运行BLASTp比对并输出为tabular格式
    subprocess.run(['blastp', '-task', 'blastp-fast','-query', fasta_file_source, '-subject', fasta_file_target, '-out', blast_output,
                    '-outfmt', '6 qseqid sseqid pident length'])

def parse_blast_output(blast_output,sequences_source, threshold):
    # 使用pandas读取tabular格式的BLAST输出
    blast_df = pd.read_csv(blast_output, sep='\t', names=['query_id', 'subject_id', 'pident', 'length'])

    to_remove = set()

    for index, row in tqdm.tqdm(blast_df.iterrows()):
        query_id = row['query_id']
        subject_id = row['subject_id']
        similarity = row['pident'] / 100
        length = row['length']


        if query_id != subject_id and similarity >= threshold and length/len(sequences_source[query_id])>=1:
            to_remove.add(query_id)

    return to_remove

def find_and_remove_similar_sequences(source_folder,target_folder, threshold=0.7):
    output_fasta_source = "dips.fasta"
    blast_output = "/public/mxp/xiejun/py_project/PPI_affinity/data_final/train_result_85/train_test_bm79_skempi26.txt"
    output_fasta_target = "skempi26.fasta"

    print("loading sequence:")

    sequences_source = extract_sequences(source_folder, output_fasta_source)
    # sequences_target = extract_sequences(target_folder, output_fasta_target)
    exit()
    sequences_source = parse_fasta(output_fasta_source)
    # sequences_target = parse_fasta(output_fasta_target)

    print("running blast:")

    # run_blast(output_fasta_source,output_fasta_target, blast_output)
    print("removing duplicate:")

    to_remove = parse_blast_output(blast_output,sequences_source, threshold)
    print(len(to_remove))
    for filename in tqdm.tqdm(to_remove):
        pdb_path = os.path.join(source_folder, filename)
        if os.path.exists(pdb_path):
            os.remove(pdb_path)
            print(f"Removed {filename}.pdb")

pdb_folder = "/public/mxp/xiejun/py_project/PPI_affinity/PP_1"  # 修改为你的文件夹路径
source_folder = "/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/train_dropdupli_60"
test_folder = "/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/test"
val_folder = "/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/val"
benchmark79_folder = "/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/benchmark79"
all_folder = "/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/dips_plus"

skempi26_folder = "/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/skempi26"

find_and_remove_similar_sequences(all_folder,val_folder)

# train_str="1avx 3fp6 2vlp 2vlo 3qhy 2vln 4gaf 1lw6 2sni 1tm1 3jza 1jiw 4ka2 2ftm 1tm5 1tm3 5j57 5glh 3mzw 2vlq 1mah 1y3c 3ukx 1dhk 1j7v 3e1z 5ma3 2nqd 5ma4 3me2 2sic 1rjc 2lxm 3uzq 1f34 1zvy 2gww 4lgr 3fpu 1d6r 5m2j 2vdb 5bnq 3m18 4k0a 4zkc 4zk9 4kt3 4hff 1y1k 5ma6 1y3b 1a22 3l9j 5xeq 3uzv 1to1 3gty 5dfw 1y34 1g5j 3idc 5d3i 4iop 4gi3 1op9 1gl0 4j2y 1dp5 5k8q 1c9p 1eja 1kxp 1y6k 1yrt 1yru 2voh 4eig 4hrl 2b42 2c1m 2voi 3v96 3wdg 1lx5 1y4a 1y4d 2omv 3zkq 4j32 1zli 3o40 4jeu 2mv7 4w6y 3fju 1y33 1y48 2oul 1ydi 1ldt 1wpx 3fii 5tp6 2hqw 2y9m 3kj0 3kj1 3kj2 3qwq 1m5n 3w8i 4z9k 1fmo 2cpk 3ul0 3eba 3ukz 2oza 2m86 3ul1 1ri8 4f38 5jds 3aon 5j56 2leh 5imm 4w6w 4jw2 2xhe 5my6 3cqc 3cqg 4udm 5vko 3dvm 3c4p 4xwj 3c4o 4ol0 5uul 4zw2 1vg0 1xg2 4w6x 2uuy 2lfw 1m10 3p92 4uem 1yvb 3zu7 2hsq 2l1l 1l8c 3rgf 3zyi 5vz4 4jeh 2v9t 2k3u 5xco 1ta3 3f50 3g7a 3o43 4l0p 5cxb 2ld7 3nvn 5omn 4c2a 5hpk 4yj4 1v18 1xdt 1zv5 3k8p 4x7s 5uzu 2kxw 4k5a 2aq2 4dzv 5nus 1wa8 5lhn 3idb 3pnr 1tdq 4jeg 2lkm 2m0j 2l2l 1vet 1nw9 1r8u 2g81 2wfx 1syq 2l29 2p8q 5e95 2x1x 2z58 4hgm 4kr0 4yeb 5djt 3u82 4g6u 2wh6 2j12 3doe 4ap2 3zwz 3h8k 2wd5 4ekd 5f5s 1h1v 2p43 2p49 3ouw 5mtn 4rs1 4nzl 5ua4 1y6n 4ct0 3wa5 4c9b 4bd9 3gqi 5m72 1kbh 1p6a 2vsm 2yq7 5dc4 2k42 5v69 1h59 4lad 4plo 1t01 1gua 2hle 3ajb 1djs 1oyh 3ro2 4krl 4cj1 2m5a 1rke 3beg 4pbz 5nqg 2ka6 2k2s 4g35 4kt1 3tu3 2lww 3oun 5oaq 2ka4 2a78 2n1d 5d1m 2f4m 2kj4 1zvh 5b64 2w84 1u0i 5ixd 5xv8 1g6v 2wg4 2uyz 2m0g 4yn0 5jw9 3qsk 3bs5 1mcv 4cj0 4dj9 4fza 2omt 4wen 2ju0 2b87 2f31 5vkl 2p45 2p47 2p48 1g9i 1i4e 1smf 2omx 5gjk 5un7 5inb 5h7y 2ru4 4pas 3ojm 5eg3 3tz1 5d1l 5fr2 2rvb 2xpx 4yh7 4pou 5v5h 3w8h 3sri 5o2t 5nqf 2rmk 1y6m 3m63 1pjm 5d1k 5lxm 2kwi 4h6j 2lox 3znz 1l4d 1xt9 2omy 3ohm 4bru 3gxu 2vog 1ry7 1u0s 2mnu 4c4p 4gn4 4b1y 3v6b 4yyp 3zet 2wel 3blh 4hep 1pd7 4u4c 4l67 2roz 5hps 1lzw 5mtj 3cx8 2ly4 1jgn 5wos 4y61 2wy8 4hdo 2k2u 2khs 4etp 5eo9 4u32 2qxv 3p71 1grn 3mca 2b7c 2omz 3lbx 1q68 4nqw 5wuj 2arp 3tkl 3zo0 3kuc 5li1 5kxh 3tei 4awx 2kwj 4nzw 3alz 3cx7 4dbg 3au4 4a1u 3tac 4dt1 2g2w 2jti 4yoc 1wlp 4apf 2ixq 4zgy 5wgg 5wrv 3ddc 5ky5 2wp3 4cmm 5uk5 5imk 2mzd 1q69 4mrt 2n2h 5npo 1t44 3kv4 5dob 5mv9 1o9a 2n01 3tg1 4c5g 4pqt 1y8n 4cu4 2g2u 2v4z 2wwk 4ehq 5vwy 4y5o 2pon 3hct 4xxb 3o5t 3p95 3qq8 1jh4 3u9z 5xbf 2ot3 4dxa 5b78 4yc7 2v52 2mcn 1mzw 1lp1 5e6p 4yvq 4giq 2rr3 5me5 2l9s 2btf 2wo3 4b93 5tar 2kwk 4aqe 4u97 4zgq 2k3s 2fyl 2hd5 4zrj 2lqc 2dsp 2l14 5ajj 1usu 5fr1 3sjh 4g01 3oiq 2ptt 4b1x 3cx6 2v3b 4bkx 4js0 5szh 4yl8 5b76 5ky9 4did 2c7m 3gb8 3gj6 5ky0 3wwn 4m0w 3uyo 4wem 4uf1 1zsg 4ds8 5ij0 5lz6 3o34 1fy8 2mur 4hcp 1xr0 2omw 5b75 5ky4 2lpb 2wo2 5b77 5bn5 2knb 3qc8 3tgk 1f7z 4h5s 1p9d 2ver 5ij9 4pw9 2kxh 5u4k 1wq1 2den 2mej 6amb 4d0n 5lz3 2mre 2b12 3bn3 2jy6 6b6u 2v8s 2lp0 2ktf 2ruk 5ee5 2v6x 2kt5 2xgy 2a24 1ozs 1t5z 2m55 1xj7 2djy 2mws 2kri 5gtb 2kwo 2kvq 2kwu 2kqs 4ika 4nm3 4nu1 2l0t 1f5r 3d7t 1yx6 2lqh 2px9 2kwv 3olm 2b0z 2hth 4hcn 1q5w 2mbb 2mro 1otr 6ba6 2k6d 2l0f 2fuh 1tlh 2dx5 4tq1 1yx5 1wrd 2k79 2k8b 2k8c 1u5s 3n06 2few 3mzg 3n0p 5g15 5jjd 5l21 5vmo 5xln 5zau 6akm 6bw9 6er6 6f0f 6f2g 6fbx 6fuz 6fv0 6gho 6gum 6gvk 6gvl 6h16 6h46 6h47 6har 6her 6i2m 6idx 6ihb 6im9 6ire 6iu7 6iua 6ivu 6iw8 6iwa 6iwd 6j4o 6j4s 6jcs 6jwj 6k06 6kbm 6kbr 6m7l 6mc9 6mud 6n85 6ne1 6ne2 6ne4 6on9 6oqj 6oqk 6osw 6ov2 6pnp 6pnq 6qb3 6r2g 6umt"
# val_str="1ay7 1dfj 1f3v 1h0t 1l4z 1p69 1q0w 1rkc 1shy 1sq0 1t0p 1te1 1tm7 1vrk 1wqj 1y3d 1z92 2a7u 2jod 2k7a 2kbw 2kc8 2kgx 2lvo 2lz6 2m04 2mfq 2mj5 2nbv 2ptc 2qc1 2qna 2rms 2vay 3f1p 3fhc 3gc3 3h3g 3kyi 3lms 3mj7 3ol2 3oux 3ul4 3vyr 3wqb 4a49 4an7 4bfi 4brw 4c4k 4euk 4iu3 4lzx 4m1l 4m5f 4n7z 4nso 4qlp 4qxa 4xl5 5cyk 5elu 5h3j 5imt 5j8h 5jw7 5mv8 5o90 5oen 5oy9 5oyl 5szi 5tzp 5u4m 5vzm 5xod 5yi8 6aaf 6azp 6bmt 6e3j 6fc3 6fp7 6fub 6g04 6gd5 6h9n 6hul 6imf"
# test_str="1wa7 2jgz 1an1 1axi 5ml9 3n4i 1zgu 4zqu 4p3y 1dpj 1pjn 5tvq 2lp4 1wr1 2jt4 2n8j 5yr0 3di3 2o8v 2k5b 6i3f 5g1x 3h6g 2o3b 1uel 1ees 1tm4 3kud 4d0g 6eg0 5eql 4rt6 3gj3 1t63 3ona 6ff3 3vv2 6fud 2wwx 2luh 5mtm 1veu 4iyp 5l8j 2p44 5szk 6arq 6cbp 5xoc 3oky 1tba 5nwm 1fle 3kw5 6aw2 3uyp 6d13 4x33 5k22 2rnr 5szj 2k8f 4rey 4apx 6isc 4je4 4i6l 2vda 5f4e 6gbg 2omu 2xtt 3ch5 4f48 6f9s 2n73 1t6b 1sb0 6fg8 5v6a 4exp 5ufe 6e3i 3knb 1jtd 4uyq 3bh6 2ftl 4wnd 5wpa"
# train_list=train_str.split(" ")
# val_list=val_str.split(" ")
# test_list=test_str.split(" ")


# print("hhh")
# # 定义源文件夹和目标文件夹
# source_folder = "PP_1"
# test_folder = "test_pdb"
# val_folder = "val_pdb"

# # 创建目标文件夹，如果不存在的话
# os.makedirs(test_folder, exist_ok=True)
# os.makedirs(val_folder, exist_ok=True)

# def move_files(pdb_list, destination_folder):
#     for pdb_name in pdb_list:
#         source_path = os.path.join(source_folder, pdb_name+".ent.pdb")
#         destination_path = os.path.join(destination_folder, pdb_name+".ent.pdb")
#         if os.path.exists(source_path):
#             shutil.move(source_path, destination_path)
#             print("Moved "+pdb_name+".ent.pdb to "+destination_folder)
#         else:
#             print(pdb_name+".ent.pdb not found in "+source_folder)

# # 移动测试集文件
# move_files(test_list, test_folder)

# # 移动验证集文件
# move_files(val_list, val_folder)

