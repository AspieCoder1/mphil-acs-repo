#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

import os
import pandas as pd
import numpy as np


def pre_process_kegg_kg():
    dataset_dir = os.path.sep.join(["DTI-data", "KEGG_MED"])
    dti = pd.read_table(os.path.sep.join([dataset_dir, "dt_kegg_med.txt"]), header=None)
    # print(dti)
    drug_list = set(dti[0])
    # print(len(drug_list))  # 4284
    target_list = set(dti[2])
    # print(len(target_list))  # 945
    kg = pd.read_table(os.path.sep.join([dataset_dir, "kegg_kg.txt"]), header=None)
    kg_drug = kg[kg[1].str.contains("PATHWAY_DRUG")]
    # print(kg_drug)
    drug_pathway = []
    for row in kg_drug.itertuples():
        if getattr(row, "_3") in drug_list:
            drug_pathway.append([getattr(row, "_1"), getattr(row, "_3")])
    # print("drug_pathway", drug_pathway)

    kg_gene = kg[kg[1].str.contains("PATHWAY_GENE")]
    gene_pathway = []
    for row in kg_gene.itertuples():
        if getattr(row, "_3") in target_list:
            gene_pathway.append([getattr(row, "_1"), getattr(row, "_3")])
    # print("gene_pathway", gene_pathway)

    drug_dict = {}
    for index, i in enumerate(drug_list):
        drug_dict[i] = index
    print("drug_dict", drug_dict)

    target_dict = {}
    for index, i in enumerate(target_list):
        target_dict[i] = index
    # print("target_dict", target_dict)

    pathway1 = np.array(drug_pathway)[:, 0]
    pathway2 = np.array(gene_pathway)[:, 0]
    pathway = set(np.concatenate((pathway1, pathway2), axis=0))
    # print(len(pathway))  # 105

    pathway_dict = {}
    for index, i in enumerate(pathway):
        pathway_dict[i] = index
    # print("pathway_dict", pathway_dict)

    drug_pathway_processed = []
    for i in drug_pathway:
        if i[0] in pathway_dict:
            drug_pathway_processed.append([pathway_dict[i[0]], drug_dict[i[1]]])
    # print("drug_pathway_processed", drug_pathway_processed)
    # print(len(drug_pathway_processed))  # 7087

    target_pathway_processed = []
    for i in gene_pathway:
        if i[0] in pathway_dict:
            target_pathway_processed.append([pathway_dict[i[0]], target_dict[i[1]]])
    # print("target_pathway_processed", target_pathway_processed)
    # print(len(target_pathway_processed))  # 3390

    drug_target_processed = []
    for row in dti.itertuples():
        drug_target_processed.append(
            [drug_dict[getattr(row, "_1")], target_dict[getattr(row, "_3")]]
        )
    # print("drug_target_processed", drug_target_processed)
    # print(len(drug_target_processed))  # 12112

    H_drug_pathway = np.zeros((len(drug_dict), len(pathway_dict)), dtype=np.int8)
    for i in drug_pathway_processed:
        H_drug_pathway[i[1], i[0]] = 1

    H_target_pathway = np.zeros((len(target_dict), len(pathway_dict)), dtype=np.int8)
    for i in target_pathway_processed:
        H_target_pathway[i[1], i[0]] = 1
    # print(len(H_target_pathway), len(H_target_pathway[0]))  # 945*105

    with open(
        os.path.sep.join([dataset_dir, "drug_target_interaction.txt"]), "w"
    ) as f0:
        for i in range(len(drug_target_processed)):
            s = str(drug_target_processed[i]).replace("[", " ").replace("]", " ")
            s = s.replace("'", " ").replace(",", "") + "\n"
            f0.write(s)

    np.savetxt(
        os.path.sep.join([dataset_dir, "drug_pathway.txt"]), H_drug_pathway, fmt="%d"
    )
    np.savetxt(
        os.path.sep.join([dataset_dir, "protein_pathway.txt"]),
        H_target_pathway,
        fmt="%d",
    )

    disease_drug = kg[kg[1].str.contains("DRUG_EFFICACY_DISEASE")]
    # print(disease_drug)
    drug_disease = []
    for row in disease_drug.itertuples():
        if getattr(row, "_1") in drug_list:
            drug_disease.append([getattr(row, "_1"), getattr(row, "_3")])
    print("drug_disease", drug_disease)

    disease_target = kg[kg[1].str.contains("GENE_DISEASE")]
    # print(disease_target)
    target_disease = []
    for row in disease_target.itertuples():
        if getattr(row, "_1") in target_list:
            target_disease.append([getattr(row, "_1"), getattr(row, "_3")])
    print("target_disease", target_disease)

    disease1 = np.array(drug_disease)[:, -1]
    disease2 = np.array(target_disease)[:, -1]
    disease = set(np.concatenate((disease1, disease2), axis=0))
    # print(len(disease))  # 360

    disease_dict = {}
    for index, i in enumerate(disease):
        disease_dict[i] = index
    print("disease_dict", disease_dict)

    drug_disease_processed = []
    for i in drug_disease:
        if i[1] in disease_dict:
            drug_disease_processed.append([disease_dict[i[1]], drug_dict[i[0]]])
    # print("drug_disease_processed", drug_disease_processed)
    # print(len(drug_disease_processed))  # 365

    target_disease_processed = []
    for i in target_disease:
        if i[1] in disease_dict:
            target_disease_processed.append([disease_dict[i[1]], target_dict[i[0]]])
    # print("target_disease_processed", target_disease_processed)
    # print(len(target_disease_processed))  # 433

    H_drug_disease = np.zeros((len(drug_dict), len(disease_dict)), dtype=np.int8)
    for i in drug_disease_processed:
        H_drug_disease[i[1], i[0]] = 1
    # print(H_drug_disease)
    # print(len(H_drug_disease), len(H_drug_disease[0]))  # 4284*360

    H_target_disease = np.zeros((len(target_dict), len(disease_dict)), dtype=np.int8)
    for i in target_disease_processed:
        H_target_disease[i[1], i[0]] = 1
    # print(len(H_target_disease), len(H_target_disease[0]))  # 945*360

    np.savetxt(
        os.path.sep.join([dataset_dir, "drug_disease.txt"]),
        H_drug_disease,
        fmt="%d",
    )
    np.savetxt(
        os.path.sep.join([dataset_dir, "protein_disease.txt"]),
        H_target_disease,
        fmt="%d",
    )


def pre_process_kegg_dt():
    dataset_dir = os.path.sep.join(["DTI-data", "KEGG_MED"])
    dti = pd.read_table(os.path.sep.join([dataset_dir, "dt_kegg_med.txt"]), header=None)
    drug_list = set(dti[0])
    target_list = set(dti[2])

    drug_dict = {}
    for index, i in enumerate(drug_list):
        drug_dict[i] = index
    print("drug_dict", drug_dict)

    target_dict = {}
    for index, i in enumerate(target_list):
        target_dict[i] = index
    print("target_dict", target_dict)
    H_drug_target = np.zeros((len(drug_list), len(target_dict)))

    for row in dti.itertuples():
        H_drug_target[drug_dict[row[1]], target_dict[row[3]]] = 1

    np.savetxt(
        os.path.sep.join([dataset_dir, "drug_protein.txt"]),
        H_drug_target,
        fmt="%d",
    )


pre_process_kegg_kg()
pre_process_kegg_dt()
