import sys
import torch
# in_file = sys.argv[1]
# entity_in_file = sys.argv[2]
# relation_in_file = sys.argv[3]
# entity = open(entity_in_file, 'r', encoding='utf-8')
# relation = open(relation_in_file, 'r', encoding='utf-8')
#
# entity2id = dict()
# for line in entity:
#     try:
#         ent, id = line.strip().split('\t')
#     except:
#         continue
#     entity2id[ent] = id
#
# relation2id = dict()
# for idx,line in enumerate(relation):
#     try:
#         rel, id = line.strip().split('\t')
#     except:
#         continue
#     relation2id[rel] = str(idx-1)
#
# for idx,line in enumerate(open(in_file, 'r', encoding='utf-8')):
#     ent1, rel, ent2 = line.strip().split('\t')
#     ent1id, relid, ent2id = entity2id[ent1], relation2id[rel], entity2id[ent2]
#     print(ent1id+' '+ent2id+' '+relid)

# coke_embed = torch.Tensor
state_dict = torch.load('./transe.ckpt')
for key, value in state_dict.items():
    print(key)
    if key == 'ent_embeddings.weight':
        entity_embed = value
    if key == 'rel_embeddings.weight':
        rel_embed = value

coke_embed = torch.cat((entity_embed, rel_embed), dim=0)
print (rel_embed)
print(coke_embed)
print(coke_embed.shape)



