# NMTmodel
易于理解，完善的seq2seq 模型

问题1：
` Tensor("IteratorGetNext:0", shape=(?, ?), dtype=string, device=/device:CPU:0) 
 must be from the same graph as Tensor("string_to_index/hash_table/hash_table:0", shape=(), dtype=resource).`

问题2：
`Exception: logits and labels must be broadcastable: logits_size=[44,3603] labels_size=[44,3]`

