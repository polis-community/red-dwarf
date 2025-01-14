from reddwarf.polis_pandas import PolisClient
import numpy as np

client = PolisClient(is_strict_moderation=False)
client.load_data('sample_data/votes.json')
client.load_data('sample_data/comments.json')
print('loading matrix...')
# print(client.get_matrix())
# print(client.get_active_statement_ids())
# print(client.get_matrix()[53])
# client.matrix = None
# print(client.get_matrix(is_filtered=True)[53].sum())
# print(client.get_matrix(is_filtered=True)[53].isna().sum())
# print(client.get_matrix(is_filtered=True)[53].isna().all())
# client.impute_missing_votes()
# print(client.get_matrix().columns.max())
# print(client.get_matrix().index.max())
client.get_matrix(is_filtered=True)
# print(client.get_matrix(is_filtered=False))

# print('inputing...')
client.impute_missing_votes()
print(client.matrix)
# print(client.matrix)
# print(np.round(client.matrix, decimals=2))
