import torch
import torch.nn as nn
import torch.nn.functional as f

class Find_Module(nn.Module):
	def __init__(self, emb_weight, padding_idx, emb_dim, hidden_dim, cuda, dropout=0.2, sliding_win_size=3):
		super(Find_Module, self).__init__()

		self.padding_idx = padding_idx
		self.emb_dim = emb_dim
		self.hidden_dim = hidden_dim
		self.sliding_win_size = sliding_win_size
		self.lstm_layers = lstm_layers
		self.dropout = dropout
		self.cuda = cuda

		self.dropout = nn.Dropout(p=self.dropout)
		self.embeddings = nn.Embedding.from_pretrained(emb_weight, freeze=False, padding_idx=self.padding_idx)
		self.bilstm = nn.LSTM(self.emb_dim, self.hidden_dim, bidirectional=True, batch_first=True)
		
		self.attention_matrix = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
		temp_att_vector = torch.zeros(2*self.hidden_dim, 1)
		nn.init.xavier_uniform_(temp_att_vector)
		self.attention_vector = nn.Parameter(temp_att_vector, requires_grad=True)
		self.softmax = nn.Softmax(dim=2)

		diagonal_vector = torch.zeros(2*self.hidden_dim)
		nn.init.xavier_uniform_(diagonal_vector)
		self.feature_weight_matrix = nn.Parameter(torch.diag(diagonal=diagonal_vector), requires_grad=True)

		temp_sliding_window_weight = torch.zeros(self.sliding_win_size, 1)
		nn.init.xavier_uniform_(temp_sliding_window_weight)
		self.sliding_window_weight = nn.Parameter(temp_sliding_window_weight, requires_grad=True)
	
	
	def attention_pooling(self, hidden_states):
		"""
			hidden_states = N x seq_len x 2*hidden_dim
		"""
		linear_transform = self.attention_matrix(hidden_states) # linear_transform = N x seq_len x 2*hidden_dim
		tanh_tensor = nn.Tanh(linear_transform) # element wise tanh
		batch_dot_products = torch.matmul(tanh_tensor, self.attention_vector).permute(0,2,1) # batch_dot_product = batch x 1 x seq_len (initially it is batch x seq_len x 1)
		batch_soft_max = self.softmax(batch_dot_products) #apply softmax along row
		pooled_rep = torch.bmm(batch_soft_max, hidden_states) # pooled_rep = batch x 1 x 2*hiddem_dim --> one per x :)

		return pooled_rep

	def get_hidden_states(self, seqs):
		"""
			seqs = N x seq_len
		"""
		seq_embs = self.embeddings(seqs) # seq_embs = N x seq_len x hidden_dim
		seq_embs = self.dropout(seq_embs)
		seq_output, _ =  self.bilstm(seq_embs) # seq_output = seq_len x N x 2*hidden_dim
		hidden_states = seq_output.permute(1, 0, 2) # seq_ouput = N x seq_len x 2*hidden_dim

		return hidden_states
	
	def pre_train_get_similarity(self, hidden_states, query_vectors):
		"""
			hidden_states = N x seq_len x 2*hidden_dim -> output of get_hidden_states for original sequence
			query_vectors = N x 1 x 2*hidden_dim -> output of attention_pooling for query sequences

			goal: get machine to be able to identify when query starts and stops in sentence, given that query vector is a pooled representation
			of the individual tokens
		"""
		updated_query_vectors = torch.matmul(query_vectors, self.feature_weight_matrix) #updated_query_vectors = N x 1 x 2*hidden_dim
		batch, seq_len, hidden_dim = hidden_states.shape
		if self.sliding_win_size == 3:
			forward_bigram_hidden_states = torch.zeros(batch, seq_len, hidden_dim) #geting pooled hidden states for w_i, w_i+1
			backward_bigram_hidden_states = torch.zeros(batch, seq_len, hidden_dim) #geting pooled hidden states for w_i-1, w_i
			for i, matrix in enumerate(hidden_states):
				for j in range(seq_len):
					if j < seq_len-1:
						forward_bigram_hidden_states[i, j, :] = attention_pooling(hidden_states[i, j:j+2, :].unsqueeze(0)).squeeze(0)
					else:
						forward_bigram_hidden_states[i, j, :] = hidden_states[i, j, :]
					
					if j > 0:
						backward_bigram_hidden_states[i, j, :] = attention_pooling(hidden_states[i, j-1:j+1, :].unsqueeze(0)).squeeze(0)
					else:
						backward_bigram_hidden_states[i, j, :] = hidden_states[i, j, :]
			
			if self.cuda:
				device = torch.device("cuda")
				forward_bigram_hidden_states = forward_bigram_hidden_states.to(device)
				backward_bigram_hidden_states = backward_bigram_hidden_states.to(device)
			
			updated_hs = torch.matmul(hidden_states, self.feature_weight_matrix) #updated_hs = N x seq_len x 2*hidden_dim
			updated_forward_hs = torch.matmul(forward_bigram_hidden_states, self.feature_weight_matrix) #updated_forward_hs = N x seq_len x 2*hidden_dim
			updated_backward_hs = torch.matmul(backward_bigram_hidden_states, self.feature_weight_matrix) #updated_backward_hs = N x seq_len x 2*hidden_dim

			normalized_hidden_states = f.normalize(updated_hs, p=2, dim=2) # normalizing rows of each matrix in the batch
			normalized_forward_hidden_states = f.normalize(updated_forward_hs, p=2, dim=2) # normalizing rows of each matrix in the batch
			normalize_backward_hidden_states = f.normalize(updated_backward_hs, p=2, dim=2) # normalizing rows of each matrix in the batch
			normalized_query_vectors = f.normalize(updated_query_vectors, p=2, dim=2) # normalizing rows of each matrix in the batch
			normalized_query_vectors = normalized_query_vectors.permute(0, 2, 1) # arranging query_vectors to be N x 2*hidden_dim x 1

			hs_cosine = torch.matmul(normalized_hidden_states, normalized_query_vectors).squeeze(2) # dot product between each normalized row and the normalized query vector 
			                                                                                        # (seq_len x 2*hidden_dim) T (2*hidden_dim x 1) = (seq_len x 1)
																						            # since normalized, dot product is cosine
			fwd_hs_cosine = torch.matmul(normalize_backward_hidden_states, normalized_query_vectors).squeeze(2) # same as above (N x seq_len), squeezing for below
			bwd_hs_cosine = torch.matmul(normalize_backward_hidden_states, normalized_query_vectors).squeeze(2) # same as above (N x seq_len)

			combined_cosines = torch.zeros(batch, seq_len, self.sliding_win_size) # building cosine similarity of sliding window
			for batch_n, row in enumerate(hs_cosine):
				cosines = [row, fwd_hs_cosine[batch_n], bwd_hs_cosine[batch_n]]
				stacked_cosines = torch.stack(cosines, dim=1)
				combined_cosines[batch_n] = stacked_cosines

			if self.cuda:
				device = torch.device("cuda")
				combined_cosines = combined_cosines.to(device)
			
			similarity_scores = torch.matmul(combined_cosines, self.sliding_window_weight) # similarity_scores = N x seq_len x 1
			
			return similarity_scores
		
	def find_forward(self, seqs, queries):
		""""
			seqs : N * seq_len
			queries : N * seq_len
		""""
		query_encodings = self.get_hidden_states(queries) # N x seq_len x 2*hidden_dim
		seq_encodings = self.get_hidden_states(seqs) # N x seq_len x 2*hidden_dim

		pooled_query_encodings = self.attention_pooling(query_encodings) # N x 1 x 2*hidden_dim
		seq_similarities = self.pre_train_get_similarity(seqs, pooled_query_encodings).squeeze(2) # N x seq_len

		return seq_similarities
	
	def compute_sim_loss_query(self, query_vector, pos_tensor, neg_tensor):
		updated_query_vector = torch.matmul(query_vector, self.feature_weight_matrix) # 1 x 2*hidden_dim
		updated_pos_tensor = torch.matmul(pos_tensor, self.feature_weight_matrix) # pos_count x 2*hidden_dim
		updated_neg_tensor = torch.matmul(neg_tensor, self.feature_weight_matrix) # neg_count x 2*hiddem_dim
		
		normalized_query_vector = f.normalize(updated_query_vector, p=2, dim=1).permute(1,0) # 2*hidden_dim x 1
		normalized_pos_tensor = f.normalize(updated_pos_tensor, p=2, dim=1)
		normalized_neg_tensor = f.normalize(updated_neg_tensor, p=2, dim=1)

		pos_scores = torch.matmul(normalized_pos_tensor, normalized_query_vector).squeeze(1) # pos_count 
		neg_scores = torch.matmul(normalized_neg_tensor, normalized_query_vector).squeeze(1) # neg_count

		pos_score = torch.max(pos_scores)
		neg_score = torch.max(neg_scores)

		return (pos_score, neg_score)

	def sim_forward(self, queries, labels):
		queries_by_label = {}
		query_hidden_states = self.get_hidden_states(queries) # N x seq_len x 2*hidden_dim
		pooled_reps = attention_pooling(query_hidden_states).squeeze(1) # N  x 2*hidden_dim
		for i, label in enumerate(labels):
			if label in queries_by_label:
				queries_by_label[label][i] = pooled_reps[i]
			else:
				queries_by_label[label] = {i : pooled_reps[i]}
		
		query_losses = []
		for i, label in enumerate(labels):
			query_rep = pooled_reps[i].unsqueeze(0)
			pos_tensor = torch.stack([queries_by_label[label][j] for j in range(len(queries_by_label[label])) if j != i]) # pos_count x 2*hidden_dim
			neg_tensor_array = []
			for label_2 in queries_by_label:
				if label_2 != label:
					for key in queries_by_label[label]:
						neg_tensor_array.append(queries_by_label[label][key])
			neg_tensor = torch.stack(neg_tensor_array) # neg_count x 2*hidden_dim
			query_losses.append(self.compute_sim_loss_query(query_rep, pos_tensor, neg_tensor))
		
		return query_losses
		
		

		