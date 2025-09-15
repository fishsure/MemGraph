import copy
import json
import os
import pickle
import torch
import datetime
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.spatial.distance import pdist, squareform

from models.retriever import RetrieverModel
from prompts.pre_process import load_get_corpus_fn, load_get_query_fn
from runners.summary import llm_summarize, llm_batch_summarize, llm_batch_summarize_parallel
class MemGraphRunner:

    @staticmethod
    def parse_args(parser):
        # BGE retriever related arguments
        parser.add_argument("--base_retriever_path",
                            default="models/bge-m3")
        parser.add_argument("--retriever_pooling", default="average")
        parser.add_argument("--retriever_normalize", type=int, default=1)
        
        # Summary related arguments
        parser.add_argument("--summary", type=int, default=0)
        parser.add_argument("--summary_llm_name", default="models/Qwen3-8B")
        parser.add_argument("--summary_k",type=int, default=50)
        parser.add_argument("--summary_batch_size", type=int, default=100, help="Batch size for parallel summary processing")
        parser.add_argument("--use_parallel_summary", type=int, default=1, help="Whether to use parallel batch processing for summary")
        parser.add_argument("--auto_cleanup_llm", type=int, default=1, help="Whether to automatically cleanup LLM after summary generation")
        parser.add_argument("--enable_token_check", type=int, default=1, help="Whether to enable token length checking and auto-truncation")
        parser.add_argument("--reserve_tokens", type=int, default=500, help="Number of tokens to reserve for generation")
        
        # New parameters for k-means clustering and time decay
        parser.add_argument("--n_clusters", type=int, default=10)
        parser.add_argument("--use_kmeans", type=int, default=1)
        parser.add_argument("--kmeans_use_embedding", type=int, default=0, help="Whether to use embedding for k-means clustering (1) or TF-IDF (0)")
        parser.add_argument("--kmeans_select_method", default="center", choices=["center", "relevance"], 
                          help="K-means cluster representative selection method: center (closest to centroid) or relevance (most relevant to query)")
        parser.add_argument("--use_recency", type=int, default=1)
        parser.add_argument("--time_decay_lambda", type=float, default=0.0)

        # New parameters for graph-based random walk
        parser.add_argument("--use_graph_walk", type=int, default=0, help="Whether to use graph-based random walk (1) or traditional method (0)")
        parser.add_argument("--walk_start_method", default="latest", choices=["latest", "semantic"], 
                          help="Random walk start method: latest (most recent) or semantic (most similar to query)")
        parser.add_argument("--walk_length", type=int, default=10, help="Maximum number of nodes to visit in random walk")
        parser.add_argument("--semantic_alpha", type=float, default=1.0, help="Weight for semantic similarity in transition probability")
        parser.add_argument("--time_lambda1", type=float, default=0.01, help="Time decay parameter for edge transition")
        parser.add_argument("--time_lambda2", type=float, default=0.01, help="Time decay parameter for distance to latest document")

        return parser

    def __init__(self, opts) -> None:
        self.task = opts.task
        self.get_query = load_get_query_fn(self.task)
        self.get_corpus = load_get_corpus_fn(self.task)
        self.use_date = opts.source.endswith('date')

        self.data_addr = opts.data_addr
        self.output_addr = opts.output_addr
        self.data_split = opts.data_split
        self.source = opts.source
        self.topk = opts.topk
        self.device = opts.device
        self.batch_size = opts.batch_size
        
        # Summary related parameters
        self.summary = getattr(opts, "summary", 0)
        self.summary_k = getattr(opts, "summary_k", 50)
        self.summary_llm_name = getattr(opts, "summary_llm_name", "Qwen/Qwen2.5-7B-Instruct")
        self.summary_batch_size = getattr(opts, "summary_batch_size", 100)
        self.use_parallel_summary = getattr(opts, "use_parallel_summary", 1)
        self.auto_cleanup_llm = getattr(opts, "auto_cleanup_llm", 1)
        self.enable_token_check = getattr(opts, "enable_token_check", 1)
        self.reserve_tokens = getattr(opts, "reserve_tokens", 500)
        
        # Memory management related
        self.llm_cleaned = False
        
        # New parameters for k-means clustering and time decay
        self.n_clusters = getattr(opts, "n_clusters", 10)
        self.use_kmeans = getattr(opts, "use_kmeans", 1)
        self.kmeans_use_embedding = getattr(opts, "kmeans_use_embedding", 0)
        self.kmeans_select_method = getattr(opts, "kmeans_select_method", "center")
        self.use_recency = getattr(opts, "use_recency", 1)
        self.time_decay_lambda = getattr(opts, "time_decay_lambda", 0.0)

        # New parameters for graph-based random walk
        self.use_graph_walk = getattr(opts, "use_graph_walk", 0)
        self.walk_start_method = getattr(opts, "walk_start_method", "latest")
        self.walk_length = getattr(opts, "walk_length", 10)
        self.semantic_alpha = getattr(opts, "semantic_alpha", 1.0)
        self.time_lambda1 = getattr(opts, "time_lambda1", 0.01)
        self.time_lambda2 = getattr(opts, "time_lambda2", 0.01)

        # Load user data
        self.load_user(opts)
        
        # Initialize LLM for summary if needed
        if self.summary:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            self.agent_tokenizer = AutoTokenizer.from_pretrained(opts.summary_llm_name)
            self.agent_tokenizer.padding_side = "left"
            if self.agent_tokenizer.pad_token_id is None:
                self.agent_tokenizer.pad_token = self.agent_tokenizer.eos_token
                self.agent_tokenizer.pad_token_id = self.agent_tokenizer.eos_token_id
            
            self.agent_llm = LLM(model=opts.summary_llm_name, gpu_memory_utilization=0.85, max_model_len=8192)
            self.agent_sampling_params = SamplingParams(seed=42, temperature=0, best_of=1, max_tokens=500)
        
        # Initialize BGE retriever
        self.retriever = RetrieverModel(
            ret_type='dense',
            model_path=opts.base_retriever_path,
            base_model_path=opts.base_retriever_path,
            user2id=self.user2id,
            user_emb_path=self.user_emb_path,
            batch_size=self.batch_size,
            device=self.device,
            max_length=opts.max_length,
            pooling=opts.retriever_pooling,
            normalize=opts.retriever_normalize).eval().to(self.device)

        # Load dataset
        input_path = os.path.join(self.data_addr, opts.data_split, self.source,
                                  'rank_merge.json')

        self.dataset = json.load(open(input_path, 'r'))
        print("orig datasize:{}".format(len(self.dataset)))
        self.dataset = self.dataset[opts.begin_idx:opts.end_idx]

    def cleanup_llm(self):
        """Clean up GPU memory occupied by LLM"""
        if hasattr(self, 'agent_llm') and not self.llm_cleaned:
            print("[Memory Management] Releasing GPU memory occupied by LLM...")
            try:
                # Delete LLM related objects
                if hasattr(self, 'agent_llm'):
                    del self.agent_llm
                if hasattr(self, 'agent_tokenizer'):
                    del self.agent_tokenizer
                if hasattr(self, 'agent_sampling_params'):
                    del self.agent_sampling_params
                
                # Clean up Python garbage collection
                import gc
                gc.collect()
                
                # Clean up CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"[Memory Management] Current GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                
                print("[Memory Management] LLM memory cleanup completed")
                self.llm_cleaned = True
            except Exception as e:
                print(f"[Memory Management] Error occurred while cleaning up LLM: {e}")
                # Mark as cleaned even if error occurs to avoid repeated attempts
                self.llm_cleaned = True

    def load_user(self, opts):
        """Load user vocabulary and embeddings"""
        vocab_addr = os.path.join(opts.data_addr, f"dev/{opts.source}")

        with open(os.path.join(vocab_addr, 'user_vocab.pkl'), 'rb') as file:
            self.user_vocab = pickle.load(file)

        with open(os.path.join(vocab_addr, 'user2id.pkl'), 'rb') as file:
            self.user2id = pickle.load(file)

        assert len(self.user_vocab) == len(self.user2id)

        self.user_emb_path = os.path.join(opts.data_addr,
                                          f"dev/{opts.source}/user_emb",
                                          "20241009-120906.pt")

    def _get_time_weight(self, profile_list, ref_date=None):
        """
        Calculate time decay weight
        profile_list: list of dict, each with 'date' field (YYYY-MM-DD or year as int)
        ref_date: datetime.date, reference date. If None, use max(profile_list)
        Returns: list of float, time weights
        """
        if ref_date is None:
            # Use the latest time among all profiles as reference
            dates = []
            for p in profile_list:
                if isinstance(p['date'], int):
                    # For LaMP_1_time and LaMP_5_time, date is just a year
                    dates.append(datetime.date(p['date'], 1, 1))
                else:
                    # For other tasks, date is in YYYY-MM-DD format
                    dates.append(datetime.datetime.strptime(p['date'], "%Y-%m-%d").date())
            ref_date = max(dates)
        
        weights = []
        for p in profile_list:
            if isinstance(p['date'], int):
                # For LaMP_1_time and LaMP_5_time, date is just a year
                d = datetime.date(p['date'], 1, 1)
            else:
                # For other tasks, date is in YYYY-MM-DD format
                d = datetime.datetime.strptime(p['date'], "%Y-%m-%d").date()
            
            delta_days = (ref_date - d).days
            weight = np.exp(-self.time_decay_lambda * delta_days)
            weights.append(weight)
        return weights

    def _get_profile_embeddings(self, profile_texts):
        """
        Get profile embedding vectors using BGE-M3 model
        
        Args:
            profile_texts: List of profile texts
            
        Returns:
            numpy array: Profile embedding matrix
        """
        profile_embeddings = []
        
        # Dynamically detect embedding dimension
        if not hasattr(self, '_embedding_dim'):
            try:
                # Use a simple test text to detect embedding dimension
                test_text = "test text for dimension detection"
                test_tokens = self.retriever.tokenizer(
                    test_text,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    test_emb = self.retriever.encode(test_tokens)
                    if isinstance(test_emb, torch.Tensor):
                        self._embedding_dim = test_emb.size(-1)
                    else:
                        self._embedding_dim = test_emb.shape[-1]
                    print(f"[Clustering] Detected BGE-M3 embedding dimension: {self._embedding_dim}")
            except Exception as e:
                print(f"[Clustering] Failed to detect embedding dimension, using default value 1024: {e}")
                self._embedding_dim = 1024  # BGE-M3 default dimension
        
        for i, text in enumerate(profile_texts):
            try:
                # Use retriever's tokenizer and encode method to get embedding
                tokens = self.retriever.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    emb = self.retriever.encode(tokens)
                    # If emb is tensor, convert to numpy array
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    profile_embeddings.append(emb.flatten())  # Flatten to 1D array
                    
            except Exception as e:
                print(f"[Clustering] Error getting profile {i} embedding: {e}")
                # If getting embedding fails, use zero vector
                profile_embeddings.append(np.zeros(self._embedding_dim))
        
        # Convert to numpy array
        embeddings_array = np.array(profile_embeddings)
        print(f"[Clustering] Successfully obtained embeddings for {len(profile_texts)} profiles, dimension: {embeddings_array.shape}")
        return embeddings_array
    
    def _build_user_history_graph(self, corpus, profile, query_embedding=None):
        """
        Build graph structure for user historical records
        
        Args:
            corpus: List of user historical record texts
            profile: List of user historical record information
            query_embedding: Query embedding vector (for semantic similarity calculation)
            
        Returns:
            networkx.Graph: Built graph structure
            dict: Node to index mapping
            dict: Index to node mapping
            list: Cluster labels list (if clustering was performed)
            sklearn.cluster.KMeans: Clustering model (if clustering was performed)
        """
        if len(corpus) < 2:
            # If historical records are too few, cannot build graph
            return None, {}, {}, None, None
        
        # Get embeddings for all historical records
        history_embeddings = self._get_profile_embeddings(corpus)
        
        # Calculate semantic similarity matrix
        semantic_sim_matrix = cosine_similarity(history_embeddings)
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(corpus)):
            G.add_node(i, text=corpus[i], profile=profile[i])
        
        cluster_labels = None
        kmeans_model = None
        
        # Add semantic edges (based on clustering)
        if self.use_kmeans:
            try:
                # Use K-means for semantic clustering
                n_clusters = min(self.n_clusters, len(corpus))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(history_embeddings)
                kmeans_model = kmeans
                
                # Add semantic edges between nodes in the same cluster
                for i in range(len(corpus)):
                    for j in range(i + 1, len(corpus)):
                        if cluster_labels[i] == cluster_labels[j]:
                            # Calculate semantic similarity as edge weight
                            semantic_weight = semantic_sim_matrix[i][j]
                            if semantic_weight > 0.3:  # Set threshold to avoid connecting nodes with too different semantics
                                G.add_edge(i, j, weight=semantic_weight, edge_type='semantic')
            except Exception as e:
                print(f"[Graph Building] Semantic clustering failed: {e}")
        
        # Add time edges (based on time order)
        try:
            # Sort by time
            time_sorted_indices = sorted(
                range(len(profile)),
                key=lambda idx: self._parse_date(profile[idx]['date'])
            )
            
            # Add time edges between temporally adjacent nodes
            for i in range(len(time_sorted_indices) - 1):
                idx1 = time_sorted_indices[i]
                idx2 = time_sorted_indices[i + 1]
                
                # Calculate time interval (days)
                time_diff = self._calculate_time_diff(
                    profile[idx1]['date'], 
                    profile[idx2]['date']
                )
                
                # Time edge weight (smaller time interval, larger weight)
                time_weight = np.exp(-self.time_lambda1 * time_diff)
                G.add_edge(idx1, idx2, weight=time_weight, edge_type='temporal')
                
        except Exception as e:
            print(f"[Graph Building] Time edge construction failed: {e}")
        
        # Create index mapping
        node_to_idx = {i: i for i in range(len(corpus))}
        idx_to_node = {i: i for i in range(len(corpus))}
        
        print(f"[Graph Building] Successfully built graph structure, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        return G, node_to_idx, idx_to_node, cluster_labels, kmeans_model
    
    def _parse_date(self, date_value):
        """Parse date value and return datetime.date object"""
        try:
            if isinstance(date_value, int):
                # Year format
                return datetime.date(date_value, 1, 1)
            elif isinstance(date_value, str):
                # YYYY-MM-DD format
                return datetime.datetime.strptime(date_value, "%Y-%m-%d").date()
            else:
                return datetime.date.today()
        except Exception as e:
            print(f"[Date Parsing] Date parsing failed: {date_value}, {e}")
            return datetime.date.today()
    
    def _calculate_time_diff(self, date1, date2):
        """Calculate the number of days between two dates"""
        try:
            d1 = self._parse_date(date1)
            d2 = self._parse_date(date2)
            return abs((d2 - d1).days)
        except Exception as e:
            print(f"[Time Diff] Time difference calculation failed: {date1}, {date2}, {e}")
            return 0
    
    def _get_random_walk_start_node(self, G, corpus, profile, query, query_embedding):
        """
        Determine the starting node for random walk
        
        Args:
            G: Graph structure
            corpus: List of historical record texts
            profile: List of historical record information
            query: Query text
            query_embedding: Query embedding vector
            
        Returns:
            int: Index of the starting node
        """
        if self.walk_start_method == "latest":
            # Select the most recent node by time
            try:
                latest_idx = max(
                    range(len(profile)),
                    key=lambda idx: self._parse_date(profile[idx]['date'])
                )
                print(f"[Random Walk] Selected most recent node as starting point: {latest_idx}")
                return latest_idx
            except Exception as e:
                print(f"[Random Walk] Failed to select most recent node: {e}")
                return 0
                
        elif self.walk_start_method == "semantic":
            # Select the node most semantically similar to the query
            try:
                # Calculate semantic similarity between query and all historical records
                query_tokens = self.retriever.tokenizer(
                    query,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    query_emb = self.retriever.encode(query_tokens)
                    if isinstance(query_emb, torch.Tensor):
                        query_emb = query_emb.cpu().numpy().flatten()
                    else:
                        query_emb = query_emb.flatten()
                
                # Get historical record embeddings
                history_embeddings = self._get_profile_embeddings(corpus)
                
                # Calculate similarity
                similarities = cosine_similarity([query_emb], history_embeddings)[0]
                most_similar_idx = np.argmax(similarities)
                
                print(f"[Random Walk] Selected most semantically similar node as starting point: {most_similar_idx}, similarity: {similarities[most_similar_idx]:.4f}")
                return most_similar_idx
                
            except Exception as e:
                print(f"[Random Walk] Failed to select most semantically similar node: {e}")
                return 0
        
        else:
            # Default to selecting the first node
            return 0
    
    def _personalized_random_walk(self, G, start_node, corpus, profile, query_embedding=None):
        """
        Execute personalized random walk
        
        Args:
            G: Graph structure
            start_node: Starting node
            corpus: List of historical record texts
            profile: List of historical record information
            query_embedding: Query embedding vector
            
        Returns:
            list: List of node indices in the walk path
        """
        if G is None or G.number_of_nodes() == 0:
            return [start_node]
        
        visited_nodes = set()
        walk_path = []
        current_node = start_node
        
        # Get the time of the latest document as reference
        try:
            latest_date = max(
                [self._parse_date(p['date']) for p in profile],
                default=datetime.date.today()
            )
        except Exception as e:
            print(f"[Random Walk] Failed to get latest time: {e}")
            latest_date = datetime.date.today()
        
        while len(walk_path) < self.walk_length and current_node is not None:
            # Add current node to path
            walk_path.append(current_node)
            visited_nodes.add(current_node)
            
            # Get neighbors of current node
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            
            # Calculate transition probabilities
            transition_probs = []
            for neighbor in neighbors:
                if neighbor in visited_nodes:
                    continue
                
                # Calculate semantic similarity
                if query_embedding is not None:
                    # Use query embedding to calculate semantic similarity
                    neighbor_text = corpus[neighbor]
                    neighbor_tokens = self.retriever.tokenizer(
                        neighbor_text,
                        padding=True,
                        truncation=True,
                        max_length=self.retriever.max_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        neighbor_emb = self.retriever.encode(neighbor_tokens)
                        if isinstance(neighbor_emb, torch.Tensor):
                            neighbor_emb = neighbor_emb.cpu().numpy().flatten()
                        else:
                            neighbor_emb = neighbor_emb.flatten()
                    
                    semantic_sim = cosine_similarity([query_embedding], [neighbor_emb])[0][0]
                else:
                    # Use graph edge weights as semantic similarity
                    edge_data = G.get_edge_data(current_node, neighbor)
                    semantic_sim = edge_data.get('weight', 0.5) if edge_data else 0.5
                
                # Calculate time interval
                current_date = self._parse_date(profile[current_node]['date'])
                neighbor_date = self._parse_date(profile[neighbor]['date'])
                time_diff = abs((neighbor_date - current_date).days)
                
                # Calculate time interval to latest document
                neighbor_to_latest = abs((latest_date - neighbor_date).days)
                
                # Calculate transition strength
                transition_strength = (
                    (semantic_sim ** self.semantic_alpha) *
                    np.exp(-self.time_lambda1 * time_diff) *
                    np.exp(-self.time_lambda2 * neighbor_to_latest)
                )
                
                transition_probs.append((neighbor, transition_strength))
            
            if not transition_probs:
                break
            
            # Select next node based on transition probabilities
            transition_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Use roulette wheel selection, but bias towards high probability nodes
            total_prob = sum(prob for _, prob in transition_probs)
            if total_prob > 0:
                # Normalize probabilities
                normalized_probs = [(node, prob / total_prob) for node, prob in transition_probs]
                
                # Select one from the top probability nodes
                top_k = min(3, len(normalized_probs))
                selected_idx = np.random.choice(top_k, p=[prob for _, prob in normalized_probs[:top_k]])
                current_node = normalized_probs[selected_idx][0]
            else:
                break
        
        print(f"[Random Walk] Random walk completed, visited {len(walk_path)} nodes")
        return walk_path
    
    def _get_profile_tfidf(self, profile_texts):
        """
        Get TF-IDF vectors for profiles
        
        Args:
            profile_texts: List of profile texts
            
        Returns:
            scipy sparse matrix: TF-IDF matrix for profiles
        """
        # Use TF-IDF for vectorization
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit number of features for efficiency
            min_df=1,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency, filter out overly common words
            stop_words='english',  # Use English stop words
            ngram_range=(1, 2),  # Use 1-gram and 2-gram
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Train TF-IDF vectorizer and transform text
        return tfidf_vectorizer.fit_transform(profile_texts)

    def run(self):
        """Run the memorizer to retrieve relevant profiles"""
        # Generate summaries if needed
        user_summaries = {}
        if self.summary:
            print(f"[Batch Summary] Starting to generate graph-based clustering summaries for all users")
            
            # Generate summary for each user using clustering from graph building process
            for data in tqdm(self.dataset, desc="Generating summaries"):
                user_id = data['user_id']
                
                # If this user has already been processed, skip
                if user_id in user_summaries:
                    continue
                
                # Get user profile
                user_idx = self.user2id[user_id]
                profile = self.user_vocab[user_idx]['profile']
                corpus = self.get_corpus(profile, self.use_date)
                
                # Build graph structure and get clustering information
                if len(corpus) > 1 and self.use_kmeans:
                    try:
                        # Get query embedding (using a simple query)
                        query_tokens = self.retriever.tokenizer(
                            "user preference analysis",
                            padding=True,
                            truncation=True,
                            max_length=self.retriever.max_length,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        with torch.no_grad():
                            query_emb = self.retriever.encode(query_tokens)
                            if isinstance(query_emb, torch.Tensor):
                                query_emb = query_emb.cpu().numpy().flatten()
                            else:
                                query_emb = query_emb.flatten()
                        
                        # Build graph structure and get clustering information
                        G, node_to_idx, idx_to_node, cluster_labels, kmeans_model = self._build_user_history_graph(corpus, profile, query_emb)
                        
                        # Use clustering information to generate summary
                        summary = self.generate_clustering_summary(profile, user_id, cluster_labels, kmeans_model)
                        user_summaries[user_id] = summary
                        
                    except Exception as e:
                        print(f"[Summary] Failed to generate clustering summary for user {user_id}: {e}")
                        # Fall back to traditional method
                        summary = self.generate_traditional_summary(profile, user_id)
                        user_summaries[user_id] = summary
                else:
                    # Use traditional method
                    summary = self.generate_traditional_summary(profile, user_id)
                    user_summaries[user_id] = summary
            
            print(f"[Batch Summary] Completed profile summary generation for all users")
        
        # Release GPU memory occupied by LLM
        if self.summary and self.auto_cleanup_llm:
            self.cleanup_llm()
        
        # Determine output directory and file name
        retriever_name = "bge-m3"
        sub_dir = f"{retriever_name}_{self.topk}"
        file_name = f"mem_graph"
        
        # Add summary parameters to filename
        if self.summary:
            file_name += f'_summaryk_{self.summary_k}_parallel-{self.use_parallel_summary}_batchsize-{self.summary_batch_size}'
            if self.enable_token_check:
                file_name += f'_tokencheck-{self.enable_token_check}_reserve-{self.reserve_tokens}'
        
        # Add k-means clustering and time decay parameters to filename
        file_name += f'_kmeans-{self.use_kmeans}_method-{self.kmeans_select_method}_clusters-{self.n_clusters}_recency-{self.use_recency}_decay-{self.time_decay_lambda}'
        
        # Add graph structure random walk parameters to filename
        if self.use_graph_walk:
            file_name += f'_graphwalk-{self.use_graph_walk}_start-{self.walk_start_method}_length-{self.walk_length}_alpha-{self.semantic_alpha}_lambda1-{self.time_lambda1}_lambda2-{self.time_lambda2}'

        results = []
        for data in tqdm(self.dataset):
            query, selected_profs = self.retrieve_topk(data['input'],
                                                       data['user_id'],
                                                       user_summaries if self.summary else None)
            result_item = {
                "input": data['input'],
                "query": query,
                "output": data['output'],
                "user_id": data['user_id'],
                "retrieval": selected_profs
            }
            
            # If summary was used, add current user's summary
            if self.summary and user_summaries and data['user_id'] in user_summaries:
                result_item["summary"] = user_summaries[data['user_id']]
            elif self.summary and user_summaries and data['user_id'] not in user_summaries:
                print(f"[Warning] No corresponding summary found for user {data['user_id']}")
            
            results.append(result_item)

        output_addr = os.path.join(self.output_addr, self.data_split,
                                   self.source, sub_dir, 'retrieval')

        if not os.path.exists(output_addr):
            os.makedirs(output_addr)

        result_path = os.path.join(output_addr, f"{file_name}.json")
        print("save file to: {}".format(result_path))
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    def retrieve_topk(self, inp, user, user_summaries=None):
        """Retrieve top-k profiles for given input and user"""
        # Get current user's profile
        user_id = self.user2id[user]
        current_profile = self.user_vocab[user_id]['profile']
        
        query = self.get_query(inp)
        
        # Check if query is None and handle it
        if query is None:
            print(f"Warning: get_query returned None for input: {inp}")
            query = str(inp)  # Use input as fallback query
        
        cur_corpus = self.get_corpus(current_profile, self.use_date)
        cur_retrieved, cur_scores = self.retrieve_topk_one_user(
            cur_corpus, current_profile, query, user, self.topk)
        
        all_retrieved = []
        for data_idx, data in enumerate(cur_retrieved):
            cur_data = copy.deepcopy(data)
            if self.task.startswith('LaMP_3'):
                cur_data['rate'] = cur_data['score']
            cur_data['score'] = cur_scores[data_idx]
            all_retrieved.append(cur_data)
            
        return query, all_retrieved

    def retrieve_topk_one_user(self, corpus, profile, query, user, topk):
        """Retrieve top-k items for one user using BGE dense retrieval with k-means clustering and time decay"""
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)
            print(f"Warning: Query converted to string: {query}")
        
        # Ensure corpus is a list of strings
        if not isinstance(corpus, list):
            print(f"Warning: Corpus is not a list: {type(corpus)}")
            corpus = [str(corpus)] if corpus is not None else [""]
        
        # Ensure profile is a list
        if not isinstance(profile, list):
            print(f"Warning: Profile is not a list: {type(profile)}")
            profile = [profile] if profile is not None else [{}]
        
        # 0. Graph structure random walk (if enabled)
        if self.use_graph_walk and len(corpus) > 1:
            try:
                print(f"[Graph Walk] Starting to build graph structure and perform random walk")
                
                # Get query embedding
                query_tokens = self.retriever.tokenizer(
                    query,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    query_emb = self.retriever.encode(query_tokens)
                    if isinstance(query_emb, torch.Tensor):
                        query_emb = query_emb.cpu().numpy().flatten()
                    else:
                        query_emb = query_emb.flatten()
                
                # Build graph structure
                G, node_to_idx, idx_to_node, cluster_labels, kmeans_model = self._build_user_history_graph(corpus, profile, query_emb)
                
                if G is not None:
                    # Determine starting node
                    start_node = self._get_random_walk_start_node(G, corpus, profile, query, query_emb)
                    
                    # Execute personalized random walk
                    walk_path = self._personalized_random_walk(G, start_node, corpus, profile, query_emb)
                    
                    # Select historical records corresponding to nodes in the walk path
                    graph_corpus = [corpus[idx] for idx in walk_path]
                    graph_profile = [profile[idx] for idx in walk_path]
                    
                    print(f"[Graph Walk] Random walk selected {len(walk_path)} nodes")
                    
                    # Use BGE retriever for final ranking of walk results
                    selected_profs, dense_scores = self.retriever.retrieve_topk_dense(
                        graph_corpus, graph_profile, query, user, min(len(walk_path), topk)
                    )
                    
                    # If walk results are insufficient, supplement with original method results
                    if len(selected_profs) < topk:
                        print(f"[Graph Walk] Walk results insufficient {topk} items, supplementing with original method results")
                        remaining_topk = topk - len(selected_profs)
                        
                        # Use original method to get remaining results
                        remaining_corpus = [c for c in corpus if c not in graph_corpus]
                        remaining_profile = [p for p in profile if p not in graph_profile]
                        
                        if remaining_corpus:
                            remaining_profs, remaining_scores = self.retriever.retrieve_topk_dense(
                                remaining_corpus, remaining_profile, query, user, remaining_topk
                            )
                            
                            # Merge results
                            selected_profs.extend(remaining_profs)
                            dense_scores.extend(remaining_scores)
                    
                    return selected_profs[:topk], dense_scores[:topk]
                    
            except Exception as e:
                print(f"[Graph Walk] Graph structure random walk failed: {e}")
                print(f"[Graph Walk] Falling back to original method")
        
        n_clusters = self.n_clusters

        # 1. kmeans clustering
        if self.use_kmeans:
            try:
                # Convert corpus to text for TF-IDF vectorization
                corpus_texts = [str(item) for item in corpus]
                
                # Select vectorization method based on configuration
                if self.kmeans_use_embedding:
                    print(f"[Clustering] Using embedding for K-means clustering, profile count: {len(corpus_texts)}")
                    X = self._get_profile_embeddings(corpus_texts)
                else:
                    print(f"[Clustering] Using TF-IDF for K-means clustering, profile count: {len(corpus_texts)}")
                    X = self._get_profile_tfidf(corpus_texts)
                
                # Use K-means clustering
                kmeans = KMeans(n_clusters=min(n_clusters, len(corpus)), random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                
                # Select representative samples from each cluster
                representative_indices = []
                for i in range(kmeans.n_clusters):
                    cluster_idx = np.where(cluster_labels == i)[0]
                    if len(cluster_idx) == 0:
                        continue
                    
                    if self.kmeans_select_method == "center":
                        # Select sample closest to cluster center
                        if self.kmeans_use_embedding:
                            cluster_vectors = X[cluster_idx]
                        else:
                            cluster_vectors = X[cluster_idx].toarray()
                        center_vector = kmeans.cluster_centers_[i]
                        distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                        selected_idx = cluster_idx[np.argmin(distances)]
                    elif self.kmeans_select_method == "relevance":
                        # Select sample most relevant to query as cluster representative
                        cluster_corpus = [corpus[idx] for idx in cluster_idx]
                        cluster_profile = [profile[idx] for idx in cluster_idx]
                        
                        # Use BGE retriever to calculate relevance scores
                        try:
                            cluster_profs, cluster_scores = self.retriever.retrieve_topk_dense(
                                cluster_corpus, cluster_profile, query, user, len(cluster_corpus)
                            )
                            # Select sample with highest score
                            best_idx = np.argmax(cluster_scores)
                            selected_idx = cluster_idx[best_idx]
                        except Exception as e:
                            print(f"[K-means Relevance Selection] Cluster {i} relevance selection failed: {e}")
                            # Fall back to center selection
                            if self.kmeans_use_embedding:
                                cluster_vectors = X[cluster_idx]
                            else:
                                cluster_vectors = X[cluster_idx].toarray()
                            center_vector = kmeans.cluster_centers_[i]
                            distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                            selected_idx = cluster_idx[np.argmin(distances)]
                    else:
                        # Default to center selection
                        if self.kmeans_use_embedding:
                            cluster_vectors = X[cluster_idx]
                        else:
                            cluster_vectors = X[cluster_idx].toarray()
                        center_vector = kmeans.cluster_centers_[i]
                        distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                        selected_idx = cluster_idx[np.argmin(distances)]
                    
                    representative_indices.append(selected_idx)
                
                kmeans_corpus = [corpus[i] for i in representative_indices]
                kmeans_profile = [profile[i] for i in representative_indices]
                
            except Exception as e:
                print(f"[K-means Clustering] Error occurred during clustering: {e}")
                # If clustering fails, use original corpus
                kmeans_corpus = corpus
                kmeans_profile = profile
        else:
            kmeans_corpus = []
            kmeans_profile = []

        # 2. recency sorting
        if self.use_recency:
            recency_sorted = sorted(
                zip(corpus, profile),
                key=lambda x: tuple(map(int, str(x[1]['date']).split("-"))),
                reverse=True
            )
            recency_corpus = [x[0] for x in recency_sorted[:topk]]
            recency_profile = [x[1] for x in recency_sorted[:topk]]
        else:
            recency_corpus = []
            recency_profile = []

        # 3. Merge kmeans and recency results
        if self.use_kmeans or self.use_recency:
            merged_corpus = kmeans_corpus + [c for c in recency_corpus if c not in kmeans_corpus]
            merged_profile = kmeans_profile + [p for p in recency_profile if p not in kmeans_profile]
        else:
            merged_corpus = corpus
            merged_profile = profile

        # 4. Use BGE retrieval
        selected_profs, dense_scores = self.retriever.retrieve_topk_dense(
            merged_corpus, merged_profile, query, user, len(merged_corpus)
        )
        
        # 5. Apply time decay weights
        if self.time_decay_lambda > 0:
            time_weights = self._get_time_weight(selected_profs)
            dense_scores = np.array(dense_scores) * np.array(time_weights)
        
        # 6. Re-sort and return topk
        sorted_idx = np.argsort(dense_scores)[::-1][:topk]
        selected_profs = [selected_profs[i] for i in sorted_idx]
        top_n_scores = [dense_scores[i] for i in sorted_idx]

        return selected_profs, top_n_scores

    def generate_clustering_summary(self, profile_list, user_id, cluster_labels=None, kmeans_model=None):
        """
        Generate hierarchical summary using existing clustering results
        
        Args:
            profile_list: List of user's historical behaviors
            user_id: User ID
            cluster_labels: List of cluster labels (obtained from graph building process)
            kmeans_model: Clustering model (obtained from graph building process)
            
        Returns:
            global_summary: Final global summary
        """
        print(f"[Clustering Summary] Starting to generate clustering summary for user {user_id}, profile count: {len(profile_list)}")
        
        # If no clustering information, use traditional method
        if cluster_labels is None or kmeans_model is None:
            print(f"[Clustering Summary] User {user_id} has no clustering information, using traditional summary method")
            return self.generate_traditional_summary(profile_list, user_id)
        
        # 1. Use existing clustering results
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Filter out clusters that are too small
        valid_clusters = []
        for cluster_indices in clusters.values():
            if len(cluster_indices) >= 3:  # Minimum cluster size
                valid_clusters.append(cluster_indices)
        
        # Assign profiles not allocated to valid clusters to the nearest cluster
        all_assigned = set()
        for cluster in valid_clusters:
            all_assigned.update(cluster)
        
        unassigned = [i for i in range(len(profile_list)) if i not in all_assigned]
        if unassigned and valid_clusters:
            # Add unassigned profiles to the nearest cluster
            for idx in unassigned:
                valid_clusters[0].append(idx)
        
        if not valid_clusters:
            print(f"[Clustering Summary] User {user_id} has no valid clusters, using traditional summary method")
            return self.generate_traditional_summary(profile_list, user_id)
        
        print(f"[Clustering Summary] User {user_id} clustering completed, total {len(valid_clusters)} clusters")
        
        # 2. Generate local summary for each cluster
        local_summaries = []
        for cluster_id, cluster_indices in enumerate(valid_clusters):
            cluster_profiles = [profile_list[i] for i in cluster_indices]
            print(f"[Clustering Summary] Generating local summary for user {user_id} cluster {cluster_id + 1}, contains {len(cluster_profiles)} profiles")
            
            local_summary = self.generate_local_summary(cluster_profiles, cluster_id)
            local_summaries.append(local_summary)
            print(f"[Clustering Summary] Cluster {cluster_id + 1} local summary completed")
        
        # 3. Integrate all local summaries into global summary
        print(f"[Clustering Summary] Integrating {len(local_summaries)} local summaries into global summary for user {user_id}")
        global_summary = self.generate_global_summary(local_summaries, user_id)
        
        print(f"[Clustering Summary] User {user_id} clustering summary generation completed")
        return global_summary

    def generate_traditional_summary(self, profile_list, user_id):
        """
        Generate summary using traditional method (when there's no clustering information)
        
        Args:
            profile_list: List of user's historical behaviors
            user_id: 用户ID
            
        Returns:
            summary: 生成的summary
        """
        print(f"[Traditional Summary] 为用户 {user_id} 使用传统方法生成summary")
        
        # 将profile转换为文本
        profile_texts = [str(profile) for profile in profile_list]
        
        # 按summary_k分批处理
        summary = ""
        for i in range(0, len(profile_texts), self.summary_k):
            batch_texts = profile_texts[i:i+self.summary_k]
            batch_summary = llm_summarize(summary, batch_texts, self.agent_tokenizer, self.agent_llm, self.agent_sampling_params)
            summary = batch_summary
        
        return summary

    def generate_local_summary(self, profile_cluster, cluster_id):
        """
        为单个聚类生成local summary
        
        Args:
            profile_cluster: 聚类中的profile列表
            cluster_id: 聚类ID
            
        Returns:
            local_summary: 该聚类的local summary
        """
        if not profile_cluster:
            return ""
        
        # 将profile转换为文本
        profile_texts = [str(profile) for profile in profile_cluster]
        
        # 构建prompt
        prompt = f"You are an expert at analyzing user behavior patterns. Please analyze the following cluster of user activities and provide a concise summary of the user's preferences and patterns in this cluster.\n\n"
        prompt += f"**Cluster {cluster_id + 1} Activities ({len(profile_texts)} records):**\n"
        for i, text in enumerate(profile_texts):
            prompt += f"{i+1}. {text}\n"
        prompt += "\n**Task:**\nProvide a concise summary of the user's preferences and behavior patterns in this cluster. Focus on key themes, preferences, and patterns. Use clear, structured language."
        
        # 检查并截断prompt
        max_model_len = 8192
        prompt = self.check_and_truncate_prompt(prompt, max_model_len, self.reserve_tokens, self.enable_token_check)
        
        message = [{"role": "user", "content": prompt}]
        chat_prompt = self.agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        outputs = self.agent_llm.generate([chat_prompt], self.agent_sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_global_summary(self, local_summaries, user_id):
        """
        将所有local summary整合成global summary
        
        Args:
            local_summaries: local summary列表
            user_id: 用户ID
            
        Returns:
            global_summary: 整合后的global summary
        """
        if not local_summaries:
            return ""
        
        if len(local_summaries) == 1:
            # 如果只有一个local summary，直接返回
            return local_summaries[0]
        
        # 构建prompt - 强调简洁性
        prompt = f"You are an expert at creating concise user preference summaries. Please synthesize the following cluster summaries into a brief, focused global summary.\n\n"
        prompt += f"**User ID:** {user_id}\n"
        prompt += f"**Cluster Summaries ({len(local_summaries)} clusters):**\n"
        for i, summary in enumerate(local_summaries):
            prompt += f"Cluster {i+1}: {summary}\n"
        prompt += "\n**Task:**\nCreate a concise global summary (max 300 words) that captures the user's key preferences and behavior patterns. Focus on the most important themes and avoid redundancy. Use bullet points or short paragraphs for clarity. Be brief but comprehensive."
        
        # 检查并截断prompt
        max_model_len = 8192
        prompt = self.check_and_truncate_prompt(prompt, max_model_len, self.reserve_tokens, self.enable_token_check)
        
        message = [{"role": "user", "content": prompt}]
        chat_prompt = self.agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        outputs = self.agent_llm.generate([chat_prompt], self.agent_sampling_params)
        return outputs[0].outputs[0].text.strip()

    def check_and_truncate_prompt(self, prompt: str, max_model_len: int = 8192, reserve_tokens: int = 500, enable_check: bool = True) -> str:
        """
        检查prompt的token长度，如果超过限制则自动截断
        """
        if not enable_check:
            return prompt
        
        # 计算当前prompt的token数量
        tokens = self.agent_tokenizer(prompt, return_tensors="pt")
        current_tokens = tokens["input_ids"].shape[1]
        
        # 计算最大允许的输入token数量
        max_input_tokens = max_model_len - reserve_tokens
        
        if current_tokens <= max_input_tokens:
            return prompt
        
        # 如果超过限制，截断到最大允许长度
        truncated_tokens = self.agent_tokenizer.decode(
            tokens["input_ids"][0][:max_input_tokens], 
            skip_special_tokens=True
        )
        
        # 尝试在句子边界截断
        sentences = truncated_tokens.split('.')
        if len(sentences) > 1:
            truncated_tokens = '.'.join(sentences[:-1]) + '.'
        
        return truncated_tokens

    def _parse_date(self, date_value):
        """Parse date value and return datetime.date object"""
        try:
            if isinstance(date_value, int):
                # Year format
                return datetime.date(date_value, 1, 1)
            elif isinstance(date_value, str):
                # YYYY-MM-DD format
                return datetime.datetime.strptime(date_value, "%Y-%m-%d").date()
            else:
                return datetime.date.today()
        except Exception as e:
            print(f"[Date Parsing] Date parsing failed: {date_value}, {e}")
            return datetime.date.today()
    
    def _calculate_time_diff(self, date1, date2):
        """Calculate the number of days between two dates"""
        try:
            d1 = self._parse_date(date1)
            d2 = self._parse_date(date2)
            return abs((d2 - d1).days)
        except Exception as e:
            print(f"[Time Diff] Time difference calculation failed: {date1}, {date2}, {e}")
            return 0
    
    def _get_random_walk_start_node(self, G, corpus, profile, query, query_embedding):
        """
        Determine the starting node for random walk
        
        Args:
            G: Graph structure
            corpus: List of historical record texts
            profile: List of historical record information
            query: Query text
            query_embedding: Query embedding vector
            
        Returns:
            int: Index of the starting node
        """
        if self.walk_start_method == "latest":
            # Select the most recent node by time
            try:
                latest_idx = max(
                    range(len(profile)),
                    key=lambda idx: self._parse_date(profile[idx]['date'])
                )
                print(f"[Random Walk] Selected most recent node as starting point: {latest_idx}")
                return latest_idx
            except Exception as e:
                print(f"[Random Walk] Failed to select most recent node: {e}")
                return 0
                
        elif self.walk_start_method == "semantic":
            # Select the node most semantically similar to the query
            try:
                # Calculate semantic similarity between query and all historical records
                query_tokens = self.retriever.tokenizer(
                    query,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    query_emb = self.retriever.encode(query_tokens)
                    if isinstance(query_emb, torch.Tensor):
                        query_emb = query_emb.cpu().numpy().flatten()
                    else:
                        query_emb = query_emb.flatten()
                
                # Get historical record embeddings
                history_embeddings = self._get_profile_embeddings(corpus)
                
                # Calculate similarity
                similarities = cosine_similarity([query_emb], history_embeddings)[0]
                most_similar_idx = np.argmax(similarities)
                
                print(f"[Random Walk] Selected most semantically similar node as starting point: {most_similar_idx}, similarity: {similarities[most_similar_idx]:.4f}")
                return most_similar_idx
                
            except Exception as e:
                print(f"[Random Walk] Failed to select most semantically similar node: {e}")
                return 0
        
        else:
            # Default to selecting the first node
            return 0
    
    def _personalized_random_walk(self, G, start_node, corpus, profile, query_embedding=None):
        """
        Execute personalized random walk
        
        Args:
            G: Graph structure
            start_node: Starting node
            corpus: List of historical record texts
            profile: List of historical record information
            query_embedding: Query embedding vector
            
        Returns:
            list: List of node indices in the walk path
        """
        if G is None or G.number_of_nodes() == 0:
            return [start_node]
        
        visited_nodes = set()
        walk_path = []
        current_node = start_node
        
        # Get the time of the latest document as reference
        try:
            latest_date = max(
                [self._parse_date(p['date']) for p in profile],
                default=datetime.date.today()
            )
        except Exception as e:
            print(f"[Random Walk] Failed to get latest time: {e}")
            latest_date = datetime.date.today()
        
        while len(walk_path) < self.walk_length and current_node is not None:
            # Add current node to path
            walk_path.append(current_node)
            visited_nodes.add(current_node)
            
            # Get neighbors of current node
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            
            # Calculate transition probabilities
            transition_probs = []
            for neighbor in neighbors:
                if neighbor in visited_nodes:
                    continue
                
                # Calculate semantic similarity
                if query_embedding is not None:
                    # Use query embedding to calculate semantic similarity
                    neighbor_text = corpus[neighbor]
                    neighbor_tokens = self.retriever.tokenizer(
                        neighbor_text,
                        padding=True,
                        truncation=True,
                        max_length=self.retriever.max_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        neighbor_emb = self.retriever.encode(neighbor_tokens)
                        if isinstance(neighbor_emb, torch.Tensor):
                            neighbor_emb = neighbor_emb.cpu().numpy().flatten()
                        else:
                            neighbor_emb = neighbor_emb.flatten()
                    
                    semantic_sim = cosine_similarity([query_embedding], [neighbor_emb])[0][0]
                else:
                    # Use graph edge weights as semantic similarity
                    edge_data = G.get_edge_data(current_node, neighbor)
                    semantic_sim = edge_data.get('weight', 0.5) if edge_data else 0.5
                
                # Calculate time interval
                current_date = self._parse_date(profile[current_node]['date'])
                neighbor_date = self._parse_date(profile[neighbor]['date'])
                time_diff = abs((neighbor_date - current_date).days)
                
                # Calculate time interval to latest document
                neighbor_to_latest = abs((latest_date - neighbor_date).days)
                
                # Calculate transition strength
                transition_strength = (
                    (semantic_sim ** self.semantic_alpha) *
                    np.exp(-self.time_lambda1 * time_diff) *
                    np.exp(-self.time_lambda2 * neighbor_to_latest)
                )
                
                transition_probs.append((neighbor, transition_strength))
            
            if not transition_probs:
                break
            
            # Select next node based on transition probabilities
            transition_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Use roulette wheel selection, but bias towards high probability nodes
            total_prob = sum(prob for _, prob in transition_probs)
            if total_prob > 0:
                # Normalize probabilities
                normalized_probs = [(node, prob / total_prob) for node, prob in transition_probs]
                
                # Select one from the top probability nodes
                top_k = min(3, len(normalized_probs))
                selected_idx = np.random.choice(top_k, p=[prob for _, prob in normalized_probs[:top_k]])
                current_node = normalized_probs[selected_idx][0]
            else:
                break
        
        print(f"[Random Walk] Random walk completed, visited {len(walk_path)} nodes")
        return walk_path
    
    def _get_profile_tfidf(self, profile_texts):
        """
        Get TF-IDF vectors for profiles
        
        Args:
            profile_texts: List of profile texts
            
        Returns:
            scipy sparse matrix: TF-IDF matrix for profiles
        """
        # Use TF-IDF for vectorization
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit number of features for efficiency
            min_df=1,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency, filter out overly common words
            stop_words='english',  # Use English stop words
            ngram_range=(1, 2),  # Use 1-gram and 2-gram
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Train TF-IDF vectorizer and transform text
        return tfidf_vectorizer.fit_transform(profile_texts)

 