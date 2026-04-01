import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    rel_score = relevance_scores[:min(len(relevance_scores), k)]
    sorted_rel_score = sorted(relevance_scores, reverse=True)[:min(len(relevance_scores), k)]
    
    dcg = sum(
        ((2 ** r) - 1) / math.log2(i + 2) 
        for i, r in enumerate(rel_score)
    )
    idcg = sum(
        ((2 ** r) - 1) / math.log2(i + 2) 
        for i, r in enumerate(sorted_rel_score)
    )
    
    if not idcg:
        return 0.0
        
    return dcg / idcg