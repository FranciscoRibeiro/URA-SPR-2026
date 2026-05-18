import ast
import io
import tokenize
from rapidfuzz.distance import Levenshtein as Lev

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# AST Metrics
# -----------------------------
def get_node_signature(node):
    
    node_type = type(node).__name__
    attrs = []

    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.AugAssign)):
        if hasattr(node, "op"):
            attrs.append(("op", type(node.op).__name__))

    if isinstance(node, ast.Compare):
        ops = tuple(type(o).__name__ for o in node.ops)
        attrs.append(("ops", ops))

    if isinstance(node, ast.Name):
        attrs.append(("id", node.id))
    if isinstance(node, ast.Attribute):
        attrs.append(("attr", node.attr))

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        attrs.append(("name", node.name))

    if isinstance(node, ast.Constant):
        attrs.append(("type", type(node.value).__name__))
        val_repr = repr(node.value)
        if len(val_repr) <= 64:
            attrs.append(("value", val_repr))

    if isinstance(node, ast.alias):
        attrs.append(("name", node.name))

    return (node_type, frozenset(attrs))


def get_ast_signatures(code):
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        wrapped = "def _wrapper_func_():\n" + "".join("    " + line + "\n" for line in code.splitlines())
        try:
            tree = ast.parse(wrapped)
        except SyntaxError:
            return {}

    counts = {}
    for node in ast.walk(tree):
        sig = get_node_signature(node)
        counts[sig] = counts.get(sig, 0) + 1
        
    return counts


def compute_ast_diff(code_a, code_b):
    
    counter_a = get_ast_signatures(code_a)
    counter_b = get_ast_signatures(code_b)

    if not counter_a and not counter_b:
        return 0, 0.0

    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())
    total = total_a + total_b

    all_sigs = set(counter_a.keys()) | set(counter_b.keys())
    changed = sum(abs(counter_a.get(sig, 0) - counter_b.get(sig, 0)) for sig in all_sigs)

    normalized = changed / total if total > 0 else 0.0

    return changed, normalized


# -----------------------------
# Levenshtein Metrics
# -----------------------------
def compute_char_levenshtein(a, b):
    
    dist = Lev.distance(a, b)
    norm = dist / len(a) if len(a) > 0 else 0.0
    
    return dist, norm


def tokenize_code(code):
    
    skip_types = {
        tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, 
        tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, 
        tokenize.ENDMARKER, tokenize.ERRORTOKEN
    }
    
    tokens = []
    try:
        reader = io.StringIO(code).readline
        for tok_type, tok_string, _, _, _ in tokenize.generate_tokens(reader):
            if tok_type not in skip_types and tok_string.strip():
                tokens.append(tok_string)
    except Exception:
        pass
        
    return tokens


def compute_token_levenshtein(a, b):
    
    toks_a = tokenize_code(a)
    toks_b = tokenize_code(b)

    raw = Lev.distance(toks_a, toks_b)
    denom = max(len(toks_a), len(toks_b))
    norm = raw / denom if denom > 0 else 0.0
    
    return raw, norm


# -----------------------------
# Embeddings & Models
# -----------------------------
def get_device():
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
        
    return torch.device("cpu")


def load_embedding_model(model_name):
    
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    model.eval()
    model.to(device)
    
    return tokenizer, model, device


def get_embedding_codebert(code, tokenizer, model, device):
    
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    mask_expanded = inputs["attention_mask"].unsqueeze(-1).float()
    sum_hidden = (outputs.last_hidden_state * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    
    embedding = (sum_hidden / sum_mask).detach().cpu().numpy()
    
    return embedding


def get_embedding_unixcoder(code, tokenizer, model, device):
    
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    
    return embedding


def compute_cosine_similarity(a, b):
    
    a_t = torch.from_numpy(a.flatten())
    b_t = torch.from_numpy(b.flatten())
    
    return float(F.cosine_similarity(a_t.unsqueeze(0), b_t.unsqueeze(0)).item())
