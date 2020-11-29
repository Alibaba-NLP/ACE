import numpy as np
from collections import defaultdict


#***************************************************************
class Tarjan:
  """
    Computes Tarjan's algorithm for finding strongly connected components (cycles) of a graph
    
    Attributes:
      edges: dictionary of edges such that edges[dep] = head
      vertices: set of dependents
      SCCs: list of sets of strongly connected components. Non-singleton sets are cycles.
  """
  
  #=============================================================
  def __init__(self, prediction, tokens):
    """
      Inputs:
        prediction: a predicted dependency tree where
          prediction[dep_idx] = head_idx
        tokens: the tokens we care about (i.e. exclude _GO, _EOS, and _PAD)
    """
    
    self._edges = defaultdict(set)
    self._vertices = set((0,))
    for dep, head in enumerate(prediction[tokens]):
      self._vertices.add(dep+1)
      self._edges[head].add(dep+1)
    self._indices = {}
    self._lowlinks = {}
    self._onstack = defaultdict(lambda: False)
    self._SCCs = []
    
    index = 0
    stack = []
    for v in self.vertices:
      if v not in self.indices:
        self.strongconnect(v, index, stack)
  
  #=============================================================
  def strongconnect(self, v, index, stack):
    """"""
    
    self._indices[v] = index
    self._lowlinks[v] = index
    index += 1
    stack.append(v)
    self._onstack[v] = True
    for w in self.edges[v]:
      if w not in self.indices:
        self.strongconnect(w, index, stack)
        self._lowlinks[v] = min(self._lowlinks[v], self._lowlinks[w])
      elif self._onstack[w]:
        self._lowlinks[v] = min(self._lowlinks[v], self._indices[w])
    
    if self._lowlinks[v] == self._indices[v]:
      self._SCCs.append(set())
      while stack[-1] != v:
        w = stack.pop()
        self._onstack[w] = False
        self._SCCs[-1].add(w)
      w = stack.pop()
      self._onstack[w] = False
      self._SCCs[-1].add(w)
    return
  
  #======================
  @property
  def edges(self):
    return self._edges
  @property
  def vertices(self):
    return self._vertices
  @property
  def indices(self):
    return self._indices
  @property
  def SCCs(self):
    return self._SCCs
    

def MST_inference(parse_probs, length, tokens_to_keep, ensure_tree = True):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
    """
    if ensure_tree:
        I = np.eye(len(tokens_to_keep))
        # block loops and pad heads
        parse_probs = parse_probs * tokens_to_keep * (1-I)
        parse_preds = np.argmax(parse_probs, axis=1)
        tokens = np.arange(1, length)
        roots = np.where(parse_preds[tokens] == 0)[0]+1
        # ensure at least one root
        if len(roots) < 1:
            # The current root probabilities
            root_probs = parse_probs[tokens,0]
            # The current head probabilities
            old_head_probs = parse_probs[tokens, parse_preds[tokens]]
            # Get new potential root probabilities
            new_root_probs = root_probs / old_head_probs
            # Select the most probable root
            new_root = tokens[np.argmax(new_root_probs)]
            # Make the change
            parse_preds[new_root] = 0
            # ensure at most one root
        elif len(roots) > 1:
            # The probabilities of the current heads
            root_probs = parse_probs[roots,0]
            # Set the probability of depending on the root zero
            parse_probs[roots,0] = 0
            # Get new potential heads and their probabilities
            new_heads = np.argmax(parse_probs[roots][:,tokens], axis=1)+1
            new_head_probs = parse_probs[roots, new_heads] / root_probs
            # Select the most probable root
            new_root = roots[np.argmin(new_head_probs)]
            # Make the change
            parse_preds[roots] = new_heads
            parse_preds[new_root] = 0
        # remove cycles
        tarjan = Tarjan(parse_preds, tokens)
        cycles = tarjan.SCCs
        for SCC in tarjan.SCCs:
            if len(SCC) > 1:
                dependents = set()
                to_visit = set(SCC)
                while len(to_visit) > 0:
                    node = to_visit.pop()
                    if not node in dependents:
                        dependents.add(node)
                        to_visit.update(tarjan.edges[node])
                # The indices of the nodes that participate in the cycle
                cycle = np.array(list(SCC))
                # The probabilities of the current heads
                old_heads = parse_preds[cycle]
                old_head_probs = parse_probs[cycle, old_heads]
                # Set the probability of depending on a non-head to zero
                non_heads = np.array(list(dependents))
                parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(parse_probs[cycle][:,tokens], axis=1)+1
                new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
                # Select the most probable change
                change = np.argmax(new_head_probs)
                changed_cycle = cycle[change]
                old_head = old_heads[change]
                new_head = new_heads[change]
                # Make the change
                parse_preds[changed_cycle] = new_head
                tarjan.edges[new_head].add(changed_cycle)
                tarjan.edges[old_head].remove(changed_cycle)
        return parse_preds
    else:
        # block and pad heads
        parse_probs = parse_probs * tokens_to_keep
        parse_preds = np.argmax(parse_probs, axis=1)
        return parse_preds
