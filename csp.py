"""CSP (Constraint Satisfaction Problems) problems and solver."""
from filecmp import cmp

from search import Problem
from utils import DefaultDict, product, argmin_list, update, count_if, argmin_random_tie, every
import itertools

class CSP(Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following three inputs:
        vars        A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b
    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases. (For example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(N^4) for the
    explicit representation.) In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP.  Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
        """

    def __init__(self, vars, domains, neighbors, constraints):
        "Construct a CSP problem. If vars is empty, it becomes domains.keys()."
        vars = vars or domains.keys()
        update(self, vars=vars, domains=domains,
               neighbors=neighbors, constraints=constraints,
               initial={}, curr_domains=None, pruned=None, nassigns=0)

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any.
        Do bookkeeping for curr_domains and nassigns."""
        self.nassigns += 1
        assignment[var] = val
        if self.curr_domains:
            if self.fc:
                self.forward_check(var, val, assignment)

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment; that is backtrack.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            # Reset the curr_domain to be the full original domain
            if self.curr_domains:
                self.curr_domains[var] = self.domains[var]
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        "Return the number of conflicts var=val has with other variables."

        # Subclasses may implement this more efficiently
        def conflict(var2):
            val2 = assignment.get(var2, None)
            return val2 is not None and not self.constraints(var, val, var2, val2)

        return count_if(conflict, self.neighbors[var])

    def nconflictsss(self, var, val, assignment):   ####
        """The number of conflicts, as recorded with each assignment.
        Count conflicts in row and in up, down diagonals. If there
        is a queen there, it can't conflict with itself, so subtract 3."""
        n = len(self.vars)
        c = self.rows[val] + self.downs[var + val] + self.ups[var - val + n - 1]
        if assignment.get(var, None) == val:
            c -= 3
        return c

    def forward_check(self, var, val, assignment):
        "Do forward checking (current domain reduction) for this assignment."
        if self.curr_domains:
            # Restore prunings from previous value of var
            for (B, b) in self.pruned[var]:
                self.curr_domains[B].append(b)
            self.pruned[var] = []
            # Prune any other B=b assignment that conflict with var=val
            for B in self.neighbors[var]:
                if B not in assignment:
                    for b in self.curr_domains[B][:]:
                        if not self.constraints(var, val, B, b):
                            self.curr_domains[B].remove(b)
                            self.pruned[var].append((B, b))


def backtracking_search(csp, mcv=False, lcv=False, fc=False, mac=False, mrv=False):
    """Set up to do recursive backtracking search. Allow the following options:
    mrv - If true, use Minimum Remaining Value Heuristic
    fc  - If true, use Forward Checking
    """
    if fc or mac:
        csp.curr_domains, csp.pruned = {}, {}
        for v in csp.vars:
            csp.curr_domains[v] = csp.domains[v][:]
            csp.pruned[v] = []
    update(csp, mcv=mcv, lcv=lcv, fc=fc, mac=mac, mrv=mrv)
    return recursive_backtracking({}, csp)


def recursive_backtracking(assignment, csp):
    """Search for a consistent assignment for the csp.
    Each recursive call chooses a variable, and considers values for it."""
    if len(assignment) == len(csp.vars):
        return assignment
    # print('domain', ' , '.join(str(v) + ' ' + str(len(d)) for v, d in csp.curr_domains.items()))
    var = select_unassigned_variable(assignment, csp)
    # print('selected var', var)
    for val in csp.curr_domains[var]:
        # print('selected value', val)
        if csp.fc or csp.nconflicts(var, val, assignment) == 0:
            csp.assign(var, val, assignment)
            # print(assignment)
            result = recursive_backtracking(assignment, csp)
            if result is not None:
                return result
        csp.unassign(var, assignment)
    # print('back track from', var)
    if csp.fc:  # undo pruned values
        for pruned_var in csp.pruned[var]:
            csp.curr_domains[pruned_var[0]].append(pruned_var[1])
        csp.pruned[var].clear()
    return None


def select_unassigned_variable(assignment, csp):
    "Select the variable to work on next.  Find"
    if csp.mcv:  # Most Constrained Variable
        unassigned = [v for v in csp.vars if v not in assignment]
        return argmin_random_tie(unassigned,
                                 lambda var: -num_legal_values(csp, var, assignment))
    if csp.mrv:  # Minimum Remaining Values
        unassigned = [v for v in csp.vars if v not in assignment]
        mins = argmin_list(unassigned, lambda var: len(csp.curr_domains[var]))
        return mins[0]
    else:  # First unassigned variable
        for v in csp.vars:
            if v not in assignment:
                return v


def order_domain_values(var, assignment, csp):
    "Decide what order to consider the domain variables."
    if csp.curr_domains:
        domain = csp.curr_domains[var]
    else:
        domain = csp.domains[var][:]
    if csp.lcv:
        # If LCV is specified, consider values with fewer conflicts first
        key = lambda val: csp.nconflicts(var, val, assignment)
        #domain.sort(lambda(x,y): cmp(key(x), key(y)))
    while domain:
        yield domain.pop()

def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count_if(lambda val: csp.nconflicts(var, val, assignment) == 0,
                        csp.domains[var])


# Map-Coloring Problems

class UniversalDict:
    """A universal dict maps any key to the same value. We use it here
    as the domains dict for CSPs in which all vars have the same domain.
    >>> d = UniversalDict(42)
    >>> d['life']
    42
    """
    def __init__(self, value): self.value = value
    def __getitem__(self, key): return self.value
    def __repr__(self): return '{Any: %r}' % self.value

def different_values_constraint(A, a, B, b):
    "A constraint saying two neighboring variables must differ in value."
    global zVariables
    if 'Z' in A:
        return a[zVariables[A][0].index(B)] == b
    else:
        return b[zVariables[B][0].index(A)] == a


def graphLabelingCSP(numbers, node_shapes):
    """Make a CSP for the problem of coloring a map with different colors
    for any two adjacent regions.  Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries.  This dict may also be
    specified as a string of the form defined by parse_neighbors"""

    if isinstance(node_shapes, str):
        node_shapes = parse_neighbors(node_shapes)
    return CSP(node_shapes.keys(), numbers, node_shapes,
               different_values_constraint)

def parse_neighbors(neighbors, vars=[]):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors.  The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name.  If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z')
    {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    """
    dict = DefaultDict([])
    for var in vars:
        dict[var] = []
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (x, xneighbors) in specs:
        x = x.strip()
        dict.setdefault(x, [])
        for B in xneighbors.split():
            dict[x].append(B)
            dict[B].append(x)
    return dict
def AC3(csp, queue=None):  ####
    """[Fig. 5.7]"""
    if queue == None:
        queue = [(Xi, Xk) for Xi in csp.vars for Xk in csp.neighbors[Xi]]
    while queue:
        (Xi, Xj) = queue.pop()
        if remove_inconsistent_values(csp, Xi, Xj):
            for Xk in csp.neighbors[Xi]:
                queue.append((Xk, Xi))

def remove_inconsistent_values(csp, Xi, Xj):
    "Return true if we remove a value."
    removed = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        if every(lambda y: not csp.constraints(Xi, x, Xj, y),
                csp.curr_domains[Xj]):
            csp.curr_domains[Xi].remove(x)
            removed = True
    return removed

zVariables = {}
if __name__ == '__main__':
    print('Input:')
    V = int(input())
    E = int(input())
    node_shapes = input().split()
    # create input graph
    neighborss = ''
    for _ in range(E):
        i, j = map(int, input().split())
        # print("i",i)
        # print("j",j)
        neighborss += node_shapes[i] + str(i) + ':' + node_shapes[j] + str(j) + ';'
        # print("neighbor_string",neighbor_string)
    neighborss = neighborss[:-1]
    for node, adjacents in parse_neighbors(neighborss).items():
        if 'C' not in node:  # circle
            # binary constraints
            zVariables['Z' + str(len(zVariables))] = [node] + adjacents, [
                x for x in list(itertools.product(*[list(range(1, 10)) for _ in range(1 + len(adjacents))]))
                if str(x[0]) == str(product(x[1:]))[0]]if 'T' in node else [  # left
                x for x in list(itertools.product(*[list(range(1, 10)) for _ in range(1 + len(adjacents))]))
                if str(x[0]) == str(sum(x[1:]))[0]] if 'P' in node else [  # right
                x for x in list(itertools.product(*[list(range(1, 10)) for _ in range(1 + len(adjacents))]))
                if str(x[0]) == str(product(x[1:]))[-1]] if 'S' in node else [  # left sum
                x for x in list(itertools.product(*[list(range(1, 10)) for _ in range(1 + len(adjacents))]))
                if str(x[0]) == str(sum(x[1:]))[-1]] if 'H' in node else []  # right sum


    neighborss = ''
    for z in zVariables:
        neighborss += z + ':' + ' '.join(zVariables[z][0]) + ';'
    neighborss = neighborss[:-1]


    node_domain = {n: list(range(1, 10)) if 'Z' not in n else zVariables[n][1].copy()
                   for n in parse_neighbors(neighborss)}

    problem = graphLabelingCSP(node_domain, neighborss)
    solutions = backtracking_search(problem, mcv=False, lcv=False, fc=True, mac=False, mrv=True)

    solutions = {k[1:]: v for k, v in solutions.items() if 'Z' not in k}

    print('Output:')
    print('...'.join(str(solutions[str(i)]) for i in range(V)))
    #print(type(adjacents))