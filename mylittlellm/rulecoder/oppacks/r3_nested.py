"""Round 3 operators for nested data: mixed nesting, rows of records, grouping, dictionaries and grid paths."""
import heapq
import math

# Helpers are pasted into the generated answer, so they use builtins only and never modify their input.
FLATTEN = """def _deep_flatten(x):
    out = []
    for v in x:
        if isinstance(v, type(x)):
            out.extend(_deep_flatten(v))
        else:
            out.append(v)
    return out
"""
DEPTH = """def _nesting_depth(x):
    if isinstance(x, dict):
        return 1 + max([_nesting_depth(v) for v in x.values() if isinstance(v, dict)], default=0)
    if isinstance(x, (list, tuple)):
        return 1 + max([_nesting_depth(v) for v in x if isinstance(v, (list, tuple))], default=0)
    return 0
"""
EVENS = """def _deep_evens(x):
    return type(x)(_deep_evens(v) if isinstance(v, (list, tuple)) else v
                   for v in x if isinstance(v, (list, tuple)) or v % 2 == 0)
"""
ODDS = """def _deep_odds(x):
    return type(x)(_deep_odds(v) if isinstance(v, (list, tuple)) else v
                   for v in x if isinstance(v, (list, tuple)) or v % 2 != 0)
"""
FLOATS = """def _rows_to_float(x):
    def convert(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return type(x)(type(row)(convert(v) for v in row) for row in x)
"""
GROUP_FIRST = """def _group_by_first(pairs):
    groups = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return groups
"""
GROUP_SECOND = """def _group_by_second(pairs):
    groups = {}
    for value, key in pairs:
        groups.setdefault(key, []).append(value)
    return groups
"""
MERGE_FIRST = """def _merge_by_first(rows):
    merged = {}
    for row in rows:
        merged.setdefault(row[0], [row[0]]).extend(row[1:])
    return type(rows)(type(rows[0])(values) for values in merged.values())
"""
SUCCESSORS = """def _successors(pairs):
    out = {}
    for a, b in pairs:
        out.setdefault(b, [])
        out.setdefault(a, []).append(b)
    return out
"""
AVERAGE_PATH = """def _max_average_path(cost):
    best = []
    for i, row in enumerate(cost):
        best.append([])
        for j, value in enumerate(row):
            before = ([best[i - 1][j]] if i else []) + ([best[i][j - 1]] if j else [])
            best[i].append(value + max(before, default=0))
    return best[-1][-1] / (len(cost) + len(cost[0]) - 1)
"""
TRIANGLE = """def _triangle_path(tri, pick):
    best = list(tri[-1])
    for i in range(len(tri) - 2, -1, -1):
        best = [tri[i][j] + pick(best[j], best[j + 1]) for j in range(i + 1)]
    return best[0]
"""
MIN_COST = """def _min_cost_path(cost, m, n):
    best = []
    for i in range(m + 1):
        best.append([])
        for j in range(n + 1):
            before = ([best[i - 1][j]] if i else []) + ([best[i][j - 1]] if j else [])
            before += [best[i - 1][j - 1]] if i and j else []
            best[i].append(cost[i][j] + min(before, default=0))
    return best[m][n]
"""
GOLD = """def _gold_mine(gold):
    best = [row[-1] for row in gold]
    for col in range(len(gold[0]) - 2, -1, -1):
        best = [gold[r][col] + max(best[max(r - 1, 0):r + 2]) for r in range(len(gold))]
    return max(best)
"""
NONADJACENT = """def _grid_nonadjacent_max(grid):
    taken, skipped = 0, 0
    for column in zip(*grid):
        taken, skipped = skipped + max(column), max(taken, skipped)
    return max(taken, skipped)
"""
SMALLEST_RANGE = """def _smallest_range(lists):
    best = None
    for low in sorted(set(v for row in lists for v in row)):
        nexts = [min((v for v in row if v >= low), default=None) for row in lists]
        if None in nexts:
            break
        if best is None or max(nexts) - low < best[1] - best[0]:
            best = (low, max(nexts))
    return best
"""


def register(lib):
    need, is_num, is_int, is_seq, seqseq = lib.need, lib.is_num, lib.is_int, lib.is_seq, lib.seqseq
    scope = {"math": math, "heapq": heapq}     # what the templates below may name besides builtins and helpers

    def add(arity, name, template, guard, keywords, helper=None, **options):
        """Register an operator whose search function is its own emitted template behind a guard.

        Compiling the template makes the searched value and the shipped code the same computation;
        the guard only says where the operator is meaningful. Everything here is gated unless stated:
        an operator that fires on every nested argument costs every task accidental matches.
        """
        text = None
        if helper:
            compiled, text = lib.helper(helper)
            scope[compiled.__name__] = compiled
        names = ["a0", "a1", "a2"][:arity]
        run = eval("lambda " + ", ".join(names) + ": " + template.format(*names), scope)

        def fn(*args):
            need(guard(*args))
            return run(*args)
        options.setdefault("gated", True)
        {1: lib.unary, 2: lib.binary, 3: lib.ternary}[arity](name, fn, template, keywords, helper=text, **options)

    def both(first, second):
        """Keywords that need one word of each group: both('a|b', 'x|y') is 'a&x|a&y|b&x|b&y'."""
        return "|".join(f"{a}&{b}" for a in first.split("|") for b in second.split("|"))

    def rows(x, least=0):
        # The size bounds keep quadratic templates cheap while the search calls them thousands of times.
        return (seqseq(x) and len(x) <= 200 and all(least <= len(r) <= 200 for r in x)
                and sum(len(r) for r in x) <= 2000)

    def cells(x, check, least=0):
        return rows(x, least) and all(check(v) for r in x for v in r)

    def pairs(x):
        return rows(x, 2) and all(len(r) == 2 for r in x)

    def comparable(values):
        return all(is_num(v) for v in values) or all(isinstance(v, str) for v in values)

    def keyed(x):
        return rows(x, 2) and comparable([r[1] for r in x])

    def nested(v):
        return isinstance(v, (list, tuple))

    def scalar(v):
        return not isinstance(v, (list, tuple, dict, set))

    def records(x):
        return is_seq(x) and 0 < len(x) <= 200 and all(isinstance(d, dict) and d for d in x)

    def number_dict(x):
        return isinstance(x, dict) and 0 < len(x) <= 200 and all(is_num(v) for v in x.values())

    def grid(x):
        return cells(x, is_num, 1) and len(x) <= 40 and len({len(r) for r in x}) == 1 and len(x[0]) <= 40

    LOW = "min|small|low|least|cheap"
    HIGH = "max|larg|high|great|most|big|top|expens|best"
    COUNT = "count|number of|how many|frequen|occur|times"
    # Words that say a record is picked by one of its fields, not by an aggregate over all rows.
    FIELD = "second|2nd|value|record|with the|having|whose"

    # ------------------------------------------------------------ mixed and deep nesting
    def mixed_nesting(x):
        # Plain two-level input stays with the core `flatten`, so programs built on it keep their meaning.
        if not is_seq(x):
            return False
        inner = [v for v in x if isinstance(v, type(x))]
        return bool(inner) and (len(inner) < len(x) or any(isinstance(w, type(x)) for v in inner for w in v))

    # Ungated: arguments that mix scalars and sublists are rare, and descriptions name this in many ways.
    add(1, "deep_flatten", "_deep_flatten({0})", mixed_nesting, "flatten|nested|recurs", helper=FLATTEN, gated=False)
    add(1, "nesting_depth", "_nesting_depth({0})", lambda x: isinstance(x, (dict, list, tuple)),
        "depth|nesting|how deep", helper=DEPTH, kind="knowledge")

    def int_leaves(x, depth=0):
        return depth < 20 and all(int_leaves(v, depth + 1) if nested(v) else is_int(v) for v in x)

    def nested_ints(x):
        return is_seq(x) and any(nested(v) for v in x) and int_leaves(x)

    # The filter keeps the shape: a tuple inside a tuple is filtered in place, not flattened.
    add(1, "deep_evens", "_deep_evens({0})", nested_ints, "even|odd", helper=EVENS)
    add(1, "deep_odds", "_deep_odds({0})", nested_ints, "even|odd", helper=ODDS)
    # Only for mixed sequences while searching: on plain rows the answer is always 0, an accident waiting.
    add(1, "count_until_nested", "next((i for i, v in enumerate({0}) if isinstance(v, (list, tuple))), len({0}))",
        lambda x: is_seq(x) and any(nested(v) for v in x) and not all(nested(v) for v in x),
        "until|before|prior to|preced")

    def encoded(x):
        runs = [g for g in x if nested(g)] if is_seq(x) else []
        return (bool(runs) and all(len(g) == 2 and is_int(g[0]) and 0 <= g[0] <= 50 for g in runs)
                and len(x) <= 100)

    add(1, "decode_runs", "[v for g in {0} for v in ([g[1]] * g[0] if isinstance(g, (list, tuple)) else [g])]",
        encoded, "decode|run-length|run length|decompress", kind="knowledge")

    # ------------------------------------------------------------ rows chosen by a condition
    CONTAIN = "contain|includ|present in|having|that have|which have"
    add(2, "count_rows_with", "sum({1} in row for row in {0})", lambda x, v: rows(x) and scalar(v), CONTAIN)
    add(2, "rows_with", "[row for row in {0} if {1} in row]", lambda x, v: rows(x) and scalar(v), CONTAIN)
    add(2, "rows_without", "[row for row in {0} if {1} not in row]", lambda x, v: rows(x) and scalar(v),
        "remove|delete|drop|exclud|without|not contain|not hav")
    add(2, "rows_all_divisible", "[row for row in {0} if all(v % {1} == 0 for v in row)]",
        lambda x, k: cells(x, is_int) and is_int(k) and k != 0, "divisible|divid|multiple")
    add(3, "rows_within", "[row for row in {0} if all({1} <= v <= {2} for v in row)]",
        lambda x, low, high: cells(x, is_num) and is_num(low) and is_num(high) and low <= high,
        "range|between|within|outside|inside")
    add(1, "drop_all_none_rows", "[row for row in {0} if not all(v is None for v in row)]",
        lambda x: rows(x) and any(all(v is None for v in r) for r in x), "none|null|empty")
    add(1, "drop_rows_with_none", "[row for row in {0} if not any(v is None for v in row)]",
        lambda x: rows(x) and any(v is None for r in x for v in r), "none|null|empty")
    add(2, "rows_keep_common", "[type(row)(v for v in row if v in {1}) for row in {0}]",
        lambda x, other: rows(x) and is_seq(other) and 0 < len(other) <= 200, "present|common|intersect|belong")
    add(1, "distinct_rows_count", "len(set(map(tuple, {0})))", rows, "unique|distinct|different")
    # Lists cannot be dictionary keys, so counting sublists always goes through tuples.
    add(1, "count_sublists", "{{tuple(row): {0}.count(row) for row in {0}}}", rows, COUNT)
    add(1, "rows_with_count", "[(*row, {0}.count(row)) for row in dict.fromkeys({0})]",
        lambda x: rows(x) and all(isinstance(r, tuple) for r in x), "frequen|occurrence")

    # ------------------------------------------------------------ every row changed the same way
    add(1, "rows_to_float", "_rows_to_float({0})",
        lambda x: cells(x, lambda v: is_num(v) or isinstance(v, str)), "float", helper=FLOATS)
    add(1, "rows_to_int", "type({0})(type(row)(int(v) for v in row) for row in {0})",
        lambda x: cells(x, lambda v: isinstance(v, str) and v.strip().lstrip("+-").isdigit()),
        "integer|int value|convert")
    add(2, "shift_rows", "type({0})(type(row)(v + {1} for v in row) for row in {0})",
        lambda x, n: cells(x, is_num, 1) and is_num(n), "add|increment|increase|plus")
    add(2, "scale_rows", "type({0})(type(row)(v * {1} for v in row) for row in {0})",
        lambda x, n: cells(x, is_num, 1) and is_num(n), "multipl|times|scale|product|double")
    add(2, "concat_rows", "type({0})(a + b for a, b in zip({0}, {1}))",
        lambda x, y: rows(x) and rows(y) and len(x) == len(y) and all(type(a) is type(b) for a, b in zip(x, y)),
        "zip|concatenat|merge|join|combin|append|extend|add")
    # Written with two slices so that a negative column index means what `del row[n]` means.
    add(2, "drop_column", "type({0})(row[:{1}] + row[{1}:][1:] for row in {0})",
        lambda x, n: rows(x, 1) and is_int(n) and -min(len(r) for r in x) <= n < min(len(r) for r in x),
        "remove|delete|drop|exclud|without")
    add(2, "trim_rows", "type({0})(row[{1}:len(row) - {1}] for row in {0})",
        lambda x, k: rows(x) and is_int(k) and 0 < k and all(len(r) >= 2 * k for r in x), "trim")

    # ------------------------------------------------------------ records: (name, value) rows
    add(2, "k_min_records", "sorted({0}, key=lambda row: row[1])[:{1}]",
        lambda x, k: keyed(x) and is_int(k) and 0 < k <= len(x), LOW)
    add(2, "k_max_records", "sorted({0}, key=lambda row: row[1], reverse=True)[:{1}]",
        lambda x, k: keyed(x) and is_int(k) and 0 < k <= len(x), HIGH)
    add(1, "min_by_second", "min({0}, key=lambda row: row[1])", keyed, both(LOW, FIELD))
    add(1, "max_by_second", "max({0}, key=lambda row: row[1])", keyed, both(HIGH, FIELD))
    # "The name with the lowest score" in one step: as two steps it only ties with accidents such as
    # "the longest of the first fields", and the accident is found first.
    add(1, "first_of_min_by_second", "min({0}, key=lambda row: row[1])[0]", keyed, both(LOW, FIELD))
    add(1, "first_of_max_by_second", "max({0}, key=lambda row: row[1])[0]", keyed, both(HIGH, FIELD))
    add(1, "sort_by_last_desc", "sorted({0}, key=lambda row: row[-1], reverse=True)",
        lambda x: rows(x, 1) and len(x) > 1 and comparable([r[-1] for r in x]),
        "descend|decreas|revers|highest|largest")

    def numeric_text(v):
        try:
            return isinstance(v, str) and math.isfinite(float(v))
        except ValueError:
            return False

    def float_keyed(x):
        return rows(x, 2) and len(x) > 1 and all(numeric_text(r[1]) for r in x)

    add(1, "sort_by_float", "sorted({0}, key=lambda row: float(row[1]))", float_keyed, "float")
    add(1, "sort_by_float_desc", "sorted({0}, key=lambda row: float(row[1]), reverse=True)", float_keyed, "float")

    def ordered_keys(x, order):
        if not (rows(x, 1) and is_seq(order) and 0 < len(order) <= 200 and all(scalar(k) for k in order)):
            return False
        return all(any(r[0] == k for r in x) for k in order)

    add(2, "reorder_by_keys", "[row for k in {1} for row in {0} if row[0] == k]", ordered_keys,
        "arrang|order|sort|according|based on")
    add(1, "shortest_with_len", "(len(min({0}, key=len)), min({0}, key=len))",
        lambda x: seqseq(x) or lib.strseq(x), "minimum length|min length|shortest|smallest length|minimum&length")
    add(1, "longest_with_len", "(len(max({0}, key=len)), max({0}, key=len))",
        lambda x: seqseq(x) or lib.strseq(x), "maximum length|max length|longest|largest length|maximum&length")

    def listed(x):
        return rows(x, 2) and all(len(r) == 2 and lib.numseq(r[1]) for r in x)

    for call, words in (("max", HIGH), ("min", LOW), ("sum", "sum|total|add")):
        add(1, "record_" + call, "type({0})(type(row)([row[0], %s(row[1])]) for row in {0})" % call, listed, words)
    # One operator in place of min(pair_absdiff(..)): with two steps, `min` of each row is named just as
    # well by "minimum difference" and is found first.
    add(1, "pair_gap_min", "min(abs(a - b) for a, b in {0})", lambda x: pairs(x) and cells(x, is_num),
        both("difference|gap", LOW))
    add(1, "pair_gap_max", "max(abs(a - b) for a, b in {0})", lambda x: pairs(x) and cells(x, is_num),
        both("difference|gap", HIGH))

    # ------------------------------------------------------------ grouping pairs
    add(1, "group_by_first", "_group_by_first({0})", pairs, "group|dictionar|key|map|categor|cluster|collect",
        helper=GROUP_FIRST)
    add(1, "group_by_second", "_group_by_second({0})", pairs,
        "group|basis of|by second|by 2nd|by value|per value|same second|same value", helper=GROUP_SECOND)

    def repeated_first(x):
        if not (rows(x, 1) and len({type(r) for r in x}) == 1):
            return False
        return len({r[0] for r in x}) < len(x)

    add(1, "merge_by_first", "_merge_by_first({0})", repeated_first,
        "join|merge|combin|group|common|similar|same first|same initial|same key", helper=MERGE_FIRST)
    add(1, "successors_dict", "_successors({0})", pairs, "assign&pair|adjacen|neighbo|graph|edge|successor",
        helper=SUCCESSORS)
    add(1, "symmetric_pairs", "{{(a, b) for a, b in {0} if a < b and (b, a) in {0}}}",
        lambda x: pairs(x) and all(isinstance(r, tuple) for r in x) and cells(x, is_num), "symmetric",
        kind="knowledge")

    # ------------------------------------------------------------ dictionaries
    add(1, "drop_none_values", "{{k: v for k, v in {0}.items() if v is not None}}",
        lambda x: isinstance(x, dict) and any(v is None for v in x.values()),
        "empty|none|null|missing|drop|remove|filter|clean")

    def pair_valued(x, a, b):
        return (isinstance(x, dict) and 0 < len(x) <= 200 and is_num(a) and is_num(b)
                and all(is_seq(v) and len(v) == 2 and is_num(v[0]) and is_num(v[1]) for v in x.values()))

    add(3, "filter_dict_pairs", "{{k: v for k, v in {0}.items() if v[0] >= {1} and v[1] >= {2}}}", pair_valued,
        "filter|above|minim|at least|greater|more than|exceed")
    add(2, "add_dicts", "{{k: {0}.get(k, 0) + {1}.get(k, 0) for k in {{**{0}, **{1}}}}}",
        lambda x, y: number_dict(x) and number_dict(y), "add|sum|total|plus|accumul")
    add(1, "invert_dict", "{{v: k for k, v in {0}.items()}}",
        lambda x: isinstance(x, dict) and 0 < len(x) <= 200, "invert|inverse|swap|revers|exchang|interchang|flip")

    # Records as dictionaries: the field to order by is not an argument, so the last numeric field is used
    # (in a {'name': .., 'price': ..} record that is the price), as `sort_by_last` does for tuple records.
    numeric_field = "key=lambda d: [v for v in d.values() if isinstance(v, (int, float))][-1]"

    def priced(x, n):
        return (records(x) and is_int(n) and 0 < n <= len(x)
                and all(any(is_num(v) for v in d.values()) for d in x))

    add(2, "n_smallest_records", "heapq.nsmallest({1}, {0}, " + numeric_field + ")", priced, LOW)
    add(2, "n_largest_records", "heapq.nlargest({1}, {0}, " + numeric_field + ")", priced, HIGH)
    add(2, "sort_dicts_by_key", "sorted({0}, key=lambda d: d[{1}])",
        lambda x, k: records(x) and len(x) > 1 and isinstance(k, (str, int)) and not isinstance(k, bool)
        and all(k in d for d in x) and comparable([d[k] for d in x]), "sort|order|arrang")
    add(1, "sort_dicts_by_last", "sorted({0}, key=lambda d: list(d.values())[-1])",
        lambda x: records(x) and len(x) > 1 and comparable([list(d.values())[-1] for d in x]),
        both("sort|order|arrang", "dictionar"))

    # ------------------------------------------------------------ results printed as text
    # Several tasks return str(result) of a record list, and no description says so, so this cannot be
    # gated. It is limited to lists of tuples: text made from a flat list or a dictionary would let every
    # string operator reinterpret the digits and letters of an argument. It keeps the full cost on
    # purpose: as a half-price conversion, "some string operator applied to str(argument)" would outrank
    # a correct two-step program whenever the description names that string operator.
    lib.unary("records_to_text", lambda x: (need(isinstance(x, list) and 0 < len(x) <= 60
                                                 and all(isinstance(r, tuple) for r in x)), str(x))[1],
              "str({0})", "as a string|to a string|to string|string representation")

    # ------------------------------------------------------------ classic paths through a grid
    add(1, "max_average_path", "_max_average_path({0})", grid, "average&path", helper=AVERAGE_PATH, kind="knowledge")

    def triangle(x):
        return cells(x, is_num, 1) and len(x) <= 40 and all(len(r) > i for i, r in enumerate(x))

    add(1, "triangle_max_path", "_triangle_path({0}, max)", triangle, both("triangle", HIGH), helper=TRIANGLE,
        kind="knowledge")
    add(1, "triangle_min_path", "_triangle_path({0}, min)", triangle, both("triangle", LOW), helper=TRIANGLE,
        kind="knowledge")
    add(3, "min_cost_path", "_min_cost_path({0}, {1}, {2})",
        lambda x, m, n: grid(x) and is_int(m) and is_int(n) and 0 <= m < len(x) and 0 <= n < len(x[0]),
        "cost&path|minimum cost|min cost", helper=MIN_COST, kind="knowledge")
    add(1, "gold_mine", "_gold_mine({0})", grid, "gold", helper=GOLD, kind="knowledge")
    add(1, "grid_nonadjacent_max", "_grid_nonadjacent_max({0})", grid,
        "adjacent&grid|adjacent&rectang|adjacent&matrix", helper=NONADJACENT, kind="knowledge")
    add(1, "smallest_range", "_smallest_range({0})", lambda x: cells(x, is_num, 1) and sum(map(len, x)) <= 200,
        "smallest range|minimum range|shortest range|smallest interval", helper=SMALLEST_RANGE, kind="knowledge")
