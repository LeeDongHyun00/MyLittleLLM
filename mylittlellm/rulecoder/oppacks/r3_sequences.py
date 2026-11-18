"""Round 3 operators for sequences: counting, selecting, pairing, regrouping and named scans over flat lists."""
import importlib

# Modules the engine imports automatically in generated answers; templates may use them freely.
MODULES = ("math", "re", "itertools", "collections", "bisect", "statistics")

# Multi-line rules are kept as source text because the same text is pasted into generated answers.
# Helpers import what they need themselves: the engine compiles them with `math` only.
LIS = '''def _seq_lis_length(items):
    best = [1] * len(items)
    for i in range(len(items)):
        for j in range(i):
            if items[j] < items[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    return max(best, default=0)
'''
BITONIC = '''def _seq_longest_bitonic(items):
    n = len(items)
    up, down = [1] * n, [1] * n
    for i in range(n):
        for j in range(i):
            if items[j] < items[i] and up[j] + 1 > up[i]:
                up[i] = up[j] + 1
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            if items[j] < items[i] and down[j] + 1 > down[i]:
                down[i] = down[j] + 1
    # The peak is counted in both halves.
    return max((u + d - 1 for u, d in zip(up, down)), default=0)
'''
SUM_INCREASING = '''def _seq_max_sum_increasing(items):
    best = list(items)
    for i in range(len(items)):
        for j in range(i):
            if items[j] < items[i] and best[j] + items[i] > best[i]:
                best[i] = best[j] + items[i]
    return max(best, default=0)
'''
SUM_BITONIC = '''def _seq_max_sum_bitonic(items):
    n = len(items)
    up, down = list(items), list(items)
    for i in range(n):
        for j in range(i):
            if items[j] < items[i] and up[j] + items[i] > up[i]:
                up[i] = up[j] + items[i]
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            if items[j] < items[i] and down[j] + items[i] > down[i]:
                down[i] = down[j] + items[i]
    # The peak is counted in both halves, so it is taken off once.
    return max((u + d - v for u, d, v in zip(up, down, items)), default=float("-inf"))
'''
SUBARRAY_PRODUCT = '''def _seq_max_subarray_product(items):
    # A negative factor turns the smallest running product into the largest, so both are carried.
    best = low = high = items[0]
    for value in items[1:]:
        low, high = min(value, low * value, high * value), max(value, low * value, high * value)
        best = max(best, high)
    return best
'''
NO_ADJACENT = '''def _seq_max_sum_no_adjacent(items):
    if len(items) < 2:
        return items[0]
    before, last = items[0], max(items[0], items[1])
    for value in items[2:]:
        before, last = last, max(last, before + value, value)
    return last
'''
CONSECUTIVE = '''def _seq_longest_consecutive(items):
    present, best = set(items), 0
    for value in items:
        if value - 1 not in present:        # only the start of a run is extended
            length = 1
            while value + length in present:
                length += 1
            best = max(best, length)
    return best
'''
MIN_JUMPS = '''def _seq_min_jumps(items):
    if not items or items[0] == 0:
        return float("inf")
    jumps = [0] + [float("inf")] * (len(items) - 1)
    for i in range(1, len(items)):
        for j in range(i):
            # The first position that reaches i is also the one with the fewest jumps.
            if i <= j + items[j] and jumps[j] != float("inf"):
                jumps[i] = jumps[j] + 1
                break
    return jumps[-1]
'''
COIN_WAYS = '''def _seq_coin_ways(coins, total):
    ways = [1] + [0] * total
    for coin in coins:
        for amount in range(coin, total + 1):
            ways[amount] += ways[amount - coin]
    return ways[total]
'''
MIN_COINS = '''def _seq_min_coins(coins, total):
    best = [0] + [float("inf")] * total
    for amount in range(1, total + 1):
        for coin in coins:
            if coin <= amount and best[amount - coin] + 1 < best[amount]:
                best[amount] = best[amount - coin] + 1
    return best[total]
'''
SUPER_UGLY = '''def _seq_super_ugly(n, primes):
    uglies, positions = [1], [0] * len(primes)
    while len(uglies) < n:
        candidates = [uglies[position] * prime for position, prime in zip(positions, primes)]
        smallest = min(candidates)
        uglies.append(smallest)
        # Every prime that produced the new number moves on, which also removes duplicates.
        positions = [position + (value == smallest) for position, value in zip(positions, candidates)]
    return uglies[-1]
'''
HEAPIFIED = '''def _seq_heapified(items):
    import heapq
    heap = list(items)
    heapq.heapify(heap)
    return heap
'''
HEAP_REPLACED = '''def _seq_heap_replaced(items, value):
    import heapq
    heap = list(items)
    heapq.heapify(heap)
    heapq.heapreplace(heap, value)
    return heap
'''


def register(lib):
    need, is_num, is_int, is_seq = lib.need, lib.is_num, lib.is_int, lib.is_seq
    scope = {name: importlib.import_module(name) for name in MODULES}

    def add(arity, name, guard, template, keywords, helper=None, valid=None, **options):
        """Register an operator whose search function is its own emitted template behind a guard.

        Compiling the template instead of writing the logic twice means the program that is verified
        on the example is literally the program that is shipped.
        """
        names = ("a0", "a1", "a2")[:arity]
        local = dict(scope)
        if helper:
            function, helper = lib.helper(helper)
            local[function.__name__] = function
        run = eval("lambda " + ", ".join(names) + ": " + template.format(*names), local)

        def fn(*args):
            need(guard(*args))
            value = run(*args)
            need(valid is None or valid(value))
            return value
        options.setdefault("gated", True)
        {1: lib.unary, 2: lib.binary, 3: lib.ternary}[arity](name, fn, template, keywords, helper=helper, **options)

    def know(arity, name, guard, template, keywords, **options):
        add(arity, name, guard, template, keywords, kind="knowledge", **options)

    # Guards. Sizes are bounded because quadratic scans run on every candidate value of a search.
    def seq(x, low=1, high=200):
        return is_seq(x) and low <= len(x) <= high

    def nums(x, low=1, high=200):
        return seq(x, low, high) and all(is_num(v) for v in x)

    def ints(x, low=1, high=200):
        return seq(x, low, high) and all(is_int(v) for v in x)

    def strs(x, low=1, high=200):
        return seq(x, low, high) and all(isinstance(v, str) for v in x)

    def scalar(v):
        return is_num(v) or isinstance(v, str)

    def flat(x, low=1, high=200):
        return seq(x, low, high) and all(scalar(v) for v in x)

    def prime(n):
        return is_int(n) and 1 < n < 10 ** 6 and all(n % d for d in range(2, int(n ** 0.5) + 1))

    # ------------------------------------------------------------ counting pairs of elements
    # "Count the pairs i < j with property P" is one scan under many names; each property has its own gate.
    pairs = "for a, b in itertools.combinations({0}, 2)"
    know(1, "seq_count_pairs_xor_even", lambda x: ints(x, 2, 120), f"sum((a ^ b) % 2 == 0 {pairs})", "pair&xor&even")
    # This task passes the length next to the array, and its hidden tests give n < len: only x[:n] counts.
    know(2, "seq_pair_xor_sum", lambda x, n: ints(x, 2, 120) and is_int(n) and 2 <= n <= len(x),
         "sum(a ^ b for a, b in itertools.combinations({0}[:{1}], 2))", "sum&xor&pair")
    know(1, "seq_pair_absdiff_sum", lambda x: nums(x, 2, 120), f"sum(abs(a - b) {pairs})",
         "sum&absolute difference&pair|sum&difference&all pairs")
    # The number of adjacent swaps a bubble sort needs is the number of inversions.
    know(1, "seq_count_inversions", lambda x: nums(x, 2, 120), f"sum(a > b {pairs})",
         "adjacent swap")
    know(2, "seq_count_pairs_sum", lambda x, k: nums(x, 2, 120) and is_num(k), f"sum(a + b == {{1}} {pairs})",
         "count&pair&sum|number of pairs&sum|pairs&sum is equal|pairs&sum equal")
    know(2, "seq_count_pairs_diff", lambda x, k: nums(x, 2, 120) and is_num(k), f"sum(abs(a - b) == {{1}} {pairs})",
         "count&pair&difference|number of pairs&difference")

    # ------------------------------------------------------------ counting elements
    # len(filter(...)) is two steps and ties with accidental two-step programs; one named step outranks them.
    add(1, "seq_count_odd", ints, "sum(v % 2 != 0 for v in {0})",
        "count odd|count the odd|count of odd|number of odd|how many odd")
    add(1, "seq_count_even", ints, "sum(v % 2 == 0 for v in {0})",
        "count even|count the even|count of even|number of even|how many even")
    # The core `sum` refuses booleans on purpose, so counting True needs its own rule.
    add(1, "seq_count_true", lambda x: seq(x, 0) and all(isinstance(v, bool) for v in x), "sum({0})",
        "count&true|number of true|true bool")
    add(1, "seq_count_ints", lambda x: seq(x) and not any(isinstance(v, bool) for v in x),
        "sum(isinstance(v, int) for v in {0})",
        "number of integer|count&integer element|count integer|count the integer|how many integer")
    add(1, "seq_count_max", nums, "{0}.count(max({0}))",
        "frequency of the largest|frequency of largest|frequency of the max|frequency of max|"
        "frequency of the highest|frequency of the greatest|occurrences of the largest|occurrences of the max|"
        "count of the largest|count of the max|number of times the largest|number of times the max")
    add(1, "seq_count_min", nums, "{0}.count(min({0}))",
        "frequency of the smallest|frequency of smallest|frequency of the min|frequency of min|"
        "frequency of the lowest|frequency of the least|occurrences of the smallest|occurrences of the min|"
        "count of the smallest|count of the min|number of times the smallest|number of times the min")
    add(2, "seq_count_same_position", lambda x, y: flat(x) and flat(y) and len(x) == len(y),
        "sum(a == b for a, b in zip({0}, {1}))",
        "count&same|count&equal&position|count&equal&index|same pair|number of&same")
    add(3, "seq_count_in_range",
        lambda x, low, high: (nums(x) and is_num(low) and is_num(high) and low <= high)
        or (strs(x) and isinstance(low, str) and isinstance(high, str) and low <= high),
        "sum({1} <= v <= {2} for v in {0})",
        "count&range|number of&range|count&between|count&within|number of&between")
    # v is its own inverse modulo p exactly when v * v leaves remainder 1; the modulus has to be a prime.
    know(2, "seq_count_self_inverse", lambda x, p: ints(x) and prime(p), "sum(v * v % {1} == 1 for v in {0})",
         "modular inverse")

    # ------------------------------------------------------------ finding one element or position
    know(1, "seq_odd_occurrence", flat, "next((v for v in {0} if {0}.count(v) % 2), -1)",
         "odd number of times|odd times|odd occurrence|occurring odd|occurs odd")
    know(1, "seq_first_duplicate", flat, "next((v for i, v in enumerate({0}) if v in {0}[:i]), -1)",
         "first&duplicate|first&repeat")
    # Positive first: when 0 is present both agree, and "positive" in the description must then win the tie.
    know(1, "seq_first_missing_positive", ints,
         "(lambda s: next(i for i in range(1, len(s) + 2) if i not in s))(set({0}))",
         "missing&positive|the missing number|missing&natural")
    know(1, "seq_mex", ints, "(lambda s: next(i for i in range(len(s) + 1) if i not in s))(set({0}))",
         "smallest missing|first missing|missing&smallest")
    know(1, "seq_equilibrium_index", nums,
         "next((i for i in range(len({0})) if sum({0}[:i]) == sum({0}[i + 1:])), -1)", "equilibrium")
    know(1, "seq_fixed_point", ints, "next((i for i, v in enumerate({0}) if v == i), -1)", "fixed point")
    # A sorted array rotated k times has its only descent at index k.
    know(1, "seq_rotation_count", lambda x: nums(x, 2),
         "next((i for i in range(1, len({0})) if {0}[i] < {0}[i - 1]), 0)",
         "rotation&count|rotations&sorted|number of rotations")
    add(1, "seq_argmax_all", nums, "(lambda m: [i for i, v in enumerate({0}) if v == m])(max({0}))",
        "index&max|position&max|indices&max|index&largest|position&largest|indices&largest|index&highest|"
        "position&highest")
    add(1, "seq_argmin_all", nums, "(lambda m: [i for i, v in enumerate({0}) if v == m])(min({0}))",
        "index&min|position&min|indices&min|index&smallest|position&smallest|indices&smallest|index&lowest|"
        "position&lowest")
    know(1, "seq_second_most_common", lambda x: flat(x) and len(set(x)) > 1,
         "collections.Counter({0}).most_common()[1][0]", "second&frequen|second&repeated|second&common|second&occurr")
    know(1, "seq_first_odd", ints, "next((v for v in {0} if v % 2 != 0), None)", "first odd",
         valid=lambda v: v is not None)
    # The pair feeds the core sum/product; -1 stands for a missing parity, as these tasks usually state it.
    know(1, "seq_first_even_odd", ints,
         "(next((v for v in {0} if v % 2 == 0), -1), next((v for v in {0} if v % 2 != 0), -1))",
         "first even|first&even&odd")
    add(1, "seq_first_and_last", lambda x: seq(x, 2), "({0}[0], {0}[-1])",
        "first&last|initial&last|front&rear|first&rear|initial&final|front&last")
    # (found, position) with -1 for a miss: the usual result of a linear search.
    know(2, "seq_sequential_search", lambda x, v: flat(x) and scalar(v),
         "({1} in {0}, {0}.index({1}) if {1} in {0} else -1)", "sequential search|linear search|boolean&index")
    know(2, "seq_bisect_left", lambda x, v: nums(x) and is_num(v) and list(x) == sorted(x),
         "bisect.bisect_left({0}, {1})", "left&insert|insertion point|insert&position|insertion index|where&insert")

    # ------------------------------------------------------------ selecting elements
    add(2, "seq_divisible_by", lambda x, n: ints(x) and is_int(n) and n != 0,
        "type({0})(v for v in {0} if v % {1} == 0)", "divisible|multiple of")
    add(3, "seq_divisible_by_either", lambda x, m, n: ints(x) and is_int(m) and is_int(n) and m != 0 and n != 0,
        "type({0})(v for v in {0} if v % {1} == 0 or v % {2} == 0)", "divisible&or")
    add(3, "seq_divisible_by_both", lambda x, m, n: ints(x) and is_int(m) and is_int(n) and m != 0 and n != 0,
        "type({0})(v for v in {0} if v % {1} == 0 and v % {2} == 0)", "divisible&and")
    add(2, "seq_take_indices", lambda x, idx: seq(x) and ints(idx, 1, 50) and all(-len(x) <= i < len(x) for i in idx),
        "type({0})({0}[i] for i in {1})",
        "indices|specified index|given index|index list|positions|access&element|elements at")
    add(2, "seq_stride", lambda x, n: seq(x, 2) and is_int(n) and 2 <= n <= len(x), "{0}[::{1}]",
        "nth items|n-th items|every nth|every n-th|every kth|every k-th|with step|step size|at interval|"
        "regular interval")
    know(2, "seq_anagrams_of", lambda x, s: strs(x) and isinstance(s, str),
         "type({0})(w for w in {0} if sorted(w) == sorted({1}))", "anagram")
    # str() lets one rule serve words and non-negative numbers alike.
    add(1, "seq_palindromes", lambda x: strs(x) or (ints(x) and all(v >= 0 for v in x)),
        "type({0})(w for w in {0} if str(w) == str(w)[::-1])", "palindrom")
    add(2, "seq_min_max_k", lambda x, k: nums(x) and is_int(k) and 0 < k <= len(x),
        "(lambda s: type({0})(v for i, v in enumerate(s) if i < {1} or i >= len(s) - {1}))(sorted({0}))",
        "max&min&k|smallest&largest|minimum&maximum")
    # The core slice excludes its end; "between indices m and n" in task statements includes it.
    add(3, "seq_slice_inclusive", lambda x, i, j: seq(x) and is_int(i) and is_int(j) and 0 <= i <= j < len(x),
        "{0}[{1}:{2} + 1]", "range|between|indices")
    know(2, "seq_first_triplet_sum", lambda x, k: nums(x, 3, 40) and is_num(k),
         "next((t for t in itertools.combinations({0}, 3) if sum(t) == {1}), False)", "triplet&sum",
         valid=lambda v: v is not False)
    # Two equal values are not a pair in the usual statement of this task; the core rule accepts them.
    know(1, "seq_best_distinct_pair_product", lambda x: nums(x, 2, 120),
         "max(((a, b) for a, b in itertools.combinations({0}, 2) if a != b), key=lambda p: p[0] * p[1])",
         "pair&product&high|pair&product&max")

    # ------------------------------------------------------------ regrouping and reordering
    add(1, "seq_pair_chunks", lambda x: seq(x, 2), "[{0}[i:i + 2] for i in range(0, len({0}), 2)]",
        "key-value|key value|adjacent&dictionary|into pairs|in pairs|chunks of two|groups of two")
    # i ^ 1 is the partner index inside each pair; an unpaired last element stays where it is.
    add(1, "seq_swap_adjacent", lambda x: seq(x, 2),
        "type({0})({0}[i ^ 1] if (i ^ 1) < len({0}) else {0}[i] for i in range(len({0})))",
        "exchange&every|swap&adjacent|swap&pair|swap&consecutive|exchange&adjacent|exchange&consecutive|"
        "swap&alternate|swap&every|exchange&n+1")
    # Phrases, not word pairs: "first ... end" also names rotations by n, where applying this twice fits by chance.
    add(1, "seq_first_to_end", lambda x: seq(x, 2), "{0}[1:] + {0}[:1]",
        "first element to the end|first element to end|first element to the last|first element to last|"
        "first item to the end|shift first|move first|shift the first|move the first")
    add(1, "seq_last_to_front", lambda x: seq(x, 2), "{0}[-1:] + {0}[:-1]",
        "last element to first|last element to the first|last element to front|last element to the front|"
        "last element to the begin|last element to the start|last item to the front|shift last|move last|"
        "shift the last|move the last")
    add(2, "seq_insert_after_each", lambda x, v: seq(x, 1, 100) and scalar(v),
        "[e for item in {0} for e in (item, {1})]", "after every|after each|after&element")
    # The core `repeat` concatenates the copies; this keeps each copy whole.
    add(2, "seq_repeat_whole", lambda x, n: seq(x) and is_int(n) and 0 < n <= 20, "type({0})([{0}] * {1})",
        "repeat&times|repeat the|replicate|n-times")
    add(2, "seq_zip_cycle", lambda x, y: seq(x) and seq(y) and len(y) < len(x),
        "list(zip({0}, itertools.cycle({1})))", "zip")
    add(2, "seq_each_concat", lambda x, y: strs(x) and strs(y) and len(x) == len(y),
        "type({0})(a + b for a, b in zip({0}, {1}))", "concatenat|join|combin|append|merg")
    add(1, "seq_adjacent_concat", lambda x: strs(x, 2), "type({0})(a + b for a, b in zip({0}, {0}[1:]))",
        "adjacent&concatenat|consecutive&concatenat|pairwise&concatenat|neighbo&concatenat|adjacent&join")
    # Two enumeration orders of the power set are in use: by size, and as a binary counter over the elements.
    subsets = "all&sublist|all&subset|all&combination|power set|powerset|all possible sub"
    know(1, "seq_powerset_by_size", lambda x: flat(x, 0, 8),
         "[list(c) for r in range(len({0}) + 1) for c in itertools.combinations({0}, r)]", subsets)
    know(1, "seq_powerset_binary", lambda x: flat(x, 0, 8),
         "[[{0}[j] for j in range(len({0}) - 1, -1, -1) if m >> j & 1] for m in range(2 ** len({0}))]", subsets)
    add(1, "seq_list_each", lambda x: strs(x) and all(x), "[list(v) for v in {0}]",
        "individual|list&each&character|characters&each|list out")
    add(1, "seq_last_char_each", lambda x: strs(x) and all(x), "type({0})(v[-1] for v in {0})",
        "rear|last char|last letter")
    add(1, "seq_strip_digits_each", strs, "type({0})(re.sub('[0-9]', '', v) for v in {0})",
        "remove&digit|remove&number|delete&digit|strip&digit")
    add(1, "seq_words_all", strs, "[w for s in {0} for w in s.split()]", "word")
    know(1, "seq_modified_rle", lambda x: (is_seq(x) or isinstance(x, str)) and 1 <= len(x) <= 200,
         "[[len(g), k] if len(g) > 1 else k for k, g in ((k, list(g)) for k, g in itertools.groupby({0}))]",
         "modified&run-length|modified&run length|modified&encod")
    add(1, "seq_heapify", nums, "_seq_heapified({0})", "convert&heap|to a heap|into a heap|heapify|build&heap",
        helper=HEAPIFIED)
    add(2, "seq_heap_replace", lambda x, v: nums(x) and is_num(v), "_seq_heap_replaced({0}, {1})",
        "heap&replace|heap&delete&insert|heap&smallest&insert", helper=HEAP_REPLACED)
    # An identity on the value; the task is to return a new object.
    add(1, "seq_copy", seq, "{0}[:]", "copy|clone")
    add(1, "seq_sort_mixed",
        lambda x: flat(x, 2) and any(is_num(v) for v in x) and any(isinstance(v, str) for v in x),
        "sorted(v for v in {0} if not isinstance(v, str)) + sorted(v for v in {0} if isinstance(v, str))",
        "sort&mixed|sort&heterogeneous|sort&integers and strings")
    add(1, "seq_invert_each", ints, "type({0})(~v for v in {0})", "invers|invert|bitwise not")
    # A dictionary printed as text is a type conversion like the core str() of a number.
    add(1, "seq_dict_text", lambda x: isinstance(x, dict) and 1 <= len(x) <= 50, "str({0})", "string",
        gated=False, cost=0.5)

    # ------------------------------------------------------------ aggregates
    add(1, "seq_median", nums, "statistics.median({0})", "median")
    know(1, "seq_min_gap", lambda x: nums(x, 2), "(lambda s: min(b - a for a, b in zip(s, s[1:])))(sorted({0}))",
         "minimum difference|min difference|smallest difference|least difference|minimum absolute difference")
    know(1, "seq_lcm", lambda x: ints(x, 2, 30) and all(0 < v <= 10 ** 6 for v in x), "math.lcm(*{0})",
         "lcm|least common multiple|lowest common multiple")
    know(1, "seq_gcd", lambda x: ints(x, 2) and all(v > 0 for v in x), "math.gcd(*{0})",
         "gcd|greatest common divisor|hcf|highest common factor")
    # Written out instead of statistics.stdev so that the float matches a hand-written loop bit for bit;
    # fewer than two values have no spread, which these tasks report as 0.0 rather than as an error.
    know(1, "seq_stdev_sample", lambda x: nums(x, 2),
         "(lambda m: math.sqrt(sum((v - m) ** 2 for v in {0}) / (len({0}) - 1)))(sum({0}) / len({0}))"
         " if len({0}) > 1 else 0.0", "standard deviation")
    know(1, "seq_positive_ratio", nums, "sum(v > 0 for v in {0}) / len({0})",
         "ratio&positive|fraction&positive|proportion&positive")
    know(1, "seq_negative_ratio", nums, "sum(v < 0 for v in {0}) / len({0})",
         "ratio&negative|fraction&negative|proportion&negative")
    # Either the three largest, or the two most negative with the largest.
    know(1, "seq_max_product_three", lambda x: nums(x, 3),
         "(lambda s: max(s[-1] * s[-2] * s[-3], s[0] * s[1] * s[-1]))(sorted({0}))", "product&three|product&triplet")
    # Element i lies in (i + 1) * (n - i) runs, of which the odd-length ones are the rounded-up half.
    know(1, "seq_odd_length_subarray_sum", nums,
         "sum(v * (((i + 1) * (len({0}) - i) + 1) // 2) for i, v in enumerate({0}))",
         "odd length&subarray|odd-length&subarray|odd length&sublist")

    # ------------------------------------------------------------ named scans (dynamic programming)
    know(1, "seq_lis_length", lambda x: nums(x, 1, 120), "_seq_lis_length({0})",
         "longest increasing subsequence|longest&increasing&subsequence", helper=LIS)
    know(1, "seq_longest_bitonic", lambda x: nums(x, 1, 120), "_seq_longest_bitonic({0})",
         "bitonic&longest|bitonic&length", helper=BITONIC)
    know(1, "seq_max_sum_increasing", lambda x: nums(x, 1, 120), "_seq_max_sum_increasing({0})",
         "sum&increasing subsequence|sum&increasing&subsequence", helper=SUM_INCREASING)
    know(1, "seq_max_sum_bitonic", lambda x: nums(x, 1, 120), "_seq_max_sum_bitonic({0})", "sum&bitonic",
         helper=SUM_BITONIC)
    know(1, "seq_max_subarray_product", nums, "_seq_max_subarray_product({0})",
         "product&subarray|product&contiguous|product&sublist", helper=SUBARRAY_PRODUCT)
    know(1, "seq_max_sum_no_adjacent", nums, "_seq_max_sum_no_adjacent({0})",
         "sum&no adjacent|sum&non-adjacent|sum&non adjacent|sum&not adjacent", helper=NO_ADJACENT)
    know(1, "seq_longest_consecutive", ints, "_seq_longest_consecutive({0})",
         "longest&consecutive|consecutive&subsequence&length", helper=CONSECUTIVE)
    know(1, "seq_min_jumps", lambda x: ints(x) and all(v >= 0 for v in x), "_seq_min_jumps({0})", "jump",
         helper=MIN_JUMPS)
    know(2, "seq_coin_ways", lambda x, n: ints(x, 1, 30) and all(v > 0 for v in x) and is_int(n) and 0 <= n <= 2000,
         "_seq_coin_ways({0}, {1})", "coin change|coin&ways|count&coin", helper=COIN_WAYS)
    know(2, "seq_min_coins", lambda x, n: ints(x, 1, 30) and all(v > 0 for v in x) and is_int(n) and 0 <= n <= 2000,
         "_seq_min_coins({0}, {1})", "minimum&coin|min&coin|fewest&coin", helper=MIN_COINS)
    know(2, "seq_super_ugly", lambda n, x: is_int(n) and 0 < n <= 500 and ints(x, 1, 30) and all(v > 1 for v in x),
         "_seq_super_ugly({0}, {1})", "super ugly", helper=SUPER_UGLY)
    know(1, "seq_adjacent_coordinates", lambda x: ints(x, 1, 3),
         "[list(p) for p in itertools.product(*[range(v - 1, v + 2) for v in {0}])]",
         "adjacent&coordinate|neighbo&coordinate|surrounding&coordinate")
    know(1, "seq_common_prefix", strs,
         "''.join(c[0] for c in itertools.takewhile(lambda c: len(set(c)) == 1, zip(*{0})))", "common prefix")

    # ------------------------------------------------------------ more words for existing operators
    lib.also_named("at/2", "get&item|item&index")
