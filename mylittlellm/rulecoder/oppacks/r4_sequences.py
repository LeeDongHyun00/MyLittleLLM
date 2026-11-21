"""Round 4 operators for sequences and nested data: reorderings, paired aggregates and named scans over lists."""
import importlib

# Modules the engine imports automatically in generated answers; templates and helpers may use them freely.
MODULES = ("collections", "heapq", "itertools", "math", "re")

# Multi-line rules are kept as source text because the same text is pasted into generated answers.
# Helper names start with _r4q_: packs are written in parallel, and two different helpers with one name
# in a single generated answer would silently replace each other.
ADJACENT_SUBSEQUENCE = '''def _r4q_longest_adjacent_subsequence(items, exact):
    # best[i] is the longest subsequence ending at i whose neighbours differ by at most one, or by
    # exactly one: in the exact reading equal values do not extend a subsequence.
    best = [1] * len(items)
    for i in range(len(items)):
        for j in range(i):
            gap = abs(items[i] - items[j])
            if (gap == 1 if exact else gap <= 1) and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    return max(best, default=0)
'''
NEGATIVES_FIRST = '''def _r4q_negatives_first(items):
    # The partition step of quicksort around 0: each negative is swapped with the first non-negative,
    # so the negatives keep their order and the non-negatives do not.
    out, front = list(items), 0
    for i in range(len(out)):
        if out[i] < 0:
            out[i], out[front] = out[front], out[i]
            front += 1
    return out
'''
PROFIT = '''def _r4q_max_profit(prices, k):
    # profit[j]: best total over prices[:j + 1] with the trades allowed so far; one more trade per round.
    profit = [0] * len(prices)
    for _ in range(k):
        best_buy, new = float("-inf"), [0] * len(prices)
        for j in range(1, len(prices)):
            best_buy = max(best_buy, profit[j - 1] - prices[j - 1])
            new[j] = max(new[j - 1], prices[j] + best_buy)
        profit = new
    return profit[-1] if profit else 0
'''
PRODUCT_BELOW = '''def _r4q_subsequences_product_below(items, k):
    # Non-empty subsequences counted by product. Without negative values a product of k or more never
    # comes back below k unless a 0 joins it, so those are pooled in one count and the table stays small.
    counts, large = {}, 0
    for v in items:
        grown, grown_large = dict(counts), large
        if v == 0:
            grown[0] = grown.get(0, 0) + sum(counts.values()) + large + 1
        else:
            for product, ways in counts.items():
                if product * v < k:
                    grown[product * v] = grown.get(product * v, 0) + ways
                else:
                    grown_large += ways
            grown_large += large
            if v < k:
                grown[v] = grown.get(v, 0) + 1
            else:
                grown_large += 1
        counts, large = grown, grown_large
    return sum(counts.values())
'''
REMOVALS = '''def _r4q_min_removals_within(items, k):
    # Whatever survives is a run of the sorted values, so keep the longest run spanning at most k.
    values, low, best = sorted(items), 0, 0
    for high, v in enumerate(values):
        while v - values[low] > k:
            low += 1
        best = max(best, high - low + 1)
    return len(values) - best
'''
PLATFORMS = '''def _r4q_min_platforms(arrivals, departures):
    # Sweep the sorted times; an arrival no later than the next departure needs another platform.
    arrive, leave = sorted(arrivals), sorted(departures)
    needed = best = 1
    i, j = 1, 0
    while i < len(arrive) and j < len(leave):
        if arrive[i] <= leave[j]:
            needed, i = needed + 1, i + 1
        else:
            needed, j = needed - 1, j + 1
        best = max(best, needed)
    return best
'''
CLOSEST_TRIPLE = '''def _r4q_closest_triple(a, b, c):
    # Three pointers over sorted lists: advancing the smallest of the three is the only move that can
    # shrink the spread; the first triple with the smallest spread is kept.
    i = j = k = 0
    best, found = None, None
    while i < len(a) and j < len(b) and k < len(c):
        low, high = min(a[i], b[j], c[k]), max(a[i], b[j], c[k])
        if best is None or high - low < best:
            best, found = high - low, (a[i], b[j], c[k])
            if best == 0:
                break
        if a[i] == low:
            i += 1
        elif b[j] == low:
            j += 1
        else:
            k += 1
    return found
'''
TOP_K_FREQUENT = '''def _r4q_top_k_frequent(rows, k):
    counts = {}
    for row in rows:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    # The usual size-k min-heap: a value replaces the least frequent one kept only when it occurs more
    # often, and the answer is read by emptying the heap from the least frequent up.
    heap = []
    for v, c in counts.items():
        if len(heap) < k:
            heap.append((c, v))
            if len(heap) == k:
                heapq.heapify(heap)
        elif c > heap[0][0]:
            heapq.heappop(heap)
            heapq.heappush(heap, (c, v))
    out = []
    while heap:
        out.append(heapq.heappop(heap)[1])
    return out
'''
GROUP_COUNT = '''def _r4q_group_first_with_count(pairs):
    groups = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    # Each group lists its distinct values once and ends with the number of rows it came from.
    return [(key, *dict.fromkeys(values), len(values)) for key, values in groups.items()]
'''


def register(lib):
    need, is_num, is_int, is_seq = lib.need, lib.is_num, lib.is_int, lib.is_seq
    scope = {name: importlib.import_module(name) for name in MODULES}

    def add(arity, name, guard, template, keywords, helper=None, **options):
        """Register an operator (gated unless said otherwise) whose search function is its own template.

        Compiling the emitted expression means the program verified on the example is literally the
        program that is shipped; the guard only says where the operator is meaningful and bounds the work.
        Helpers are compiled with the auto-imported modules in scope, the way the generated answer runs them.
        """
        if helper:
            exec(helper, scope)
        names = ("a0", "a1", "a2")[:arity]
        run = eval("lambda " + ", ".join(names) + ": " + template.format(*names), scope)

        def fn(*args):
            need(guard(*args))
            return run(*args)
        options.setdefault("gated", True)
        {1: lib.unary, 2: lib.binary, 3: lib.ternary}[arity](name, fn, template, keywords, helper=helper, **options)

    def know(arity, name, guard, template, keywords, **options):
        add(arity, name, guard, template, keywords, kind="knowledge", **options)

    # Guards. Sizes are bounded because the search calls every operator on thousands of candidate values.
    def seq(x, low=1, high=200):
        return is_seq(x) and low <= len(x) <= high

    def nums(x, low=1, high=200):
        return seq(x, low, high) and all(is_num(v) for v in x)

    def ints(x, low=1, high=200):
        return seq(x, low, high) and all(is_int(v) for v in x)

    def strs(x, low=1, high=200):
        return seq(x, low, high) and all(isinstance(v, str) for v in x)

    def scalars(x, low=1, high=200):
        return seq(x, low, high) and all(is_num(v) or isinstance(v, str) for v in x)

    def signed(x):
        # Sign-based reorderings say nothing about a list that has only one sign.
        return nums(x, 2) and any(v < 0 for v in x) and any(v >= 0 for v in x)

    def ascending(x):
        return all(a <= b for a, b in zip(x, x[1:]))

    def is_count(x, k):
        # Tasks translated from C pass the array length next to the array; an integer equal to that
        # length is taken to be the length, not the count or step the operator is asked about.
        return is_int(k) and k != len(x)

    # ------------------------------------------------------------ reordering by sign
    know(1, "negatives_first_swap_r4q", signed, "type({0})(_r4q_negatives_first({0}))",
         "negative&before|negative&beginning|negative&front", helper=NEGATIVES_FIRST)
    # "Alternating" starts with either sign in task statements; the example decides which one is meant.
    # Values left over once one sign runs out keep their order at the end.
    add(1, "alternate_neg_pos_r4q", signed,
        "(lambda neg, pos: type({0})(v for i in range(max(len(neg), len(pos))) for v in neg[i:i + 1] + pos[i:i + 1]))"
        "([v for v in {0} if v < 0], [v for v in {0} if v >= 0])", "alternat&negative")
    add(1, "alternate_pos_neg_r4q", signed,
        "(lambda pos, neg: type({0})(v for i in range(max(len(pos), len(neg))) for v in pos[i:i + 1] + neg[i:i + 1]))"
        "([v for v in {0} if v >= 0], [v for v in {0} if v < 0])", "alternat&negative")
    # Positives ascending, then zeros, then negatives ascending: the order of the key -1 / v.
    add(1, "positives_then_negatives_r4q", lambda x: signed(x) and any(v > 0 for v in x),
        "sorted({0}, key=lambda v: (v <= 0, v < 0, v))", "arrang&positive&negative")

    # ------------------------------------------------------------ aggregates over one list
    # Both orders of the pair are in use (0-based even positions first, or 1-based even positions first).
    add(1, "alternate_sums_r4q", lambda x: nums(x, 2), "(sum({0}[::2]), sum({0}[1::2]))", "alternate&sum")
    add(1, "alternate_sums_odd_first_r4q", lambda x: nums(x, 2), "(sum({0}[1::2]), sum({0}[::2]))", "alternate&sum")
    add(1, "frequency_spread_r4q", lambda x: scalars(x, 1, 500),
        "(lambda counts: max(counts) - min(counts))(list(collections.Counter({0}).values()))", "difference&frequen")
    # One step each: as len(shortest(x)) the two-step program ties with len(x) and loses to it.
    add(1, "shortest_length_r4q", lambda x: seq(x, 1, 500) and all(isinstance(v, (str, list, tuple)) for v in x),
        "min(len(v) for v in {0})",
        "length of the shortest|length of shortest|shortest word|minimum length|min length|smallest length")
    add(1, "longest_length_r4q", lambda x: seq(x, 1, 500) and all(isinstance(v, (str, list, tuple)) for v in x),
        "max(len(v) for v in {0})",
        "length of the longest|length of longest|longest word|maximum length|max length|largest length")
    add(1, "run_values_and_counts_r4q", lambda x: seq(x, 1, 500),
        "(lambda runs: ([k for k, _ in runs], [n for _, n in runs]))"
        "([(k, len(list(g))) for k, g in itertools.groupby({0})])",
        "consecutive&frequen|consecutive&count|consecutive duplicate")
    add(2, "first_with_count_r4q", lambda x, k: scalars(x, 1, 500) and is_count(x, k) and k >= 1,
        "(lambda counts: next((v for v in {0} if counts[v] == {1}), -1))(collections.Counter({0}))",
        "first element occurring|occurring k times|occurs k times|first&occurring&times|first&appear&times")
    # Recursive search signatures pass the bounds of the part searched (low, high), both inclusive.
    add(3, "max_between_r4q", lambda x, i, j: nums(x) and is_int(i) and is_int(j) and 0 <= i <= j < len(x),
        "max({0}[{1}:{2} + 1])", "max&rotated|max&between&ind")
    add(3, "min_between_r4q", lambda x, i, j: nums(x) and is_int(i) and is_int(j) and 0 <= i <= j < len(x),
        "min({0}[{1}:{2} + 1])", "min&rotated|min&between&ind")

    # ------------------------------------------------------------ named scans
    # Both readings of "the difference between adjacent elements"; the exact one is named by the longer
    # phrase "... is one", so it wins where both reproduce the example and the task says "is one".
    know(1, "longest_close_subsequence_r4q", lambda x: nums(x, 1, 120),
         "_r4q_longest_adjacent_subsequence({0}, False)", "subsequence&difference&adjacent",
         helper=ADJACENT_SUBSEQUENCE)
    know(1, "longest_one_apart_subsequence_r4q", lambda x: nums(x, 1, 120),
         "_r4q_longest_adjacent_subsequence({0}, True)",
         "subsequence&difference&adjacent&is one|subsequence&difference&adjacent&of one"
         "|subsequence&difference&adjacent&is 1|subsequence&adjacent&differ by one", helper=ADJACENT_SUBSEQUENCE)
    # Raising every element to the maximum in steps of k; -1 when some gap is not a multiple of k.
    know(2, "min_increments_to_equal_r4q", lambda x, k: ints(x) and is_count(x, k) and k > 0,
         "(lambda top: sum((top - v) // {1} for v in {0}) if all((top - v) % {1} == 0 for v in {0}) else -1)"
         "(max({0}))",
         "operations&make all&equal|operations required&equal|operations&all elements equal")
    know(2, "max_profit_k_trades_r4q", lambda p, k: nums(p, 1, 200) and is_int(k) and 0 <= k <= 100,
         "_r4q_max_profit({0}, {1})", "stock|profit&transaction", helper=PROFIT)
    know(2, "subsequences_product_below_r4q",
         lambda x, k: ints(x, 1, 30) and all(v >= 0 for v in x) and is_int(k) and 1 <= k <= 10 ** 4,
         "_r4q_subsequences_product_below({0}, {1})", "subsequence&product&smaller|subsequence&product&less",
         helper=PRODUCT_BELOW)
    know(2, "min_removals_within_r4q", lambda x, k: nums(x, 1, 500) and is_num(k) and k >= 0,
         "_r4q_min_removals_within({0}, {1})", "removed&amax|removed&max-min|removed&max - min",
         helper=REMOVALS)
    # The count of the first n elements: tasks translated from C pass the length, and a rule that ignores
    # it ties with accidental programs that use every argument.
    know(2, "rotation_count_upto_r4q", lambda x, n: nums(x, 2) and is_int(n) and 2 <= n <= len(x),
         "next((i for i in range(1, {1}) if {0}[i] < {0}[i - 1]), 0)",
         "rotation&count|rotations&sorted|number of rotations")
    know(2, "min_abs_diff_sum_r4q", lambda x, y: nums(x, 1, 500) and nums(y, 1, 500) and len(x) == len(y),
         "sum(abs(a - b) for a, b in zip(sorted({0}), sorted({1})))",
         "sum&absolute difference|sum of absolute|minimum sum&difference")
    know(2, "min_platforms_r4q", lambda a, d: nums(a, 1, 500) and nums(d, 1, 500) and len(a) == len(d),
         "_r4q_min_platforms({0}, {1})", "platform", helper=PLATFORMS)
    know(3, "closest_triple_r4q", lambda a, b, c: all(nums(v) and ascending(v) for v in (a, b, c)),
         "_r4q_closest_triple({0}, {1}, {2})", "three closest|closest&three", helper=CLOSEST_TRIPLE)
    # Ties are those of the classic heap: sums first, then the position in the first list, then the second.
    know(3, "k_smallest_pairs_r4q",
         lambda a, b, k: nums(a, 1, 60) and nums(b, 1, 60) and is_int(k) and 0 < k <= len(a) * len(b),
         "[list(p) for p in sorted(itertools.product({0}, {1}), key=sum)[:{2}]]",
         "k number of pairs|k pairs|smallest pairs|k smallest pair")
    know(2, "top_k_frequent_r4q",
         lambda x, k: seq(x, 1, 200) and all(ints(r, 0, 200) for r in x) and sum(map(len, x)) <= 2000
         and is_int(k) and 1 <= k <= 100,
         "_r4q_top_k_frequent({0}, {1})", "top k&frequen|k most frequent|frequent&heap", helper=TOP_K_FREQUENT)

    # ------------------------------------------------------------ strings inside lists
    # A predicate, so gated: any of the given literals occurs in the text.
    know(2, "any_literal_in_r4q", lambda p, t: strs(p, 1, 50) and all(p) and isinstance(t, str) and len(t) <= 600,
         "any(p in {1} for p in {0})", "literal|search&pattern|any&pattern")
    add(1, "matched_text_r4q", lambda x: isinstance(x, bool), "'Matched!' if {0} else 'Not Matched!'",
        "literal|match|regex|search")
    add(2, "drop_words_containing_r4q", lambda x, y: strs(x, 1, 100) and strs(y, 1, 50) and all(y),
        "type({0})(' '.join(w for w in s.split() if not any(p in w for p in {1})) for s in {0})",
        "remove&word|delete&word|drop&word")
    # Only ASCII digit strings are numbers here: '²'.isdigit() is true and int('²') fails.
    add(2, "increment_numeric_strings_r4q",
        lambda x, k: strs(x) and any(s.isdigit() and s.isascii() for s in x)
        and all(s.isascii() or not s.isdigit() for s in x) and is_int(k),
        "type({0})(str(int(s) + {1}) if s.isdigit() else s for s in {0})",
        "numeric&increment|numeric&add|increment&string|numeric value")
    add(1, "drop_parenthesized_r4q", lambda s: isinstance(s, str) and "(" in s and ")" in s and len(s) <= 600,
        "re.sub(r' ?\\([^)]+\\)', '', {0})", "parenthes")
    # A (whole part, fractional digits) pair read as one decimal number.
    add(1, "parts_to_float_r4q", lambda x: is_seq(x) and len(x) == 2 and all(is_int(v) and v >= 0 for v in x),
        "float('.'.join(str(v) for v in {0}))", "float")

    # ------------------------------------------------------------ nested values
    # A bare list is one list: a one-element tuple written as ([..]) is just that list in Python.
    add(1, "count_lists_r4q",
        lambda x: (isinstance(x, tuple) and any(isinstance(v, list) for v in x))
        or (isinstance(x, list) and len(x) > 0 and not any(is_seq(v) for v in x)),
        "sum(isinstance(v, list) for v in {0}) if isinstance({0}, tuple) else int(isinstance({0}, list))",
        "number of lists|count&lists|how many lists|lists present")
    add(1, "group_first_with_count_r4q",
        lambda x: seq(x, 2) and all(isinstance(r, tuple) and len(r) == 2 for r in x)
        and all(isinstance(v, (int, str)) for r in x for v in r),
        "_r4q_group_first_with_count({0})", "occurrence&first element|occurrence of first|group&count&first",
        helper=GROUP_COUNT)
    add(1, "flatten_columns_r4q",
        lambda x: seq(x) and all(seq(r) and all(seq(c, 1, 50) for c in r) for r in x)
        and len({len(c) for r in x for c in r}) == 1 and sum(map(len, x)) <= 500,
        "list(zip(*[v for row in {0} for v in row]))", "flatten&column|matrix&column")
    add(1, "sort_by_digit_count_r4q", lambda x: seq(x, 2) and all(ints(r, 0) for r in x),
        "sorted({0}, key=lambda row: sum(len(str(v)) for v in row))", "sort&digit")

    # ------------------------------------------------------------ more words for existing operators
    # "count ... under given prime number" also names the prime-counting rule, by a longer phrase.
    try:
        lib.also_named("seq_count_self_inverse/2", "modular inverse&prime number|modular inverse&equal to itself")
    except KeyError:
        pass        # the round-3 sequence pack is not loaded; nothing to extend
